"""R1Pro Isaac Sim adapter: data-driven scene, joint mapping, control, state.

This module is the only place that touches isaaclab/omni APIs for the robot.
It builds the scene from a :class:`SceneModel` (objects, cameras, contact
sensors, world/ground all come from the model -- nothing is hard-coded to the
pickplace task), discovers the articulation, exposes a name-based joint
mapping, writes ControlCommands, and reads back actual state as domain
Observations.

Passing ``scene=None`` builds the bare robot scene used by lightweight robot
smoke runs (robot + default camera + ground, no objects).

Note: PhysX DOF-velocity readback is a pseudo-reading on several R1Pro joints
(torso_joint1/2/3, arm_joint1/4/6 report constant +/-5..7 rad/s; verified in
the reference r1pro_datagen project). Use :meth:`joint_vel_estimate` (position
differences) for velocities.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from r1pro_data_gen.domain import (
    ContactEvent,
    ControlCommand,
    EntityState,
    GraspContext,
    Observation,
    SceneModel,
    object_surface_distance_m,
    object_xy_radius_m,
    object_vertical_extent_m,
)
from r1pro_data_gen.robot import (
    R1PRO_EFFORT_PLANNING_UTILIZATION,
    R1PRO_GRIPPER_FINGER_COLLISION_BOXES_LOCAL,
    R1PRO_GRIPPER_FINGER_PRISMATIC_AXIS_LOCAL,
    R1PRO_JOINT_GROUP_EXPR,
    R1PRO_RUNTIME_EFFORT_ABORT_PERSISTENCE_S,
    R1PRO_RUNTIME_EFFORT_ABORT_UTILIZATION,
    R1PRO_ROOT_HEIGHT_RISE_ABORT_M,
    R1PRO_ROOT_TILT_ABORT_RAD,
    R1PRO_WHEEL_CONTACT_LOSS_TIMEOUT_S,
    R1PRO_TORSO_EFFORT_LIMIT,
    arm_torque_by_joint,
    arm_velocity_by_joint,
    gripper_min_vertical_overlap_m,
)
from r1pro_data_gen.robot.joints import JointMapping
from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS

# Repository root = <repo>/src/r1pro_data_gen/simulation/isaac_sim/adapter.py.
_REPO_ROOT = Path(__file__).resolve().parents[4]
@dataclass(slots=True)
class AdapterCfg:
    """Options to construct the R1Pro scene in Isaac Sim.

    A :class:`SceneModel` (``scene``) drives everything -- objects, cameras,
    sensors, ground, and the robot asset path. ``usd_path`` is only used when
    ``scene`` is None (legacy bare-robot scene); ``camera_eye/target`` are the
    fallback camera when the scene declares none.
    """

    usd_path: Path | None = None
    device: str = "cuda:0"
    headless: bool = True
    width: int = 640
    height: int = 480
    fps: int = DEFAULT_VIDEO_FPS
    num_envs: int = 1
    env_spacing: float = 8.0
    camera_eye: tuple[float, float, float] = (3.5, -3.5, 2.5)
    camera_target: tuple[float, float, float] = (0.0, 0.0, 0.8)
    wheel_control: str = "position"
    """Wheel actuator mode: "position" (locked hold, Phase 2) or "velocity"
    (drive via velocity targets, Phase 3 chassis motion)."""
    scene: SceneModel | None = None
    task_progress_path: str | None = None
    """The data-driven scene; None builds the bare robot scene."""


def _robot_usd_path(cfg: AdapterCfg) -> Path:
    """Resolve the robot USDA path from the scene model or the explicit arg."""
    if cfg.scene is not None:
        return (_REPO_ROOT / cfg.scene.robot.asset).resolve()
    if cfg.usd_path is None:
        raise ValueError("AdapterCfg requires usd_path when scene is None")
    return cfg.usd_path.resolve()


def build_robot_articulation_cfg(usd_path: Path, wheel_control: str = "position"):
    """ArticulationCfg with GPU-verified hold-pose actuators."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    # All-zero neutral home: arms hang straight down, gravity-balanced
    # (reference-project verified: max cumulative drift < 0.008 rad).
    home_pos = {
        "steer_motor_joint.*": 0.0,
        "wheel_motor_joint.*": 0.0,
        "torso_joint.*": 0.0,
        "left_arm_joint.*": 0.0,
        "right_arm_joint.*": 0.0,
        "left_gripper_finger_joint.*": 0.0,
        "right_gripper_finger_joint.*": 0.0,
    }
    if wheel_control == "velocity":
        wheel_actuator = ImplicitActuatorCfg(
            joint_names_expr=["wheel_motor_joint.*"],
            stiffness=0.0,
            damping=50.0,
            velocity_limit_sim=10.0,
        )
    elif wheel_control == "position":
        wheel_actuator = ImplicitActuatorCfg(
            joint_names_expr=["wheel_motor_joint.*"],
            stiffness=500.0,
            damping=100.0,
        )
    else:
        raise ValueError(f"unsupported wheel_control mode: {wheel_control!r}")

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            # Contact reporters on all robot bodies so the finger contact
            # sensors can report forces (reference-project pattern).
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                # Sleep disabled: with sleep_threshold=0.005, slow interpolated
                # motions drop below the threshold and PhysX sleeps the
                # joints, freezing the drive at a mid-trajectory pose.
                sleep_threshold=0.0,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=home_pos),
        actuators={
            "steer": ImplicitActuatorCfg(
                joint_names_expr=["steer_motor_joint.*"],
                stiffness=500.0,
                damping=100.0,
            ),
            "wheel": wheel_actuator,
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["torso_joint.*"],
                # These are the validated hold gains for the supplied asset.
                # The physical effort limit remains the authored 100 N*m;
                # lowering the gains makes the torso drift even at zero wheel
                # command and eventually saturate its own safety monitor.
                stiffness=2000.0,
                damping=500.0,
                effort_limit_sim=R1PRO_TORSO_EFFORT_LIMIT,
            ),
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint.*"],
                # Compliant tracking gains: the arm torque limits (55/25/18 Nm)
                # saturate at 0.0275 rad of error under stiffness 2000, so the
                # drive runs in saturation during gravity-loaded poses (steady
                # error ~0.09 rad and mid-path overshoots -- the "weird"
                # motion). At 500/60 the PD stays online across the workspace
                # (reference project uses 80/10 impedance; 500 keeps the
                # tracking error in the ~0.05-0.1 rad band with the same
                # physics limits).
                stiffness=800.0,
                damping=80.0,
                effort_limit_sim=arm_torque_by_joint("left"),
                velocity_limit_sim=arm_velocity_by_joint("left"),
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_joint.*"],
                stiffness=800.0,
                damping=80.0,
                effort_limit_sim=arm_torque_by_joint("right"),
                velocity_limit_sim=arm_velocity_by_joint("right"),
            ),
            "left_gripper": ImplicitActuatorCfg(
                joint_names_expr=["left_gripper_finger_joint.*"],
                # Stiffness 500 (reference play_record default): the soft
                # 30/5 gains cannot hold a 0.1 kg cylinder (closing force
                # ~1 N). 500/50 gives a solid pinch grip.
                stiffness=500.0,
                damping=50.0,
                effort_limit_sim=100.0,
                velocity_limit_sim=0.25,
            ),
            "right_gripper": ImplicitActuatorCfg(
                joint_names_expr=["right_gripper_finger_joint.*"],
                stiffness=500.0,
                damping=50.0,
                effort_limit_sim=100.0,
                velocity_limit_sim=0.25,
            ),
        },
    )


def _object_spawn_cfg(obj):
    """Build the Isaac Lab spawn cfg for one scene object (no isaaclab import
    at module scope -- called from inside build_scene_cfg)."""
    import isaaclab.sim as sim_utils

    collision = sim_utils.CollisionPropertiesCfg()
    collision.collision_enabled = bool(obj.physics.collision_enabled)
    if obj.physics.contact_offset is not None:
        collision.contact_offset = obj.physics.contact_offset

    rigid_props = None
    mass_props = None
    material = None
    if obj.physics.kinematic:
        rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
    else:
        rigid_props = sim_utils.RigidBodyPropertiesCfg()
        if obj.physics.mass is not None:
            mass_props = sim_utils.MassPropertiesCfg(mass=obj.physics.mass)
    if obj.physics.static_friction is not None or obj.physics.dynamic_friction is not None:
        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=obj.physics.static_friction,
            dynamic_friction=obj.physics.dynamic_friction,
            friction_combine_mode=obj.physics.friction_combine,
        )
    visual = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(obj.visual.color))
    if obj.visual.roughness is not None:
        visual.roughness = obj.visual.roughness

    from r1pro_data_gen.domain import ObjectType

    if obj.type is ObjectType.CUBOID:
        return sim_utils.CuboidCfg(
            size=tuple(obj.size),
            collision_props=collision,
            rigid_props=rigid_props,
            mass_props=mass_props,
            physics_material=material,
            visual_material=visual,
        )
    return sim_utils.CylinderCfg(
        radius=obj.radius,
        height=obj.height,
        collision_props=collision,
        rigid_props=rigid_props,
        mass_props=mass_props,
        physics_material=material,
        visual_material=visual,
    )


def _object_prim_path(name: str) -> str:
    """/World/<Name> prim path for an object (unique, readable)."""
    return f"/World/{name.capitalize()}"


def _contact_filter_prim_paths(names: tuple[str, ...]) -> list[str]:
    """Resolve scene object names to absolute PhysX contact-filter paths.

    PhysX rejects suffix-only patterns such as ``*Cylinder$`` because contact
    filters must start at an absolute USD path. Scene objects are authored by
    this adapter at ``/World/<Name>``, so use exactly the same path contract.
    """
    return [_object_prim_path(name) for name in names]


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1.0e-9 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)


def _quat_inverse(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    return np.asarray([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def _quat_multiply_raw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _quat_normalize(_quat_multiply_raw(left, right))


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    unit_quat = _quat_normalize(quat)
    rotated = _quat_multiply_raw(
        _quat_multiply_raw(unit_quat, np.asarray([0.0, *vector], dtype=float)),
        _quat_inverse(unit_quat),
    )
    return rotated[1:]


def _predicted_closed_finger_contact(
    adapter,
    object_model,
    object_position: np.ndarray,
    side: str,
    joint_positions,
) -> dict[str, object]:
    """Check whether both fingers can contact the object after closing.

    The open finger-link origins are not the contact surfaces.  A window based
    only on those origins or on their conservative boxes can therefore become
    true while one jaw still misses the object.  This helper projects each
    measured link along its authored prismatic axis to the zero-opening pose
    and checks the supplied link mesh against the object's contact envelope.
    It is a prediction only; :meth:`GripperGrasp` still requires live filtered
    contact from both fingers before creating an attachment.
    """
    try:
        import hppfcl

        from r1pro_data_gen.methods.collision import (
            collision_mesh_for_body,
            object_obstacle,
        )

        object_position = np.asarray(object_position, dtype=float)
        if object_position.shape != (3,) or not np.all(np.isfinite(object_position)):
            return {"checked": False, "reason": "object position is invalid"}
        physics = getattr(object_model, "physics", None)
        contact_offset = max(
            0.0,
            float(getattr(physics, "contact_offset", 0.0) or 0.0),
        )
        obstacle = object_obstacle(object_model, contact_offset)
        obstacle_transform = hppfcl.Transform3f(object_position)
        axis_local = np.asarray(R1PRO_GRIPPER_FINGER_PRISMATIC_AXIS_LOCAL, dtype=float)
        if axis_local.shape != (3,) or not np.all(np.isfinite(axis_local)):
            return {"checked": False, "reason": "finger prismatic axis is invalid"}

        contacts: list[bool] = []
        records: list[dict[str, object]] = []
        for index in (1, 2):
            body_name = f"{side}_gripper_finger_link{index}"
            joint_name = f"{side}_gripper_finger_joint{index}"
            position, quaternion = adapter.body_pose(body_name)
            position = np.asarray(position, dtype=float)
            quaternion = _quat_normalize(np.asarray(quaternion, dtype=float))
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                return {"checked": False, "reason": "finger body position is invalid"}
            opening = float(joint_positions.get(joint_name, 0.0))
            if not np.isfinite(opening):
                return {"checked": False, "reason": "finger opening is invalid"}
            rotation = np.column_stack(
                [
                    _quat_rotate(quaternion, np.array([1.0, 0.0, 0.0])),
                    _quat_rotate(quaternion, np.array([0.0, 1.0, 0.0])),
                    _quat_rotate(quaternion, np.array([0.0, 0.0, 1.0])),
                ]
            )
            closed_position = position - _quat_rotate(
                quaternion,
                axis_local * opening,
            )
            mesh = collision_mesh_for_body(body_name)
            shape_source = "asset_mesh"
            shape_position = closed_position
            if mesh is None:
                profile_name = "finger_link1" if index == 1 else "finger_link2"
                profile = R1PRO_GRIPPER_FINGER_COLLISION_BOXES_LOCAL[profile_name]
                half_extents = np.asarray(profile["half_extents"], dtype=float)
                center_local = np.asarray(profile["center"], dtype=float)
                shape = hppfcl.Box(*(2.0 * half_extents))
                shape_position = closed_position + rotation @ center_local
                shape_source = "profiled_box_fallback"
            else:
                shape = mesh
            result = hppfcl.CollisionResult()
            hit = bool(
                hppfcl.collide(
                    obstacle.shape,
                    obstacle_transform,
                    shape,
                    hppfcl.Transform3f(rotation, shape_position),
                    hppfcl.CollisionRequest(),
                    result,
                )
            )
            contacts.append(hit)
            records.append(
                {
                    "body_name": body_name,
                    "opening_m": opening,
                    "closed_position_world": closed_position.round(6).tolist(),
                    "contact": hit,
                    "shape_source": shape_source,
                }
            )
        return {
            "checked": True,
            "all_fingers_contact": bool(contacts) and all(contacts),
            "finger_contacts": contacts,
            "contact_offset_m": contact_offset,
            "records": records,
        }
    except (
        AttributeError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        return {
            "checked": False,
            "reason": "closed finger contact prediction unavailable",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }


def _gf_quat(quat: np.ndarray, gf_module):
    """Convert a wxyz numpy quaternion to a USD ``Gf.Quatf``."""
    unit = _quat_normalize(np.asarray(quat, dtype=float))
    return gf_module.Quatf(
        float(unit[0]),
        gf_module.Vec3f(float(unit[1]), float(unit[2]), float(unit[3])),
    )


def _build_scene_cfg(scene: SceneModel, cfg: AdapterCfg):
    """InteractiveSceneCfg built data-driven from ``scene`` (or bare robot)."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors.camera import CameraCfg
    from isaaclab.sensors import ContactSensorCfg, ImuCfg
    from isaaclab.utils import configclass

    robot_cfg = build_robot_articulation_cfg(_robot_usd_path(cfg), cfg.wheel_control)
    # SceneModel owns the robot's world start pose. Previously this field was
    # silently ignored, so every scene spawned at the origin; a navigation
    # obstacle placed on the direct route could instead spawn around the robot
    # itself and make the start cell occupied. Isaac Lab quaternions are wxyz.
    import math

    rx, ry, ryaw = (float(v) for v in scene.robot.init_pose)
    robot_cfg.init_state.pos = (rx, ry, 0.0)
    robot_cfg.init_state.rot = (
        math.cos(ryaw / 2.0),
        0.0,
        0.0,
        math.sin(ryaw / 2.0),
    )

    attrs: dict[str, object] = {
        "__doc__": "Data-driven scene cfg",
        "robot": robot_cfg,
        "dome_light": AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.85, 0.88, 0.95)),
        ),
    }

    # Ground: from the scene model; top surface at z=0 (CollisionAPI only).
    ground_cfg = sim_utils.CuboidCfg(
        size=(20.0, 20.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=tuple(scene.world.ground_color),
            # Roughness 1.0: the default 0.5 makes the ground glossy and it
            # reflects the lights in RTX rendering.
            roughness=1.0,
            metallic=0.0,
        ),
    )
    if scene.world.ground:
        attrs["ground"] = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=ground_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        )

    # Static kinematic geometry is an AssetBase.  Objects that need a runtime
    # pose/velocity handle opt into ``rigid_object``; this keeps ordinary walls
    # and tables on the stable static path while allowing reusable grasp skills
    # to manipulate a kinematic validation object.
    for obj in scene.objects:
        spawn = _object_spawn_cfg(obj)
        pos = tuple(obj.pos)
        rot = tuple(obj.quat)
        if obj.physics.kinematic and not obj.physics.rigid_object:
            attrs[obj.name] = AssetBaseCfg(
                prim_path=_object_prim_path(obj.name),
                spawn=spawn,
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            )
        else:
            attrs[obj.name] = RigidObjectCfg(
                prim_path=_object_prim_path(obj.name),
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            )

    # Camera (always present; the recorder / set_camera_view depend on it).
    cam = scene.cameras[0] if scene.cameras else None
    eye = tuple(cam.eye) if cam is not None else tuple(cfg.camera_eye)
    target = tuple(cam.target) if cam is not None else tuple(cfg.camera_target)
    attrs["camera"] = CameraCfg(
        prim_path="/World/Camera",
        height=int(cam.height) if cam is not None else cfg.height,
        width=int(cam.width) if cam is not None else cfg.width,
        data_types=list(cam.data_types) if cam is not None else ["rgb"],
        update_period=0.0,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(cam.focal_length) if cam is not None else 24.0,
            focus_distance=10.0,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=eye,
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="world",
        ),
    )

    # The base IMU is a real Isaac Lab sensor attached to the articulated
    # chassis.  It is always present, including scenes that do not declare a
    # task-specific sensor, because dynamic stability is a simulator-wide
    # acceptance invariant rather than a property of one task family.
    attrs["base_imu"] = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r1_pro_with_gripper/base_link",
        update_period=0.0,
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # Support contacts are kept separate from task interaction contacts.  The
    # latter are authored by the scene and feed task predicates; these three
    # sensors provide the runtime three-wheel support invariant used to abort
    # an incipient tip/ground-loss event.
    for wheel_index in (1, 2, 3):
        attrs[f"support_contact_wheel{wheel_index}"] = ContactSensorCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/r1_pro_with_gripper/"
                f"wheel_motor_link{wheel_index}"
            ),
            update_period=0.0,
            history_length=0,
            track_air_time=True,
            # GPU PhysX does not support a contact filter whose collider is
            # the ground body (it reports "GPU contact filter ... not
            # supported" and returns an empty/zero matrix).  Keep the sensor
            # unfiltered and let ``support_contact_forces`` extract the real
            # upward contact component.  This remains physical contact
            # telemetry; it does not synthesize support from pose or a ray.
            filter_prim_paths_expr=[],
        )

    # Contact sensors on robot bodies, filtered to absolute object prim paths.
    # The objects above are spawned at /World/<Name>; PhysX requires absolute
    # filter paths and rejects suffix-only expressions such as ``*Cylinder$``.
    for sensor in scene.contact_sensors:
        filters = _contact_filter_prim_paths(sensor.filter)
        attrs[sensor.name] = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/r1_pro_with_gripper/" + sensor.body,
            update_period=0.0,
            history_length=0,
            filter_prim_paths_expr=filters or None,
        )

    # Collision sensors have a separate semantic contract from allowed
    # interaction contacts.  Only these sensors feed collision_free evidence.
    for sensor in scene.collision_sensors:
        filters = _contact_filter_prim_paths(sensor.filter)
        attrs[sensor.name] = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/r1_pro_with_gripper/" + sensor.body,
            update_period=0.0,
            history_length=0,
            filter_prim_paths_expr=filters or None,
        )

    SceneCfg = configclass(type("SceneCfg", (InteractiveSceneCfg,), attrs))
    return SceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.env_spacing)


def build_scene_cfg(cfg: AdapterCfg):
    """InteractiveSceneCfg for ``cfg`` (kept for callers that pass no scene)."""
    return _build_scene_cfg(cfg.scene or _empty_scene(cfg), cfg)


def effort_within_runtime_limit(
    *,
    hard_exceeded: bool,
    reserve_active_s: float,
    persistence_s: float = R1PRO_RUNTIME_EFFORT_ABORT_PERSISTENCE_S,
) -> bool:
    """Return whether the live effort gate still allows the episode to continue.

    Evidence may keep the episode-max utilization and reserve duration. The
    abort uses the current reserve crossing so a recovered spike does not
    fail a later skill.
    """
    return (not bool(hard_exceeded)) and float(reserve_active_s) < float(persistence_s)


def _empty_scene(cfg: AdapterCfg) -> SceneModel:
    """A bare-robot SceneModel with no objects or contact sensors."""
    return SceneModel.from_dict(
        {
            "name": "bare",
            "robot": {
                "asset": str(cfg.usd_path.resolve()) if cfg.usd_path else "",
                "home_joint_pos": {},
            },
            "objects": [],
            "cameras": [],
            "contact_sensors": [],
        }
    )


class R1ProSimAdapter:
    """Loads the R1Pro scene and exposes name-based control/observation."""

    def __init__(self, cfg: AdapterCfg) -> None:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from isaaclab.sensors.camera import Camera

        self.cfg = cfg
        self.scene_model = cfg.scene or _empty_scene(cfg)
        # ``sim.step(render=True)`` is called by every skill physics step so
        # that the normal adapter contract stays unchanged.  Render only at
        # the video sampling cadence, however: rendering more frequently than
        # the fixed output cadence wastes work without adding video frames.
        # This changes no physics or evidence timestamps and keeps the required
        # video artifact intact.
        render_interval = max(1, round(60.0 / max(1, int(cfg.fps))))
        self._render_interval = render_interval
        self._step_counter = 0
        # LLM / Python planning must not advance physics.  Video frames come
        # from ``step_hook`` on ``step()``, so a frozen clock keeps the
        # recorded trajectory continuous across planner waits.
        self._clock_frozen = False
        self.sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 60.0,
            render_interval=render_interval,
            device=cfg.device,
        )
        self.sim = SimulationContext(self.sim_cfg)
        self.scene: InteractiveScene = InteractiveScene(build_scene_cfg(cfg))
        self.robot: Articulation = self.scene["robot"]
        self.camera: Camera = self.scene["camera"]
        self.dt = self.sim_cfg.dt
        # Name mapping is built on first reset(): articulation data is not
        # initialized before the simulation context is reset.
        self.mapping: JointMapping | None = None
        # Position-difference velocity estimate (PhysX velocity readback is a
        # pseudo-reading on several joints).
        self._prev_joint_pos: np.ndarray | None = None
        self._last_joint_velocity: np.ndarray | None = None
        # ``read_observation`` is called by both a skill and the evidence
        # recorder in the same physics interval.  Only the first read after a
        # step should advance the finite-difference sample; repeated reads
        # must see the same velocity instead of manufacturing a zero sample.
        self._velocity_sample_pending = False
        self._wheels_locked = cfg.wheel_control == "position"
        # A joint-mask lock stores immutable, measured position targets for a
        # manipulation phase.  Sparse skill commands are allowed to move only
        # joints outside this mask; locked targets are re-applied every physics
        # step so load-induced drift cannot silently become the new setpoint.
        self._joint_lock_targets: dict[str, float] = {}
        self._joint_lock_groups: tuple[str, ...] = ()
        self._joint_lock_original_stiffness = None
        self._joint_lock_original_damping = None
        self._joint_lock_max_error = 0.0
        self._joint_lock_max_error_by_group: dict[str, float] = {}
        self._joint_lock_max_joint_by_group: dict[str, str] = {}
        self._joint_lock_max_root_tilt = 0.0
        self._root_pose_at_reset = None
        self._max_root_tilt_rad = 0.0
        self._max_root_height_rise_m = 0.0
        self._max_effort_utilization = 0.0
        self._max_effort_joint_name: str | None = None
        self._max_effort_value_nm = 0.0
        # Evidence metrics must remain JSON-finite even before the first
        # finite-effort sample arrives.  Zero is the explicit "unobserved"
        # sentinel; the utilization and source fields carry the validity
        # context once a real joint sample is recorded.
        self._max_effort_limit_nm = 0.0
        self._max_effort_source: str | None = None
        self._current_effort_utilization = 0.0
        self._current_effort_joint_name: str | None = None
        self._effort_reserve_active_s = 0.0
        self._max_effort_reserve_duration_s = 0.0
        self._hard_effort_limit_exceeded = False
        self._root_wrench_calls = 0
        self._pose_write_counts = {"root": 0, "object": 0}
        self._support_contact_loss_s = 0.0
        self._support_contact_seen = False
        self._imu_seen = False
        # Last whole-body planner selection, exposed as scalar evidence so a
        # physical abort can be attributed to candidate reachability rather
        # than inferred from the final joint sample alone.  These diagnostics
        # are controller metadata; they never alter the simulated state.
        self._whole_body_pregrasp_phase = ""
        self._whole_body_pregrasp_target_distance_m = 0.0
        self._whole_body_pregrasp_target_torso_q = ""
        self._whole_body_pregrasp_target_arm_q = ""
        self._whole_body_pregrasp_candidate_count = 0.0
        self._whole_body_pregrasp_execution_order = ""
        self._whole_body_pregrasp_effort_utilization = 0.0
        self._whole_body_pregrasp_tracking_error_rad = 0.0
        self._whole_body_pregrasp_runtime_phase = ""
        self._whole_body_pregrasp_certified_arm_q = ""
        self._whole_body_pregrasp_certified_torso_q = ""
        self._whole_body_pregrasp_runtime_index = -1
        self._whole_body_pregrasp_runtime_plan_count = 0
        self._last_commanded_torso_q = ""
        self._last_commanded_torso_velocity = ""
        self._last_commanded_left_arm_q = ""
        self._last_commanded_right_arm_q = ""
        self._last_requested_torso_q = ""
        self._last_requested_left_arm_q = ""
        self._last_requested_right_arm_q = ""
        self._alignment_live_collision_checks = 0
        self._alignment_live_collision_last = ""
        self._alignment_live_collision_object = ""
        self._alignment_finger_collision_checks = 0
        self._alignment_finger_collision_last = ""
        self._alignment_finger_pose_last = ""
        self._alignment_geometry_surface_distance_m = 0.0
        self._alignment_geometry_segment_fraction = 0.0
        self._alignment_geometry_contact_margin_m = 0.0
        self._alignment_geometry_finger_p1 = ""
        self._alignment_geometry_finger_p2 = ""
        self._alignment_geometry_object_position = ""
        self._alignment_planning_phase = ""
        self._alignment_planning_candidate_count = 0

    def add_distant_light(self) -> None:
        from pxr import Gf, UsdGeom, UsdLux

        stage = self.sim.stage
        distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant.CreateIntensityAttr(1000.0)
        distant_xf = distant.GetPrim().GetAttribute("xformOp:rotateXYZ")
        if not distant_xf:
            UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(
                Gf.Vec3f(-45.0, -25.0, 25.0)
            )

    def set_camera_view(self) -> None:
        import torch

        cam = self.scene_model.cameras[0] if self.scene_model.cameras else None
        eye = tuple(cam.eye) if cam is not None else self.cfg.camera_eye
        target = tuple(cam.target) if cam is not None else self.cfg.camera_target
        eye_t = torch.tensor([eye], device=self.cfg.device, dtype=torch.float32)
        target_t = torch.tensor([target], device=self.cfg.device, dtype=torch.float32)
        self.camera.set_world_poses_from_view(eye_t, target_t)

    def reset(self) -> None:
        """Full reset: articulation to home, targets cleared, history reset."""
        self.sim.reset()
        self.scene.update(self.dt)
        if self.mapping is None:
            self.mapping = JointMapping(
                joint_names=tuple(self.robot.data.joint_names),
                group_exprs=dict(R1PRO_JOINT_GROUP_EXPR),
            )
            self.mapping.validate()
        self._prev_joint_pos = None
        self._last_joint_velocity = None
        self._velocity_sample_pending = False
        self._step_counter = 0
        self._clock_frozen = False
        self._wheels_locked = self.cfg.wheel_control == "position"
        self._joint_lock_targets = {}
        self._joint_lock_groups = ()
        self._joint_lock_original_stiffness = None
        self._joint_lock_original_damping = None
        self._joint_lock_max_error = 0.0
        self._joint_lock_max_error_by_group: dict[str, float] = {}
        self._joint_lock_max_joint_by_group: dict[str, str] = {}
        self._joint_lock_max_root_tilt = 0.0
        self._root_pose_at_reset = self.robot.data.root_link_pose_w.detach().clone()
        self._max_root_tilt_rad = 0.0
        self._max_root_height_rise_m = 0.0
        self._max_effort_utilization = 0.0
        self._max_effort_joint_name = None
        self._max_effort_value_nm = 0.0
        self._max_effort_limit_nm = 0.0
        self._max_effort_source = None
        self._current_effort_utilization = 0.0
        self._current_effort_joint_name = None
        self._effort_reserve_active_s = 0.0
        self._max_effort_reserve_duration_s = 0.0
        self._hard_effort_limit_exceeded = False
        self._root_wrench_calls = 0
        self._pose_write_counts = {"root": 0, "object": 0}
        self._support_contact_loss_s = 0.0
        self._support_contact_seen = False
        self._imu_seen = False
        self._whole_body_pregrasp_phase = ""
        self._whole_body_pregrasp_target_distance_m = 0.0
        self._whole_body_pregrasp_target_torso_q = ""
        self._whole_body_pregrasp_target_arm_q = ""
        self._whole_body_pregrasp_candidate_count = 0.0
        self._whole_body_pregrasp_execution_order = ""
        self._whole_body_pregrasp_effort_utilization = 0.0
        self._whole_body_pregrasp_tracking_error_rad = 0.0
        self._whole_body_pregrasp_runtime_phase = ""
        self._whole_body_pregrasp_certified_arm_q = ""
        self._whole_body_pregrasp_certified_torso_q = ""
        self._whole_body_pregrasp_runtime_index = -1
        self._whole_body_pregrasp_runtime_plan_count = 0
        self._last_commanded_torso_q = ""
        self._last_commanded_torso_velocity = ""
        self._last_commanded_left_arm_q = ""
        self._last_commanded_right_arm_q = ""
        self._last_requested_torso_q = ""
        self._last_requested_left_arm_q = ""
        self._last_requested_right_arm_q = ""
        self._alignment_live_collision_checks = 0
        self._alignment_live_collision_last = ""
        self._alignment_live_collision_object = ""
        self._alignment_finger_collision_checks = 0
        self._alignment_finger_collision_last = ""
        self._alignment_finger_pose_last = ""
        self._alignment_geometry_surface_distance_m = 0.0
        self._alignment_geometry_segment_fraction = 0.0
        self._alignment_geometry_contact_margin_m = 0.0
        self._alignment_geometry_finger_p1 = ""
        self._alignment_geometry_finger_p2 = ""
        self._alignment_geometry_object_position = ""
        self._alignment_planning_phase = ""
        self._alignment_planning_candidate_count = 0
        self._grasp_joints: dict[str, object] = {}
        self._grasp_anchors: dict[str, dict[str, object]] = {}
        self._last_grasp_attachment_failure: dict[str, object] | None = None

    @contextmanager
    def freeze_simulation_clock(self) -> Iterator[None]:
        """Hold physics (and therefore video) still during LLM/compute.

        Do not call Isaac ``play()`` on exit: restarting the Kit timeline
        independently of ``step()`` would insert idle physics into the
        recording.  The contract is that no ``step()`` happens in this
        context; the next skill resumes by stepping again.
        """
        previous = bool(self._clock_frozen)
        self._clock_frozen = True
        try:
            yield
        finally:
            self._clock_frozen = previous

    def step(self, render: bool = True) -> None:
        if self._clock_frozen:
            raise RuntimeError(
                "simulation clock is frozen during LLM/planning; "
                "physics must not advance until the planner returns"
            )
        self._apply_joint_lock_targets()
        self._apply_gravity_compensation()
        # ``SimulationContext.step(render=True)`` can force a full renderer
        # update even when ``render_interval`` is larger than one.  Skills
        # intentionally call the adapter at physics cadence, while the video
        # recorder only retains one frame per configured output interval.  Do
        # the same throttling at the adapter boundary so low-FPS diagnostic
        # and benchmark runs do not spend most of their wall time rendering
        # discarded frames.  Physics, observations and evidence timestamps
        # still advance at the same 60 Hz cadence.
        should_render = bool(render) and (
            self._render_interval <= 1
            or self._step_counter % self._render_interval == 0
        )
        self.sim.step(render=should_render)
        self._step_counter += 1
        self._velocity_sample_pending = True
        self.scene.update(self.dt)
        self._update_joint_lock_metrics()
        self._update_physical_metrics()

    @property
    def joint_mask_locked(self) -> bool:
        """Whether a reusable phase-level joint mask is currently active."""
        return bool(self._joint_lock_targets)

    @property
    def joint_lock_groups(self) -> tuple[str, ...]:
        """Groups represented by the active phase-level joint mask."""
        return tuple(self._joint_lock_groups)

    def _apply_joint_lock_targets(self) -> None:
        """Re-assert immutable masked targets before every physics substep."""
        if not self._joint_lock_targets:
            return
        import torch

        names = tuple(self._joint_lock_targets)
        indices = [self._index_of(name) for name in names]
        positions = torch.tensor(
            [[self._joint_lock_targets[name] for name in names]],
            device=self.robot.device,
            dtype=self.robot.data.joint_pos.dtype,
        )
        velocities = torch.zeros_like(positions)
        self.robot.set_joint_position_target(positions, joint_ids=indices)
        self.robot.set_joint_velocity_target(velocities, joint_ids=indices)
        self.scene.write_data_to_sim()

    def _joint_lock_errors(self) -> dict[str, float]:
        """Return absolute measured error for every locked joint."""
        if not self._joint_lock_targets:
            return {}
        current = self.robot.data.joint_pos[0].detach().cpu().numpy()
        return {
            name: abs(float(current[self._index_of(name)]) - target)
            for name, target in self._joint_lock_targets.items()
        }

    def _update_joint_lock_metrics(self) -> None:
        """Accumulate physical lock error and root tilt for validation."""
        errors = self._joint_lock_errors()
        if not errors:
            return
        self._joint_lock_max_error = max(self._joint_lock_max_error, max(errors.values()))
        for name, error in errors.items():
            group = self.mapping.group_of(name)
            if error > self._joint_lock_max_error_by_group.get(group, -1.0):
                self._joint_lock_max_error_by_group[group] = error
                self._joint_lock_max_joint_by_group[group] = name
        quat = self.robot.data.root_quat_w[0].detach().cpu().numpy()
        w, x, y, z = (float(v) for v in quat)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))
        sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        pitch = float(np.arcsin(sinp))
        self._joint_lock_max_root_tilt = max(
            self._joint_lock_max_root_tilt, float(np.hypot(roll, pitch))
        )

    def _apply_gravity_compensation(self) -> None:
        """Feed PhysX's live gravity bias to the joint drives while parked.

        The R1Pro shoulder torque limits are intentionally the real asset
        limits.  A long horizontal reach therefore has a large steady-state
        position error if the implicit PD drive has to create both the motion
        torque and the complete gravity torque from position error alone.
        PhysX exposes the articulated inverse-dynamics gravity term ``G(q)``;
        adding it as feed-forward keeps the configured effort limits intact
        while allowing the position drive to track the smooth reference.

        This is evaluated every physics step from the *live* configuration.
        It is not a cached pose or task-specific torque profile.
        """
        if not self._wheels_locked:
            return
        gravity_effort = self.robot.root_physx_view.get_gravity_compensation_forces()
        # For a floating-base articulation PhysX prepends the six root
        # generalized forces (xyz force + xyz torque). Isaac Lab's actuator
        # target only accepts the actual joint DOFs, so retain the tail.
        joint_dofs = self.robot.data.joint_effort_target.shape[-1]
        if gravity_effort.shape[-1] != joint_dofs:
            gravity_effort = gravity_effort[..., -joint_dofs:]
        # Gravity feed-forward is useful for a real position drive, but it is
        # still an actuator command. Reserve part of every finite effort limit
        # for tracking and contact reactions; never let feed-forward turn an
        # impossible posture into an apparently successful one.
        limits = np.asarray(
            [self._effort_limit(name) for name in self.mapping.joint_names],
            dtype=float,
        )
        import torch

        limit_tensor = torch.as_tensor(
            limits * R1PRO_EFFORT_PLANNING_UTILIZATION,
            device=gravity_effort.device,
            dtype=gravity_effort.dtype,
        )
        finite = torch.isfinite(limit_tensor)
        gravity_effort = torch.where(
            finite,
            torch.clamp(gravity_effort, -limit_tensor, limit_tensor),
            gravity_effort,
        )
        self.robot.set_joint_effort_target(gravity_effort)
        # set_joint_effort_target only fills Isaac Lab's command buffer.
        # Flush it after the latest position/velocity targets and before the
        # physics substep so G(q) is refreshed throughout trajectory settling.
        self.scene.write_data_to_sim()

    def _effort_limit(self, joint_name: str) -> float:
        """Return the smallest known physical effort limit for one joint."""
        if joint_name.startswith("torso_joint"):
            return float(R1PRO_TORSO_EFFORT_LIMIT)
        if joint_name.startswith("left_arm_joint"):
            return float(arm_torque_by_joint("left")[joint_name])
        if joint_name.startswith("right_arm_joint"):
            return float(arm_torque_by_joint("right")[joint_name])
        if "gripper_finger_joint" in joint_name:
            return 100.0
        # The wheel and steer drive limits are authored by their USD drive and
        # are not represented by the current robot_config table. Returning
        # infinity here keeps the metric honest instead of inventing a limit.
        return float("inf")

    @staticmethod
    def _root_tilt(quat: np.ndarray) -> float:
        """Return the world tilt magnitude of a wxyz root quaternion."""
        q = _quat_normalize(np.asarray(quat, dtype=float))
        w, x, y, z = (float(value) for value in q)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))
        sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        pitch = float(np.arcsin(sinp))
        return float(np.hypot(roll, pitch))

    def _update_physical_metrics(self) -> None:
        """Accumulate physical integrity metrics from live simulator state."""
        try:
            root_pose = self.robot.data.root_link_pose_w[0].detach().cpu().numpy()
            root_tilt = self._root_tilt(root_pose[3:7])
            self._max_root_tilt_rad = max(self._max_root_tilt_rad, root_tilt)
            if self._root_pose_at_reset is not None:
                initial_z = float(self._root_pose_at_reset[0, 2])
                self._max_root_height_rise_m = max(
                    self._max_root_height_rise_m,
                    float(root_pose[2]) - initial_z,
                )
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
            pass
        for attr in ("joint_effort", "applied_torque", "joint_effort_target"):
            try:
                raw = getattr(self.robot.data, attr)
                values = raw[0].detach().cpu().numpy()
                limits = np.asarray(
                    [self._effort_limit(name) for name in self.mapping.joint_names],
                    dtype=float,
                )
                finite = np.isfinite(limits) & (limits > 0.0)
                if np.any(finite):
                    ratios = np.full(limits.shape, -np.inf, dtype=float)
                    ratios[finite] = np.abs(values[finite]) / limits[finite]
                    index = int(np.argmax(ratios))
                    ratio = float(ratios[index])
                    self._current_effort_utilization = ratio
                    self._current_effort_joint_name = str(self.mapping.joint_names[index])
                    if ratio >= R1PRO_RUNTIME_EFFORT_ABORT_UTILIZATION:
                        self._effort_reserve_active_s += float(self.dt)
                        self._max_effort_reserve_duration_s = max(
                            self._max_effort_reserve_duration_s,
                            self._effort_reserve_active_s,
                        )
                    else:
                        self._effort_reserve_active_s = 0.0
                    if ratio > 1.0 + 1.0e-6:
                        self._hard_effort_limit_exceeded = True
                    if ratio > self._max_effort_utilization:
                        self._max_effort_utilization = ratio
                        self._max_effort_joint_name = str(self.mapping.joint_names[index])
                        self._max_effort_value_nm = float(abs(values[index]))
                        self._max_effort_limit_nm = float(limits[index])
                        self._max_effort_source = attr
                break
            except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
                continue
        support_contacts = self.support_contact_forces()
        if len(support_contacts) == 3:
            self._support_contact_seen = True
            in_contact = sum(1 for force in support_contacts.values() if force > 0.5)
            if in_contact < 3:
                self._support_contact_loss_s += float(self.dt)
            else:
                self._support_contact_loss_s = 0.0
        imu = self._imu_telemetry()
        if imu is not None:
            self._imu_seen = True

    def physical_metrics(self) -> dict[str, Any]:
        """Return fail-closed physical integrity metrics for evidence."""
        lock_metrics = self.joint_lock_metrics()
        return {
            "max_root_tilt_rad": float(self._max_root_tilt_rad),
            "max_root_height_rise_m": float(self._max_root_height_rise_m),
            "max_effort_utilization": float(self._max_effort_utilization),
            "max_effort_joint": self._max_effort_joint_name or "",
            "max_effort_value_nm": float(self._max_effort_value_nm),
            "max_effort_limit_nm": float(self._max_effort_limit_nm),
            "max_effort_source": self._max_effort_source or "",
            "current_effort_utilization": float(self._current_effort_utilization),
            "current_effort_joint": self._current_effort_joint_name or "",
            "effort_reserve_active_s": float(self._effort_reserve_active_s),
            "max_effort_reserve_duration_s": float(self._max_effort_reserve_duration_s),
            "hard_effort_limit_exceeded": bool(self._hard_effort_limit_exceeded),
            "root_wrench_calls": float(self._root_wrench_calls),
            "root_pose_write_count": float(self._pose_write_counts.get("root", 0)),
            "object_pose_write_count": float(self._pose_write_counts.get("object", 0)),
            "support_contact_loss_s": float(self._support_contact_loss_s),
            "support_contact_sensor_count": float(
                len(self.support_contact_forces())
            ),
            "support_contact_observation_complete": bool(self._support_contact_seen),
            "imu_observation_complete": bool(self._imu_seen),
            "within_root_tilt_limit": self._max_root_tilt_rad <= R1PRO_ROOT_TILT_ABORT_RAD,
            "within_root_height_limit": self._max_root_height_rise_m <= R1PRO_ROOT_HEIGHT_RISE_ABORT_M,
            # ``max_effort_utilization`` and ``max_effort_reserve_duration_s``
            # are evidence peaks and may contain a legitimate transient. The
            # runtime gate uses a hard physical over-limit or the *current*
            # reserve crossing. A recovered spike must not fail later skills.
            "within_effort_limit": effort_within_runtime_limit(
                hard_exceeded=self._hard_effort_limit_exceeded,
                reserve_active_s=self._effort_reserve_active_s,
            ),
            "within_support_contact_limit": (
                self._support_contact_loss_s <= R1PRO_WHEEL_CONTACT_LOSS_TIMEOUT_S
            ),
            "no_external_root_wrench": self._root_wrench_calls == 0,
            "no_runtime_pose_writes": all(value == 0 for value in self._pose_write_counts.values()),
            # Keep controller-phase state in the same physical evidence
            # record.  A torso effort spike can only be interpreted correctly
            # when the audit shows whether the immutable wheel/torso mask was
            # active and how far the measured joints had drifted from it.
            "joint_mask_active": bool(self.joint_mask_locked),
            "joint_mask_groups": ",".join(self._joint_lock_groups),
            "joint_mask_wheels_locked": bool(self._wheels_locked),
            "joint_mask_locked_joint_count": float(lock_metrics.get("locked_joint_count", 0.0)),
            "joint_mask_current_error_rad": float(lock_metrics.get("current_locked_joint_error", 0.0)),
            "whole_body_pregrasp_phase": str(self._whole_body_pregrasp_phase),
            "whole_body_pregrasp_target_distance_m": float(self._whole_body_pregrasp_target_distance_m),
            "whole_body_pregrasp_target_torso_q": str(self._whole_body_pregrasp_target_torso_q),
            "whole_body_pregrasp_target_arm_q": str(self._whole_body_pregrasp_target_arm_q),
            "whole_body_pregrasp_candidate_count": float(self._whole_body_pregrasp_candidate_count),
            "whole_body_pregrasp_execution_order": str(
                self._whole_body_pregrasp_execution_order
            ),
            "whole_body_pregrasp_effort_utilization": float(
                self._whole_body_pregrasp_effort_utilization
            ),
            "whole_body_pregrasp_tracking_error_rad": float(
                self._whole_body_pregrasp_tracking_error_rad
            ),
            "whole_body_pregrasp_runtime_phase": str(
                self._whole_body_pregrasp_runtime_phase
            ),
            "whole_body_pregrasp_certified_arm_q": str(
                self._whole_body_pregrasp_certified_arm_q
            ),
            "whole_body_pregrasp_certified_torso_q": str(
                self._whole_body_pregrasp_certified_torso_q
            ),
            "whole_body_pregrasp_runtime_index": float(
                self._whole_body_pregrasp_runtime_index
            ),
            "whole_body_pregrasp_runtime_plan_count": float(
                self._whole_body_pregrasp_runtime_plan_count
            ),
            "last_commanded_torso_q": str(self._last_commanded_torso_q),
            "last_commanded_torso_velocity": str(self._last_commanded_torso_velocity),
            "last_commanded_left_arm_q": str(self._last_commanded_left_arm_q),
            "last_commanded_right_arm_q": str(self._last_commanded_right_arm_q),
            "last_requested_torso_q": str(self._last_requested_torso_q),
            "last_requested_left_arm_q": str(self._last_requested_left_arm_q),
            "last_requested_right_arm_q": str(self._last_requested_right_arm_q),
            "alignment_live_collision_checks": float(
                self._alignment_live_collision_checks
            ),
            "alignment_live_collision_last": str(
                self._alignment_live_collision_last
            ),
            "alignment_live_collision_object": str(
                self._alignment_live_collision_object
            ),
            "alignment_finger_collision_checks": float(
                self._alignment_finger_collision_checks
            ),
            "alignment_finger_collision_last": str(
                self._alignment_finger_collision_last
            ),
            "alignment_finger_pose_last": str(self._alignment_finger_pose_last),
            "alignment_geometry_surface_distance_m": float(
                self._alignment_geometry_surface_distance_m
            ),
            "alignment_geometry_segment_fraction": float(
                self._alignment_geometry_segment_fraction
            ),
            "alignment_geometry_contact_margin_m": float(
                self._alignment_geometry_contact_margin_m
            ),
            "alignment_geometry_finger_p1": str(self._alignment_geometry_finger_p1),
            "alignment_geometry_finger_p2": str(self._alignment_geometry_finger_p2),
            "alignment_geometry_object_position": str(
                self._alignment_geometry_object_position
            ),
            "alignment_planning_phase": str(self._alignment_planning_phase),
            "alignment_planning_candidate_count": float(
                self._alignment_planning_candidate_count
            ),
        }

    def physical_safety_violation(self) -> str | None:
        """Return the first measured physical violation, if any."""
        metrics = self.physical_metrics()
        if not bool(metrics["within_root_tilt_limit"]):
            return "root_tilt_exceeded"
        if not bool(metrics["within_root_height_limit"]):
            return "root_height_rise_exceeded"
        if not bool(metrics["within_effort_limit"]):
            return "joint_effort_limit_exceeded"
        if not bool(metrics["within_support_contact_limit"]):
            return "support_contact_lost"
        if not bool(metrics["imu_observation_complete"]):
            return "imu_observation_unavailable"
        return None

    def rebaseline_physical_metrics(self) -> None:
        """Start task-time physical monitoring from the settled setup state.

        Scene construction and the bounded startup settling window are
        allowed to resolve initial contact penetration.  The setup window is
        checked before this method is called; rebaselining only removes that
        known settling transient from task-time height/tilt deltas and never
        writes a simulator pose.
        """
        self._root_pose_at_reset = self.robot.data.root_link_pose_w.detach().clone()
        self._max_root_tilt_rad = 0.0
        self._max_root_height_rise_m = 0.0
        self._max_effort_utilization = 0.0
        self._max_effort_joint_name = None
        self._max_effort_value_nm = 0.0
        self._max_effort_limit_nm = 0.0
        self._max_effort_source = None
        self._current_effort_utilization = 0.0
        self._current_effort_joint_name = None
        self._effort_reserve_active_s = 0.0
        self._max_effort_reserve_duration_s = 0.0
        self._hard_effort_limit_exceeded = False
        self._support_contact_loss_s = 0.0

    def attach_object(self, object_name: str, body_name: str = "left_gripper_finger_midpoint") -> bool:
        """Create a physical fixed joint at the measured grasp frame.

        The object is connected to the robot through a measured USD/PhysX joint.
        The joint frames are measured at verified two-finger contact, so the
        first solver step preserves the grasp without a task-space snap. If
        USD/PhysX cannot create the joint, the method returns ``False``; there
        is no pose-synchronization fallback. Detach removes the joint and
        leaves the body under normal dynamics.
        """
        self._last_grasp_attachment_failure = None
        if object_name in self._grasp_joints:
            return True
        if object_name not in self.scene.rigid_objects:
            self._last_grasp_attachment_failure = {
                "reason": "object is not a live rigid body",
                "object_name": object_name,
            }
            return False
        obj_model = next(
            (item for item in self.scene_model.objects if item.name == object_name),
            None,
        )
        if obj_model is None:
            self._last_grasp_attachment_failure = {
                "reason": "object has no declarative scene model",
                "object_name": object_name,
            }
            return False
        if bool(obj_model.physics.kinematic):
            self._last_grasp_attachment_failure = {
                "reason": "object is kinematic",
                "object_name": object_name,
            }
            return False
        midpoint_side = next(
            (
                side
                for side in ("left", "right")
                if body_name
                in {
                    f"{side}_gripper_finger_midpoint",
                    f"{side}_finger_midpoint",
                }
            ),
            None,
        )
        # Keep the semantic midpoint name in the public attachment state, but
        # use a real finger rigid body as the PhysX endpoint.  The object
        # origin is the measured joint frame (the same construction used by
        # the previously validated grasp path); using a palm body together
        # with a synthetic midpoint orientation leaves PhysX with disjoint
        # local frames when the object is still supported by the table.
        physics_body_name = (
            f"{midpoint_side}_gripper_finger_link1"
            if midpoint_side is not None
            else str(body_name)
        )
        body_name_for_joint = physics_body_name
        try:
            body_prim = self._robot_body_prim(body_name_for_joint)
            object_prim = self.sim.stage.GetPrimAtPath(_object_prim_path(object_name))
            if not object_prim or not object_prim.IsValid():
                self._last_grasp_attachment_failure = {
                    "reason": "object prim is invalid",
                    "object_name": object_name,
                    "object_prim_path": _object_prim_path(object_name),
                }
                return False
            object_pos = self.scene.rigid_objects[object_name].data.root_pos_w[0].detach().cpu().numpy()
            object_quat = self.scene.rigid_objects[object_name].data.root_quat_w[0].detach().cpu().numpy()
            body_pos, body_quat = self._body_pose(body_name_for_joint)
            body_inv = _quat_inverse(body_quat)
            # Body 0 is the object at its own origin; body 1 is the measured
            # finger frame at the same world object-origin pose.  This makes
            # both joint frames coincident at authoring time and avoids a
            # solver snap while the object is still resting on its support.
            object_local_pos = np.zeros(3, dtype=float)
            object_local_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
            body_local_pos = _quat_rotate(body_inv, object_pos - body_pos)
            body_local_quat = _quat_multiply(body_inv, object_quat)
            joint = self._define_fixed_grasp_joint(
                self._runtime_grasp_joint_path(object_name),
                object_prim,
                body_prim,
                object_local_pos,
                object_local_quat,
                body_local_pos,
                body_local_quat,
            )
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            self._last_grasp_attachment_failure = {
                "reason": "exception while defining fixed grasp joint",
                "object_name": object_name,
                "body_name": body_name,
                "physics_body_name": body_name_for_joint,
                "object_prim_path": _object_prim_path(object_name),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
            return False
        if joint is None or not joint.GetPrim().IsValid():
            self._last_grasp_attachment_failure = {
                "reason": "fixed grasp joint prim is invalid after definition",
                "object_name": object_name,
                "body_name": body_name,
                "physics_body_name": body_name_for_joint,
                "object_prim_path": _object_prim_path(object_name),
            }
            return False
        self._grasp_joints[object_name] = {
            "body_name": body_name,
            "physics_body_name": body_name_for_joint,
            "joint_path": str(joint.GetPath()),
            "constraint_type": "usd_physics_fixed_joint",
            "relative_position_body": tuple(
                float(value) for value in _quat_rotate(body_inv, object_pos - body_pos)
            ),
            "relative_quaternion_body": tuple(
                float(value) for value in _quat_multiply(body_inv, object_quat)
            ),
            "anchor_position_world": tuple(float(value) for value in object_pos),
        }
        self._grasp_anchors[object_name] = {
            "body_name": body_name_for_joint,
            "relative_position_body": tuple(float(value) for value in body_local_pos),
        }
        return True

    @property
    def last_grasp_attachment_failure(self) -> dict[str, object] | None:
        """Return diagnostics from the most recent failed attachment attempt."""
        if self._last_grasp_attachment_failure is None:
            return None
        return dict(self._last_grasp_attachment_failure)

    def _robot_body_prim(self, body_name: str):
        """Resolve one robot rigid-body prim by its authored link name."""
        from pxr import UsdPhysics

        candidates = [
            prim
            for prim in self.sim.stage.Traverse()
            if prim.IsValid()
            and prim.GetName() == body_name
            and "/Robot/" in str(prim.GetPath())
        ]
        # A robot link commonly contains ``visuals/<link_name>`` and
        # ``collisions/<link_name>`` helper Xforms.  They intentionally share
        # the authored name but are not PhysX bodies and cannot be used as a
        # FixedJoint endpoint.  Select the link carrying the rigid-body schema
        # rather than relying on traversal order or a path-depth coincidence.
        rigid_candidates = [
            prim for prim in candidates if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(rigid_candidates) == 1:
            return rigid_candidates[0]
        if len(rigid_candidates) != 1:
            raise RuntimeError(
                f"expected one rigid robot body prim named {body_name!r}, "
                f"found {len(rigid_candidates)} among {len(candidates)} same-name prims"
            )

    def _body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Read one articulated link's current world pose."""
        index = self.robot.data.body_names.index(body_name)
        return (
            self.robot.data.body_link_pos_w[0, index].detach().cpu().numpy(),
            self.robot.data.body_link_quat_w[0, index].detach().cpu().numpy(),
        )

    @staticmethod
    def _runtime_grasp_joint_path(object_name: str) -> str:
        safe_name = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in str(object_name)
        )
        return f"/World/RuntimeGraspConstraints/{safe_name}"

    def _define_fixed_grasp_joint(
        self,
        joint_path: str,
        body0_prim,
        body1_prim,
        body0_local_pos: np.ndarray,
        body0_local_quat: np.ndarray,
        body1_local_pos: np.ndarray,
        body1_local_quat: np.ndarray,
    ):
        """Author one measured-frame PhysX fixed joint on the live stage.

        The explicit ``body0``/``body1`` names are intentional: a grasp uses
        the dynamic object as body 0 and the measured finger rigid body as
        body 1, matching the validated USD/PhysX construction.
        """
        from pxr import Gf, Sdf, UsdPhysics

        stage = self.sim.stage
        path = Sdf.Path(joint_path)
        existing = stage.GetPrimAtPath(path)
        if existing and existing.IsValid():
            stage.RemovePrim(path)
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([body0_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([body1_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*(float(value) for value in body0_local_pos))
        )
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(*(float(value) for value in body1_local_pos))
        )
        joint.CreateLocalRot0Attr().Set(_gf_quat(body0_local_quat, Gf))
        joint.CreateLocalRot1Attr().Set(_gf_quat(body1_local_quat, Gf))
        return joint

    def _grasp_anchor_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return a world pose for a reusable grasp anchor.

        The physical grasp point is the midpoint of the two live finger-link
        origins, not necessarily the origin of the palm/link prim exposed by
        the articulation.  Keeping the anchor name configurable preserves the
        adapter contract for other tools while making the default parallel
        gripper attachment follow the actual pinch location.
        """
        midpoint_side = next(
            (side for side in ("left", "right") if body_name in {f"{side}_gripper_finger_midpoint", f"{side}_finger_midpoint"}),
            None,
        )
        if midpoint_side is not None:
            names = (f"{midpoint_side}_gripper_finger_link1", f"{midpoint_side}_gripper_finger_link2")
            indices = [self.robot.data.body_names.index(name) for name in names]
            positions = self.robot.data.body_link_pos_w[0, indices]
            midpoint = positions.mean(dim=0).detach().cpu().numpy()
            # The midpoint has no independent orientation.  Use the palm
            # link orientation only to define a stable local frame for the
            # object's measured relative quaternion.
            palm_index = self.robot.data.body_names.index(f"{midpoint_side}_gripper_link")
            quat = self.robot.data.body_link_quat_w[0, palm_index].detach().cpu().numpy()
            return midpoint, quat
        try:
            index = self.robot.data.body_names.index(body_name)
        except ValueError:
            raise RuntimeError(f"grasp anchor body {body_name!r} is not present") from None
        return (
            self.robot.data.body_link_pos_w[0, index].detach().cpu().numpy(),
            self.robot.data.body_link_quat_w[0, index].detach().cpu().numpy(),
        )

    def detach_object(self, object_name: str) -> bool:
        """Release a measured grasp and return the object to normal dynamics."""
        constraint = self._grasp_joints.pop(object_name, None)
        self._grasp_anchors.pop(object_name, None)
        if constraint is not None and constraint.get("joint_path"):
            try:
                self.sim.stage.RemovePrim(str(constraint["joint_path"]))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False
        if object_name not in self.scene.rigid_objects:
            return True
        # Do not rewrite pose or zero velocity at release.  The fixed joint
        # removal hands the object back to PhysX with its measured linear and
        # angular velocity, so placement success is determined by dynamics.
        return True

    def grasp_attachment_error(self, object_name: str) -> float:
        """Return the measured object-to-gripper anchor error in metres."""
        constraint = self._grasp_joints.get(object_name)
        if constraint is None:
            return float("inf")
        physics_body_name = str(
            constraint.get("physics_body_name", constraint["body_name"])
        )
        body_pos, body_quat = self._grasp_anchor_pose(physics_body_name)
        expected = body_pos + _quat_rotate(body_quat, np.asarray(constraint["relative_position_body"], dtype=float))
        actual = np.asarray(self.object_position(object_name), dtype=float)
        return float(np.linalg.norm(actual - expected))

    def is_object_attached(self, object_name: str) -> bool:
        """Return whether the adapter currently maintains a measured grasp."""
        return object_name in self._grasp_joints

    def get_grasp_context(self, object_name: str, side: str = "left") -> GraspContext:
        """Read the live object-to-finger relationship for generic carry skills."""
        attached = self.is_object_attached(object_name)
        alignment = self.gripper_object_alignment(object_name, side=side)
        object_position = tuple(float(v) for v in alignment["object_position"])
        grasp_center = tuple(float(v) for v in alignment["finger_midpoint"])
        attachment_error = (
            float(self.grasp_attachment_error(object_name)) if attached else None
        )
        return GraspContext(
            object_name=object_name,
            side=side,
            attached=attached,
            object_position_world=object_position,
            grasp_center_world=grasp_center,
            object_to_grasp_center_world=tuple(
                float(grasp_center[index] - object_position[index]) for index in range(3)
            ),
            attachment_error_m=attachment_error,
        )

    def _sync_grasped_objects(self) -> None:
        """Compatibility hook; live PhysX joints own carried-object motion.

        A per-frame object pose or velocity write would hide bad arm dynamics
        and violate the physical grasp contract. Older internal callers may
        still invoke this symbol, so it remains as an explicit no-op.
        """
        return

    def set_targets(self, position: dict[str, float], velocity: dict[str, float] | None = None) -> None:
        """Write partial named targets with phase-appropriate hold semantics.

        A skill command is intentionally sparse (arm-only, gripper-only, or
        wheel-only).  Filling the other position targets with zero made a
        gripper command pull the arm home and made locked wheels roll back to
        angle zero.  Navigation retains the validated zero/home targets;
        manipulation starts from live positions so unrelated groups remain
        where they are after :meth:`lock_wheels` switches the phase.
        """
        import torch

        def _requested_vector(names: tuple[str, ...]) -> str:
            if not all(name in position for name in names):
                return ""
            return ",".join(f"{float(position[name]):.5f}" for name in names)

        self._last_requested_torso_q = _requested_vector(
            tuple(f"torso_joint{index}" for index in range(1, 5))
        )
        self._last_requested_left_arm_q = _requested_vector(
            tuple(f"left_arm_joint{index}" for index in range(1, 8))
        )
        self._last_requested_right_arm_q = _requested_vector(
            tuple(f"right_arm_joint{index}" for index in range(1, 8))
        )
        device = self.robot.device
        if self._wheels_locked:
            # Manipulation mode: sparse arm/gripper commands must preserve
            # every unrelated joint and the locked wheel angles.
            pos = self.robot.data.joint_pos.detach().clone().to(device)
            # The R1Pro finger prismatic drives are force/velocity driven
            # (their authored position stiffness is zero).  Copying their
            # *measured* value into every sparse arm command therefore turns
            # an explicit ``gripper_set(open)`` into a moving target: while
            # the arm is approaching, a finger can drift toward its limit and
            # the jaw closes asymmetrically.  Preserve the last commanded
            # finger position until a gripper skill explicitly replaces it.
            # This is still generic sparse-command semantics: the live state
            # remains the baseline for unrelated groups, while a previously
            # issued actuator target remains authoritative for a tool group
            # whose drive cannot hold position from measurement alone.
            previous_targets = getattr(self.robot.data, "joint_pos_target", None)
            if previous_targets is not None:
                for joint_name in self.mapping.joint_names:
                    if "gripper_finger_joint" not in joint_name:
                        continue
                    index = self._index_of(joint_name)
                    pos[0, index] = previous_targets[0, index].to(device)
        else:
            # Navigation mode: retain the validated all-zero home targets for
            # unspecified joints.  Letting those targets follow small live
            # articulation drift changed the chassis load and caused the
            # forward tracker to miss late waypoints.
            pos = torch.zeros(1, self.mapping.num_joints, device=device)
        vel = torch.zeros_like(pos)
        for name, value in position.items():
            pos[0, self._index_of(name)] = value
        if velocity:
            for name, value in velocity.items():
                vel[0, self._index_of(name)] = value
        # The phase mask has final authority.  In particular, an arm command
        # must not replace a locked torso target with the torso's slightly
        # deflected live position on every frame.
        for name, value in self._joint_lock_targets.items():
            index = self._index_of(name)
            pos[0, index] = value
            vel[0, index] = 0.0
        self.robot.set_joint_position_target(pos)
        self.robot.set_joint_velocity_target(vel)
        torso_indices = [self._index_of(f"torso_joint{index}") for index in range(1, 5)]
        commanded_torso = pos[0, torso_indices].detach().cpu().numpy()
        commanded_velocity = vel[0, torso_indices].detach().cpu().numpy()
        self._last_commanded_torso_q = ",".join(
            f"{float(value):.5f}" for value in commanded_torso
        )
        self._last_commanded_torso_velocity = ",".join(
            f"{float(value):.5f}" for value in commanded_velocity
        )
        for side in ("left", "right"):
            arm_names = tuple(f"{side}_arm_joint{index}" for index in range(1, 8))
            commanded_arm = [
                pos[0, self._index_of(name)].detach().cpu().item()
                for name in arm_names
            ]
            setattr(
                self,
                f"_last_commanded_{side}_arm_q",
                ",".join(f"{float(value):.5f}" for value in commanded_arm),
            )
        self.scene.write_data_to_sim()

    def _index_of(self, joint_name: str) -> int:
        names = self.mapping.joint_names
        try:
            return names.index(joint_name)
        except ValueError:
            raise ValueError(f"unknown joint: {joint_name}") from None

    def read_observation(self, timestamp: float) -> Observation:
        """Read actual state into a domain Observation (name-keyed)."""
        pos = self.robot.data.joint_pos[0].detach().cpu()
        vel_est = self.joint_vel_estimate()
        root_pos = self.robot.data.root_pos_w[0].detach().cpu().numpy()
        root_velocity = self.robot.data.root_vel_w[0].detach().cpu().numpy()
        root_quat = self.robot.data.root_quat_w[0].detach().cpu().numpy()
        w, x, y, z = (float(v) for v in root_quat)
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        imu = self._imu_telemetry()
        support_contacts = self.support_contact_forces()
        effort = self._joint_effort_observation()
        return Observation(
            timestamp=timestamp,
            joint_positions=dict(zip(self.mapping.joint_names, pos.tolist())),
            joint_velocities=dict(
                zip(self.mapping.joint_names, vel_est.tolist() if vel_est is not None else [0.0] * len(self.mapping.joint_names))
            ),
            # Domain contract is (x, y, yaw), not the root XYZ position.
            base_pose=(float(root_pos[0]), float(root_pos[1]), yaw),
            base_velocity=(
                float(root_velocity[0]),
                float(root_velocity[1]),
                float(root_velocity[5]),
            ),
            end_effector_pose=self.end_effector_poses().get("left_ee"),
            contacts=self.contact_events(),
            object_states=self.all_object_states(),
            base_orientation=tuple(float(value) for value in root_quat),
            base_height_m=float(root_pos[2]),
            imu_linear_acceleration=(
                None if imu is None else tuple(float(value) for value in imu["linear_acceleration"])
            ),
            imu_angular_velocity=(
                None if imu is None else tuple(float(value) for value in imu["angular_velocity"])
            ),
            support_contacts=support_contacts,
            joint_efforts=effort,
            physical_metrics=self.physical_metrics(),
        )

    def _joint_effort_observation(self) -> dict[str, float]:
        """Read actual joint effort, never the requested target effort."""
        for attribute in ("joint_effort", "applied_torque"):
            try:
                values = getattr(self.robot.data, attribute)[0].detach().cpu().numpy()
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                continue
            return {
                name: float(value)
                for name, value in zip(self.mapping.joint_names, values)
            }
        return {}

    def support_contact_forces(self) -> dict[str, float]:
        """Return measured normal force for each configured support wheel.

        GPU PhysX cannot filter a contact sensor to the ground collider.  The
        support sensors therefore expose all contacts on each wheel and this
        method retains only the upward world-force component, which is the
        normal reaction for the horizontal ground plane.  Missing sensors are
        omitted rather than represented as synthetic zeroes; this lets the
        runtime gate distinguish a real three-wheel loss from an unavailable
        contact observation.
        """
        forces: dict[str, float] = {}
        for wheel_index in (1, 2, 3):
            sensor = self.scene.sensors.get(f"support_contact_wheel{wheel_index}")
            if sensor is None:
                continue
            try:
                data = sensor.data
                matrix = getattr(data, "force_matrix_w", None)
                if matrix is not None:
                    values = matrix[0].detach().cpu().numpy()
                    force = float(np.maximum(values[..., 2], 0.0).max()) if values.size else 0.0
                else:
                    net = getattr(data, "net_forces_w", None)
                    values = net[0].detach().cpu().numpy() if net is not None else np.empty((0, 3))
                    force = float(np.maximum(values[..., 2], 0.0).max()) if values.size else 0.0
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                continue
            forces[f"wheel{wheel_index}"] = force
        return forces

    def _imu_telemetry(self) -> dict[str, np.ndarray] | None:
        """Read the real base IMU sensor, if its buffers are available."""
        sensor = self.scene.sensors.get("base_imu")
        if sensor is None:
            return None
        try:
            data = sensor.data
            acceleration = data.lin_acc_b[0].detach().cpu().numpy()
            angular_velocity = data.ang_vel_b[0].detach().cpu().numpy()
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            return None
        if acceleration.shape != (3,) or angular_velocity.shape != (3,):
            return None
        return {
            "linear_acceleration": np.asarray(acceleration, dtype=float),
            "angular_velocity": np.asarray(angular_velocity, dtype=float),
        }

    def joint_vel_estimate(self) -> np.ndarray | None:
        """Velocity from (pos - prev_pos) / dt; None before the first step."""
        pos = self.robot.data.joint_pos[0].detach().cpu().numpy()
        if self._prev_joint_pos is None:
            self._prev_joint_pos = pos.copy()
            self._last_joint_velocity = np.zeros_like(pos)
            self._velocity_sample_pending = False
            return None
        if not self._velocity_sample_pending:
            return (
                None
                if self._last_joint_velocity is None
                else self._last_joint_velocity.copy()
            )
        vel = (pos - self._prev_joint_pos) / self.dt
        self._prev_joint_pos = pos.copy()
        self._last_joint_velocity = vel.copy()
        self._velocity_sample_pending = False
        return vel

    def position_error(self, target: dict[str, float]) -> dict[str, float]:
        """Per-joint |actual - target| at the current step."""
        obs = self.read_observation(timestamp=0.0)
        return {
            name: abs(obs.joint_positions[name] - target.get(name, obs.joint_positions[name]))
            for name in self.mapping.joint_names
        }

    def apply_command(self, command: ControlCommand) -> None:
        """Convert a domain ControlCommand into simulation targets."""
        self.set_targets(position=dict(command.position_targets), velocity=dict(command.velocity_targets))

    def lock_wheels(self, stiffness: float = 500.0, damping: float = 100.0) -> None:
        """Switch wheels to stiff position hold at runtime.

        In velocity mode the wheels are free (stiffness=0): arm motion shifts
        the center of mass and the base rolls slowly, which disturbs arm
        execution. Locking the wheel drives (verified: wheel_control=position
        scenes execute arm poses perfectly) stabilizes the base during arm
        motion. Uses the PhysX view directly so no scene rebuild is needed.
        """
        if self.joint_mask_locked:
            # A task-level mask is stronger and must retain its original
            # targets. Arm skills call this legacy helper defensively.
            return
        self.lock_joint_mask(
            mask_mode="lock",
            joint_groups=("wheel",),
            lock_root=False,
            gain_overrides={"wheel": (float(stiffness), float(damping))},
        )

    def unlock_wheels(self, damping: float = 50.0) -> None:
        """Restore free-spin velocity control on the wheels (inverse of
        :meth:`lock_wheels`). Use when the base must move again after an
        arm-manipulation phase."""
        del damping
        self.unlock_joint_mask()

    def lock_joint_mask(
        self,
        mask_mode: str = "lock",
        joint_groups: tuple[str, ...] | list[str] = (),
        joint_names: tuple[str, ...] | list[str] = (),
        lock_root: bool = False,
        stiffness_scale: float = 1.0,
        gain_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, object]:
        """Lock joints selected by a reusable group/name mask.

        ``mask_mode='lock'`` freezes the selected entries. ``'allow'`` keeps
        the selected entries active and freezes every other articulation
        joint. Targets are captured once from measured state and remain fixed
        until :meth:`unlock_joint_mask` is called.
        """
        import torch

        if mask_mode not in {"lock", "allow"}:
            raise ValueError("mask_mode must be 'lock' or 'allow'")
        groups = tuple(dict.fromkeys(str(group) for group in joint_groups))
        explicit = tuple(dict.fromkeys(str(name) for name in joint_names))
        unknown_groups = sorted(set(groups) - set(self.mapping.group_exprs))
        unknown_names = sorted(set(explicit) - set(self.mapping.joint_names))
        if unknown_groups:
            raise ValueError(f"unknown joint groups: {unknown_groups}")
        if unknown_names:
            raise ValueError(f"unknown joint names: {unknown_names}")
        selected = set(explicit)
        for group in groups:
            selected.update(self.mapping.names_of(group))
        if not selected:
            raise ValueError("joint mask must select at least one group or joint")
        locked = (
            selected
            if mask_mode == "lock"
            else set(self.mapping.joint_names) - selected
        )
        if not locked:
            raise ValueError("joint mask does not lock any joints")

        # Replacing a mask is explicit and starts a fresh measured phase.
        if self.joint_mask_locked:
            self.unlock_joint_mask()
        current = self.robot.data.joint_pos[0].detach().cpu().numpy()
        ordered_locked = tuple(name for name in self.mapping.joint_names if name in locked)
        self._joint_lock_targets = {
            name: float(current[self._index_of(name)]) for name in ordered_locked
        }
        self._joint_lock_groups = tuple(
            group for group in self.mapping.group_exprs
            if any(name in locked for name in self.mapping.names_of(group))
        )
        self._joint_lock_original_stiffness = self.robot.data.joint_stiffness.clone().cpu()
        self._joint_lock_original_damping = self.robot.data.joint_damping.clone().cpu()
        self._joint_lock_max_error = 0.0
        self._joint_lock_max_error_by_group: dict[str, float] = {}
        self._joint_lock_max_joint_by_group: dict[str, str] = {}
        self._joint_lock_max_root_tilt = 0.0

        # Minimum group gains preserve the validated actuator tuning while
        # giving free-spin wheels a real position hold. The immutable target,
        # rather than extreme stiffness, is what prevents torso creep.
        minimum_gains = {
            "steer": (500.0, 100.0),
            "wheel": (500.0, 100.0),
            "torso": (2000.0, 500.0),
            "left_arm": (800.0, 80.0),
            "right_arm": (800.0, 80.0),
            "left_gripper": (500.0, 50.0),
            "right_gripper": (500.0, 50.0),
        }
        minimum_gains.update(gain_overrides or {})
        stiffness_full = self.robot.data.joint_stiffness.clone().cpu()
        damping_full = self.robot.data.joint_damping.clone().cpu()
        scale = max(0.1, float(stiffness_scale))
        for name in ordered_locked:
            index = self._index_of(name)
            group = self.mapping.group_of(name)
            min_stiffness, min_damping = minimum_gains[group]
            stiffness_full[0, index] = max(float(stiffness_full[0, index]), min_stiffness * scale)
            damping_full[0, index] = max(float(damping_full[0, index]), min_damping * np.sqrt(scale))
        env_ids = torch.tensor([0], dtype=torch.long)
        self.robot.root_physx_view.set_dof_stiffnesses(stiffness_full, indices=env_ids)
        self.robot.root_physx_view.set_dof_dampings(damping_full, indices=env_ids)
        self._wheels_locked = all(name in locked for name in self.mapping.names_of("wheel"))
        self._apply_joint_lock_targets()
        return {
            "mask_mode": mask_mode,
            "selected_groups": list(groups),
            "selected_joint_names": list(explicit),
            "locked_groups": list(self._joint_lock_groups),
            "locked_joint_names": list(ordered_locked),
            "active_joint_names": [name for name in self.mapping.joint_names if name not in locked],
            # Retain the legacy request field for callers and logs. The
            # floating base is never locked by a wrench or pose write.
            "lock_root": bool(lock_root),
            "root_assist_used": False,
        }

    def extend_joint_mask(
        self,
        *,
        joint_groups: tuple[str, ...] | list[str] = (),
        joint_names: tuple[str, ...] | list[str] = (),
        gain_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, object]:
        """Atomically add joints to the active lock mask.

        A whole-body phase often starts with steer/wheel position holds and
        adds the torso after a coordinated motion.  Calling
        :meth:`unlock_joint_mask` followed by :meth:`lock_joint_mask` clears
        the old command buffers and leaves one physics frame exposed to the
        velocity-mode drive.  That can create a false-looking target jump and
        saturate the wheels and torso together.  Preserve the active mask,
        gains, and targets, then add only the newly requested joints at their
        measured positions.  The floating root remains completely physical.
        """
        import torch

        if not self.joint_mask_locked:
            return self.lock_joint_mask(
                mask_mode="lock",
                joint_groups=joint_groups,
                joint_names=joint_names,
                lock_root=False,
                gain_overrides=gain_overrides,
            )
        groups = tuple(dict.fromkeys(str(group) for group in joint_groups))
        explicit = tuple(dict.fromkeys(str(name) for name in joint_names))
        unknown_groups = sorted(set(groups) - set(self.mapping.group_exprs))
        unknown_names = sorted(set(explicit) - set(self.mapping.joint_names))
        if unknown_groups:
            raise ValueError(f"unknown joint groups: {unknown_groups}")
        if unknown_names:
            raise ValueError(f"unknown joint names: {unknown_names}")

        selected = set(self._joint_lock_targets)
        selected.update(explicit)
        for group in groups:
            selected.update(self.mapping.names_of(group))
        current = self.robot.data.joint_pos[0].detach().cpu().numpy()
        # This method marks a new physical phase after the previously locked
        # joints have been allowed to move under load.  Rebase every selected
        # target to its measured position before the next sparse command;
        # retaining the old wheel target would ask the drive to correct the
        # accumulated encoder drift in one frame and recreate the very torque
        # spike this atomic transition is meant to avoid.
        for name in self.mapping.joint_names:
            if name in selected:
                self._joint_lock_targets[name] = float(current[self._index_of(name)])
        self._joint_lock_groups = tuple(
            group
            for group in self.mapping.group_exprs
            if any(name in selected for name in self.mapping.names_of(group))
        )

        minimum_gains = {
            "steer": (500.0, 100.0),
            "wheel": (500.0, 100.0),
            "torso": (2000.0, 500.0),
            "left_arm": (800.0, 80.0),
            "right_arm": (800.0, 80.0),
            "left_gripper": (500.0, 50.0),
            "right_gripper": (500.0, 50.0),
        }
        minimum_gains.update(gain_overrides or {})
        stiffness_full = self.robot.data.joint_stiffness.clone().cpu()
        damping_full = self.robot.data.joint_damping.clone().cpu()
        for name in self._joint_lock_targets:
            index = self._index_of(name)
            group = self.mapping.group_of(name)
            min_stiffness, min_damping = minimum_gains[group]
            stiffness_full[0, index] = max(float(stiffness_full[0, index]), min_stiffness)
            damping_full[0, index] = max(float(damping_full[0, index]), min_damping)
        env_ids = torch.tensor([0], dtype=torch.long)
        self.robot.root_physx_view.set_dof_stiffnesses(stiffness_full, indices=env_ids)
        self.robot.root_physx_view.set_dof_dampings(damping_full, indices=env_ids)
        self._wheels_locked = all(
            name in selected for name in self.mapping.names_of("wheel")
        )
        self._apply_joint_lock_targets()
        return {
            "mask_mode": "lock",
            "selected_groups": list(groups),
            "selected_joint_names": list(explicit),
            "locked_groups": list(self._joint_lock_groups),
            "locked_joint_names": list(self._joint_lock_targets),
            "active_joint_names": [
                name for name in self.mapping.joint_names if name not in selected
            ],
            "lock_root": False,
            "root_assist_used": False,
            "atomic_extension": True,
            "targets_rebased_to_measured_state": True,
        }

    def rebase_joint_mask_targets(self) -> dict[str, object]:
        """Rebase the active lock targets to the current measured state.

        This is used at a boundary between two controller phases after a
        loaded motion.  It changes only future joint-drive targets; it never
        writes an articulation pose or applies a root wrench.  Rebaselining
        before a final settle prevents an old wheel/steer target from causing
        a one-frame corrective impulse while an arm/torso skill is issuing a
        sparse command.
        """
        if not self.joint_mask_locked:
            return {
                "rebased": False,
                "reason": "no active joint mask",
            }
        current = self.robot.data.joint_pos[0].detach().cpu().numpy()
        for name in self._joint_lock_targets:
            self._joint_lock_targets[name] = float(current[self._index_of(name)])
        self._apply_joint_lock_targets()
        return {
            "rebased": True,
            "joint_count": len(self._joint_lock_targets),
            "root_assist_used": False,
        }

    def joint_lock_metrics(self) -> dict[str, float]:
        """Current and worst physical errors for the active joint mask."""
        errors = self._joint_lock_errors()
        if not errors:
            return {
                "locked_joint_count": 0.0,
                "current_locked_joint_error": 0.0,
                "max_locked_joint_error": 0.0,
                "max_root_tilt_rad": 0.0,
            }
        metrics = {
            "locked_joint_count": float(len(errors)),
            "current_locked_joint_error": float(max(errors.values())),
            "max_locked_joint_error": float(max(max(errors.values()), self._joint_lock_max_error)),
            "max_root_tilt_rad": float(self._joint_lock_max_root_tilt),
        }
        for group in self._joint_lock_groups:
            group_errors = {
                name: error for name, error in errors.items()
                if self.mapping.group_of(name) == group
            }
            if not group_errors:
                continue
            current_error = max(group_errors.values())
            metrics[f"joint_lock_{group}_current_error_rad"] = float(current_error)
            metrics[f"joint_lock_{group}_max_error_rad"] = float(
                max(current_error, self._joint_lock_max_error_by_group.get(group, 0.0))
            )
        return metrics

    def joint_lock_diagnostics(self) -> dict[str, object]:
        """Identify which joint produced each lock-phase peak error."""
        if not self._joint_lock_targets:
            return {"joint_lock_max_error_joints": {}}
        return {
            "joint_lock_max_error_joints": {
                group: self._joint_lock_max_joint_by_group.get(
                    group,
                    max(
                        (name for name in self._joint_lock_targets if self.mapping.group_of(name) == group),
                        key=lambda name: self._joint_lock_errors()[name],
                    ),
                )
                for group in self._joint_lock_groups
            }
        }

    def unlock_joint_mask(self) -> None:
        """Release the current mask and restore the actuator gains exactly."""
        import torch

        if self._joint_lock_original_stiffness is not None:
            env_ids = torch.tensor([0], dtype=torch.long)
            self.robot.root_physx_view.set_dof_stiffnesses(
                self._joint_lock_original_stiffness, indices=env_ids
            )
            self.robot.root_physx_view.set_dof_dampings(
                self._joint_lock_original_damping, indices=env_ids
            )
        self._joint_lock_targets = {}
        self._joint_lock_groups = ()
        self._joint_lock_original_stiffness = None
        self._joint_lock_original_damping = None
        self._joint_lock_max_error = 0.0
        self._joint_lock_max_error_by_group = {}
        self._joint_lock_max_joint_by_group = {}
        self._joint_lock_max_root_tilt = 0.0
        self._wheels_locked = self.cfg.wheel_control == "position"
        self.robot.set_joint_effort_target(torch.zeros_like(self.robot.data.joint_effort_target))
        self.scene.write_data_to_sim()

    def object_state(self, name: str) -> EntityState:
        """Return live state for rigid objects and declared state for static assets."""
        rigid = self.scene.rigid_objects.get(name)
        if rigid is not None:
            pos = rigid.data.root_pos_w[0].detach().cpu().numpy()
            quat = rigid.data.root_quat_w[0].detach().cpu().numpy()
            velocity = rigid.data.root_vel_w[0].detach().cpu().numpy()
            return EntityState(
                position=tuple(float(value) for value in pos),
                quaternion=tuple(float(value) for value in quat),
                linear_velocity=tuple(float(value) for value in velocity[:3]),
                angular_velocity=tuple(float(value) for value in velocity[3:6]),
                source="live",
            )
        try:
            obj = self.scene_model.object(name)
        except KeyError as exc:
            raise RuntimeError(f"object {name!r} is not in the scene") from exc
        return EntityState(
            position=tuple(float(value) for value in obj.pos),
            quaternion=tuple(float(value) for value in obj.quat),
            linear_velocity=(0.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            source="declared",
        )

    def all_object_states(self) -> dict[str, EntityState]:
        """Return state for every object declared in the SceneModel."""
        return {
            obj.name: self.object_state(obj.name)
            for obj in self.scene_model.objects
        }

    def end_effector_poses(self) -> dict[str, tuple[float, ...]]:
        """Return world poses for public EE links and grasp anchors.

        The public ``*_ee`` pose remains the stable observation interface.
        The midpoint pose is additionally recorded because the runtime grasp
        constraint is anchored there; attachment verification must compare the
        object with the same physical point rather than a nearby wrist/link
        origin whose offset changes with posture.
        """
        poses: dict[str, tuple[float, ...]] = {}
        for side in ("left", "right"):
            body_name = f"{side}_gripper_link"
            try:
                index = self.robot.data.body_names.index(body_name)
            except ValueError:
                continue
            position = self.robot.data.body_link_pos_w[0, index].detach().cpu().numpy()
            quaternion = self.robot.data.body_link_quat_w[0, index].detach().cpu().numpy()
            poses[f"{side}_ee"] = tuple(
                float(value) for value in (*position, *quaternion)
            )
            try:
                finger_indices = [
                    self.robot.data.body_names.index(f"{side}_gripper_finger_link1"),
                    self.robot.data.body_names.index(f"{side}_gripper_finger_link2"),
                ]
            except ValueError:
                continue
            midpoint = (
                self.robot.data.body_link_pos_w[0, finger_indices]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )
            poses[f"{side}_gripper_finger_midpoint"] = tuple(
                float(value)
                for value in (*midpoint, *quaternion)
            )
        return poses

    def contact_events(self) -> tuple[ContactEvent, ...]:
        """Return allowed interaction contacts from contact sensors."""
        return self._events_from_sensors(self.scene_model.contact_sensors)

    def collision_events(self) -> tuple[ContactEvent, ...]:
        """Return disallowed collision events from collision sensors only."""
        return self._events_from_sensors(self.scene_model.collision_sensors)

    def _events_from_sensors(
        self,
        sensor_models: tuple[object, ...],
    ) -> tuple[ContactEvent, ...]:
        events: list[ContactEvent] = []
        timestamp = float(self.sim.current_time)
        for sensor_model in sensor_models:
            sensor = self.scene.sensors.get(sensor_model.name)
            if sensor is None:
                continue
            matrix = sensor.data.force_matrix_w
            if not sensor_model.filter or matrix is None:
                continue
            filtered = matrix[0].detach().cpu().numpy()
            if filtered.ndim < 3:
                continue
            for filter_index, object_name in enumerate(sensor_model.filter):
                if filter_index >= filtered.shape[-2]:
                    break
                force = float(np.linalg.norm(filtered[..., filter_index, :], axis=-1).max())
                if force <= 0.0:
                    continue
                events.append(
                    ContactEvent(
                        timestamp=timestamp,
                        body_a=sensor_model.body,
                        body_b=object_name,
                        force_n=force,
                    )
                )
        return tuple(events)

    @property
    def collision_observation_complete(self) -> bool:
        """Return whether every declared collision sensor has live filter data."""
        sensors = self.scene_model.collision_sensors
        if not sensors:
            return False
        for sensor_model in sensors:
            sensor = self.scene.sensors.get(sensor_model.name)
            if sensor is None:
                return False
            matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            if matrix is None:
                return False
            try:
                filtered = matrix[0]
                dimensions = int(filtered.ndim)
                filter_slots = int(filtered.shape[-2]) if dimensions >= 2 else 0
            except (IndexError, AttributeError, TypeError, ValueError):
                return False
            if dimensions < 3 or filter_slots < len(sensor_model.filter):
                return False
        return True

    def attachment_state(self) -> dict[str, str]:
        """Return current entity-to-effector attachment facts."""
        return {
            object_name: str(constraint["body_name"])
            for object_name, constraint in self._grasp_joints.items()
        }

    def object_position(self, name: str) -> tuple[float, float, float]:
        """World position of a scene object by name.

        Dynamic rigid objects are read from physics. Static scene objects fall
        back to the declarative model, which lets the planner combine live
        pushed-object state with authored static collision geometry.
        """
        return self.object_state(name).position

    def robot_mass_kg(self) -> float:
        """Return the articulated robot mass read from the live PhysX view.

        Planning may use the model COM, but the mass used for a payload
        stability certificate must come from the instantiated articulation.
        This keeps USD asset changes and per-scene robot variants from being
        hidden behind a task-specific constant.
        """
        try:
            values = self.robot.data.default_mass[0].detach().cpu().numpy()
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError("live robot mass is unavailable") from exc
        mass = float(np.asarray(values, dtype=float).sum())
        if not np.isfinite(mass) or mass <= 0.0:
            raise RuntimeError("live robot mass is invalid")
        return mass

    def object_mass_kg(self, name: str) -> float:
        """Return a rigid object's mass from its live Isaac Lab asset."""
        try:
            obj = self.scene.rigid_objects[name]
            values = obj.data.default_mass[0].detach().cpu().numpy()
        except (AttributeError, KeyError, IndexError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"live mass for object {name!r} is unavailable") from exc
        mass = float(np.asarray(values, dtype=float).reshape(-1)[0])
        if not np.isfinite(mass) or mass <= 0.0:
            raise RuntimeError(f"live mass for object {name!r} is invalid")
        return mass

    def base_footprint(self) -> dict[str, float]:
        """Conservative chassis footprint derived from authored wheel joints.

        The asymmetric extents are useful for interaction skills.  Navigation
        still consumes the orientation-free circumscribed radius, while a
        forward push can use the actual front support instead of the rear
        wheel's distance from the root.
        """
        from r1pro_data_gen.robot.chassis import STEER_POSITIONS

        xs = [p[0] for p in STEER_POSITIONS.values()]
        ys = [p[1] for p in STEER_POSITIONS.values()]
        front = max(xs) + 0.06
        rear = max(-v for v in xs) + 0.06
        left = max(ys) + 0.06
        right = max(-v for v in ys) + 0.06
        half_x = max(front, rear)
        half_y = max(left, right)
        return {
            "half_length_m": float(half_x),
            "half_width_m": float(half_y),
            "front_extent_m": float(front),
            "rear_extent_m": float(rear),
            "left_extent_m": float(left),
            "right_extent_m": float(right),
            "circumscribed_radius_m": float(np.hypot(half_x, half_y)),
        }

    def cylinder_position(self) -> tuple[float, float, float]:
        """World position of the target cylinder (compat helper)."""
        return self.object_position("cylinder")

    def body_position(self, body_name: str) -> tuple[float, float, float]:
        """World position of a named articulation link origin.

        Use the explicit link-frame property so this measurement has the same
        URDF actor-frame meaning as Pinocchio FK.  In Isaac Lab 2.3 the legacy
        ``body_pos_w`` property aliases this tensor.
        """
        try:
            index = self.robot.data.body_names.index(body_name)
        except ValueError:
            raise RuntimeError(f"robot body {body_name!r} is not present") from None
        pos = self.robot.data.body_link_pos_w[0, index].detach().cpu().numpy()
        return tuple(float(v) for v in pos)

    def body_pose(self, body_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """World position and ``wxyz`` orientation of a link frame.

        Collision-sensitive manipulation needs the orientation of the actual
        finger link mesh, not only its origin.  Keeping this as a small
        name-based adapter method preserves the same backend contract as
        :meth:`body_position` and avoids exposing Isaac tensors to skills.
        """
        try:
            index = self.robot.data.body_names.index(body_name)
        except ValueError:
            raise RuntimeError(f"robot body {body_name!r} is not present") from None
        position = self.robot.data.body_link_pos_w[0, index].detach().cpu().numpy()
        quaternion = self.robot.data.body_link_quat_w[0, index].detach().cpu().numpy()
        return (
            tuple(float(v) for v in position),
            tuple(float(v) for v in quaternion),
        )

    def gripper_object_alignment(self, object_name: str, side: str = "left") -> dict[str, object]:
        """Measure whether an object is in the open gripper window.

        This is a geometry gate used before closing the fingers; contact force
        alone is insufficient because a fingertip can hit and push an object
        from the side.  The test uses the live finger-link positions and the
        shortest distance to their connecting segment.
        """
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        p1 = np.asarray(self.body_position(f"{side}_gripper_finger_link1"), dtype=float)
        p2 = np.asarray(self.body_position(f"{side}_gripper_finger_link2"), dtype=float)
        obj = np.asarray(self.object_position(object_name), dtype=float)
        span = p2 - p1
        denom = float(np.dot(span, span))
        alpha = 0.5 if denom < 1e-10 else float(np.dot(obj - p1, span) / denom)
        closest = p1 + np.clip(alpha, 0.0, 1.0) * span
        distance = float(np.linalg.norm(obj - closest))
        obj_model = next((item for item in self.scene_model.objects if item.name == object_name), None)
        object_radius = object_xy_radius_m(obj_model) if obj_model is not None else 0.02
        # Measure proximity to the primitive surface, not merely its center.
        # Finger-link origins sit on the upper finger mesh, so a valid pinch
        # can be slightly more than half the object height from its center.
        window_radius = object_radius + 0.025
        surface_tolerance = 0.012
        if obj_model is not None:
            surface_distance = object_surface_distance_m(obj, closest, obj_model)
        else:
            surface_distance = distance
        window_geometry: dict[str, object] = {}
        between = 0.08 <= alpha <= 0.92 and surface_distance <= surface_tolerance
        # The link origins are above the actual contact portions of the open
        # finger meshes. For a floor object, use the projected jaw line plus
        # the measured vertical overlap of both collision boxes; the legacy
        # origin-distance result remains in ``surface_distance_m`` for
        # diagnostics and for adapters without complete body-pose telemetry.
        if obj_model is not None and hasattr(self, "body_pose"):
            try:
                object_height = float(object_vertical_extent_m(obj_model))
                object_half_height = 0.5 * object_height
                required_vertical_overlap = gripper_min_vertical_overlap_m(
                    object_height
                )
                object_bottom = float(obj[2] - object_half_height)
                object_top = float(obj[2] + object_half_height)
                span_xy = p2[:2] - p1[:2]
                span_xy_denom = float(np.dot(span_xy, span_xy))
                if span_xy_denom > 1.0e-10:
                    alpha_xy = float(np.dot(obj[:2] - p1[:2], span_xy) / span_xy_denom)
                    closest_xy = p1[:2] + np.clip(alpha_xy, 0.0, 1.0) * span_xy
                    planar_surface_distance = max(
                        0.0,
                        float(np.linalg.norm(obj[:2] - closest_xy))
                        - float(object_xy_radius_m(obj_model)),
                    )
                    intervals: list[dict[str, float]] = []
                    vertical_overlaps: list[float] = []
                    for body_name in (
                        f"{side}_gripper_finger_link1",
                        f"{side}_gripper_finger_link2",
                    ):
                        position, quaternion = self.body_pose(body_name)
                        position = np.asarray(position, dtype=float)
                        unit_quat = _quat_normalize(np.asarray(quaternion, dtype=float))
                        rotation = np.column_stack(
                            [
                                _quat_rotate(unit_quat, np.array([1.0, 0.0, 0.0])),
                                _quat_rotate(unit_quat, np.array([0.0, 1.0, 0.0])),
                                _quat_rotate(unit_quat, np.array([0.0, 0.0, 1.0])),
                            ]
                        )
                        profile_name = "finger_link1" if body_name.endswith("link1") else "finger_link2"
                        profile = R1PRO_GRIPPER_FINGER_COLLISION_BOXES_LOCAL[profile_name]
                        center_local = np.asarray(profile["center"], dtype=float)
                        half_extents = np.asarray(profile["half_extents"], dtype=float)
                        box_center = position + rotation @ center_local
                        half_z = float(np.sum(np.abs(rotation[2, :]) * half_extents))
                        bottom = float(box_center[2] - half_z)
                        top = float(box_center[2] + half_z)
                        # ``between_fingers`` is the terminal contact-window
                        # certificate consumed by grasping.  A pregrasp band
                        # with a positive margin is useful for routing, but
                        # it must not be used to declare a close-ready pose:
                        # the prismatic fingers only move laterally and
                        # cannot repair a remaining vertical gap.
                        vertical_contact_margin = 0.0
                        overlap = min(
                            top, object_top + vertical_contact_margin
                        ) - max(bottom, object_bottom - vertical_contact_margin)
                        vertical_overlaps.append(overlap)
                        intervals.append(
                            {
                                "bottom_z_m": bottom,
                                "top_z_m": top,
                                "overlap_m": float(overlap),
                            }
                        )
                    joint_positions = getattr(
                        self.read_observation(0.0),
                        "joint_positions",
                        {},
                    )
                    closed_contact = _predicted_closed_finger_contact(
                        self,
                        obj_model,
                        obj,
                        side,
                        joint_positions,
                    )
                    close_ready = bool(
                        closed_contact.get("checked", False)
                        and closed_contact.get("all_fingers_contact", False)
                    )
                    box_ready = (
                        0.08 <= alpha_xy <= 0.92
                        and planar_surface_distance <= surface_tolerance
                        and bool(vertical_overlaps)
                        and all(
                            value >= required_vertical_overlap
                            for value in vertical_overlaps
                        )
                        # The open-finger boxes are only a spatial window.
                        # The prismatic close must also have a two-finger
                        # contact solution against the actual supplied mesh;
                        # otherwise a box overlap can still produce a one-sided
                        # empty pinch.
                        and close_ready
                    )
                    between = bool(box_ready)
                    window_geometry = {
                        "window_geometry_source": "projected_finger_boxes",
                        "segment_fraction_xy": alpha_xy,
                        "planar_surface_distance_m": planar_surface_distance,
                        "surface_distance_ready": bool(surface_distance <= surface_tolerance),
                        "vertical_overlap_m": min(vertical_overlaps),
                        "required_vertical_overlap_m": required_vertical_overlap,
                        "finger_vertical_intervals": intervals,
                        "vertical_contact_margin_m": vertical_contact_margin,
                        "closed_finger_contact_ready": close_ready,
                        "closed_finger_contact": closed_contact,
                    }
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, np.linalg.LinAlgError):
                # Keep the conservative origin-based result when a legacy
                # adapter cannot expose a complete collision-box pose.
                window_geometry = {"window_geometry_source": "origin_segment"}
        return {
            "between_fingers": bool(between),
            "segment_fraction": alpha,
            "distance_m": distance,
            "window_radius_m": window_radius,
            "surface_distance_m": surface_distance,
            "surface_tolerance_m": surface_tolerance,
            "finger_midpoint": ((p1 + p2) / 2.0).tolist(),
            "finger_position_1": p1.tolist(),
            "finger_position_2": p2.tolist(),
            "closest_point": closest.tolist(),
            "object_position": obj.tolist(),
            **window_geometry,
        }

    def finger_contact_forces(self, side: str = "left") -> tuple[float, ...]:
        """Net filtered contact force (N) for one gripper's finger sensors."""
        if side not in {"left", "right"}:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        forces: list[float] = []
        for sensor in self.scene_model.contact_sensors:
            if not sensor.body.startswith(f"{side}_gripper_"):
                continue
            s = self.scene.sensors.get(sensor.name)
            if s is None:
                forces.append(0.0)
                continue
            # Use the filtered matrix when the sensor names target objects.
            # net_forces_w includes unrelated contacts (for example a finger
            # touching the table), which can falsely report a successful
            # object grasp.
            matrix = s.data.force_matrix_w
            if sensor.filter and matrix is not None:
                filtered = matrix[0].detach().cpu().numpy()  # (bodies, filters, 3)
                forces.append(float(np.linalg.norm(filtered, axis=-1).max()) if filtered.size else 0.0)
                continue
            net = s.data.net_forces_w
            if net is None:
                forces.append(0.0)
                continue
            values = net[0].detach().cpu().numpy()  # (num_bodies, 3)
            forces.append(float(np.linalg.norm(values, axis=1).max()) if values.size else 0.0)
        return tuple(forces)  # type: ignore[return-value]

    def cleanup(self) -> None:
        """Release render resources before closing the app."""
        try:
            del self.camera
        except Exception:
            pass
