"""Collision checking with inflated obstacles for arm path planning.

The arm links are approximated as spheres (radius per link, FK position from
the Pinocchio model); static scene obstacles (table, cylinder, ground) are
inflated by a safety margin. A joint configuration is collision-free iff no
link proxy intersects any inflated obstacle.  Selected links may use the
supplied MPlib collision mesh instead of a sphere when a contact-sensitive
motion needs the asset geometry exactly.

Obstacles are derived from a :class:`SceneModel` (``obstacles_from_scene``),
so the module stays task-agnostic -- the planner/skill decides which objects
are obstacles for a given motion segment (e.g. the cylinder is excluded while
the fingers descend to surround it). Uses hpp-fcl for the geometric checks
(primitives only, no mesh files needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hppfcl
import numpy as np

from r1pro_data_gen.domain import ObjectModel, ObjectType, SceneModel
from r1pro_data_gen.robot.robot_config import (
    R1PRO_GRIPPER_LINK_COLLISION_CENTER_LOCAL,
    R1PRO_GRIPPER_LINK_COLLISION_RADIUS_M,
)

# Arm link -> sphere radius (m). The arm is ~0.73 m over 7 links; the shoulder
# and elbow links are the thickest (~0.05), while the wrist and gripper are
# slimmer. A uniform 0.05 makes the wrist falsely collide with the table when
# the fingers surround a low object (the grasp pose), so the distal links are
# sized closer to their real geometry.
_LINK_RADII = (0.05, 0.05, 0.05, 0.05, 0.045, 0.035, 0.035)
LINK_SPHERE_RADII_BY_SIDE: dict[str, dict[str, float]] = {
    side: {
        **{f"{side}_arm_link{i}": radius for i, radius in enumerate(_LINK_RADII, 1)},
        f"{side}_gripper_link": R1PRO_GRIPPER_LINK_COLLISION_RADIUS_M,
        f"{side}_gripper_finger_link1": 0.02,
        f"{side}_gripper_finger_link2": 0.02,
    }
    for side in ("left", "right")
}
LINK_SPHERE_RADII = LINK_SPHERE_RADII_BY_SIDE["left"]
LINK_SPHERE_OFFSETS_BY_SIDE: dict[str, dict[str, tuple[float, float, float]]] = {
    side: {f"{side}_gripper_link": R1PRO_GRIPPER_LINK_COLLISION_CENTER_LOCAL}
    for side in ("left", "right")
}

# Default inflation margins (m) per obstacle kind.
DEFAULT_TABLE_MARGIN = 0.05
DEFAULT_CYLINDER_MARGIN = 0.04
DEFAULT_GROUND_MARGIN = 0.01

_COLLISION_MESH_CACHE: dict[str, object | None] = {}


def collision_mesh_for_body(body_name: str):
    """Load one supplied robot collision mesh, cached for this process.

    The R1Pro USD/URDF and the Pinocchio model use the link-frame OBJ meshes
    under ``asset/r1pro/mplib/meshes``.  The similarly named files directly
    under ``asset/r1pro/meshes`` use a Blender/USD axis convention and must not
    be used as link-frame collision geometry.  Keeping this loader in the
    shared collision module lets runtime adapters and motion certificates use
    the same supplied geometry.
    """
    if body_name in _COLLISION_MESH_CACHE:
        return _COLLISION_MESH_CACHE[body_name]
    try:
        from r1pro_data_gen.robot.robot_config import R1PRO_USDA_RELPATH

        repo_root = Path(__file__).resolve().parents[3]
        mesh_path = (
            repo_root
            / R1PRO_USDA_RELPATH.parent
            / "mplib"
            / "meshes"
            / f"{body_name}.obj"
        )
        if not mesh_path.is_file():
            _COLLISION_MESH_CACHE[body_name] = None
            return None
        mesh = hppfcl.MeshLoader().load(str(mesh_path))
    except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
        mesh = None
    _COLLISION_MESH_CACHE[body_name] = mesh
    return mesh


@dataclass(frozen=True, slots=True)
class Obstacle:
    """An inflated static obstacle (fcl shape + world transform)."""

    shape: hppfcl.ShapeBase
    transform: hppfcl.Transform3f


def object_obstacle(obj: ObjectModel, margin: float) -> Obstacle:
    """An inflated obstacle for one scene object (world frame)."""
    if obj.type is ObjectType.CUBOID:
        sx, sy, sz = obj.size
        shape = hppfcl.Box(sx + 2 * margin, sy + 2 * margin, sz + 2 * margin)
    elif obj.type is ObjectType.CYLINDER:
        # hppfcl's Python binding accepts the full cylinder length and exposes
        # the stored half length through ``halfLength``. Inflate the authored
        # top/bottom surfaces by ``margin`` while preserving the primitive's
        # actual vertical extent.
        shape = hppfcl.Cylinder(obj.radius + margin, obj.height + 2 * margin)
    else:  # pragma: no cover - guarded by ObjectType membership
        raise ValueError(f"no obstacle shape for object type {obj.type}")
    return Obstacle(shape, hppfcl.Transform3f(np.array(obj.pos, dtype=float)))


def ground_obstacle(margin: float = DEFAULT_GROUND_MARGIN) -> Obstacle:
    """Ground plane (top z=0) inflated by ``margin`` so links stay above it."""
    shape = hppfcl.Box(30.0, 30.0, 0.1 + 2 * margin)
    return Obstacle(shape, hppfcl.Transform3f(np.array([0.0, 0.0, -(0.05 + margin)])))


def obstacles_from_scene(
    scene: SceneModel,
    margins: dict[str, float] | None = None,
    exclude: tuple[str, ...] = (),
    default_margin: float = DEFAULT_TABLE_MARGIN,
    include_ground: bool = True,
) -> list[Obstacle]:
    """Build the inflated obstacle set for a scene.

    Args:
        scene: the runtime scene model.
        margins: per-object-name inflation margin override.
        exclude: object names that are NOT obstacles for this motion segment.
        default_margin: margin used for objects without an explicit margin.
        include_ground: include the inflated ground plane.
    """
    margins = margins or {}
    obstacles = [
        object_obstacle(
            obj,
            margins.get(
                obj.name,
                obj.physics.planning_margin if obj.physics.planning_margin is not None else default_margin,
            ),
        )
        for obj in scene.objects
        if obj.name not in exclude and obj.physics.collision_enabled
    ]
    if include_ground and scene.world.ground:
        obstacles.append(ground_obstacle())
    return obstacles


class CollisionChecker:
    """Check arm configurations against an inflated obstacle scene."""

    def __init__(
        self,
        kin,
        obstacles: list[Obstacle] | None = None,
        link_radii: dict[str, float] | None = None,
        link_offsets: dict[str, tuple[float, float, float]] | None = None,
        link_meshes: dict[str, object] | None = None,
    ) -> None:
        self.kin = kin
        self.obstacles = obstacles if obstacles is not None else []
        side = getattr(kin, "side", "left")
        self.link_radii = link_radii if link_radii is not None else LINK_SPHERE_RADII_BY_SIDE[side]
        self.link_offsets = (
            link_offsets if link_offsets is not None else LINK_SPHERE_OFFSETS_BY_SIDE[side]
        )
        self.link_meshes = dict(link_meshes or {})
        self._frame_ids = {
            name: kin.model.getFrameId(name) for name in self.link_radii
        }
        self._request = hppfcl.CollisionRequest()
        self._result = hppfcl.CollisionResult()

    @staticmethod
    def _world_position(
        position: np.ndarray,
        base_xy: tuple[float, float],
        base_yaw: float,
        model_to_world_rotation: np.ndarray | None = None,
        model_to_world_translation: np.ndarray | None = None,
    ) -> np.ndarray:
        if model_to_world_rotation is not None and model_to_world_translation is not None:
            rotation = np.asarray(model_to_world_rotation, dtype=float)
            translation = np.asarray(model_to_world_translation, dtype=float)
            if rotation.shape != (3, 3) or translation.shape != (3,):
                raise ValueError("model_to_world transform must be a 3x3 rotation and 3-vector")
            return rotation @ np.asarray(position, dtype=float) + translation
        c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))
        x = c * float(position[0]) - s * float(position[1]) + float(base_xy[0])
        y = s * float(position[0]) + c * float(position[1]) + float(base_xy[1])
        return np.array([x, y, float(position[2])])

    @staticmethod
    def _world_rotation(
        rotation: np.ndarray,
        base_yaw: float,
        model_to_world_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map a link-frame rotation into the live world frame."""
        link_rotation = np.asarray(rotation, dtype=float)
        if link_rotation.shape != (3, 3):
            raise ValueError("link rotation must be a 3x3 matrix")
        if model_to_world_rotation is not None:
            registration = np.asarray(model_to_world_rotation, dtype=float)
            if registration.shape != (3, 3):
                raise ValueError("model_to_world rotation must be a 3x3 matrix")
            return registration @ link_rotation
        c, s = float(np.cos(base_yaw)), float(np.sin(base_yaw))
        base_rotation = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )
        return base_rotation @ link_rotation

    def _link_shape_transform(
        self,
        name: str,
        pose: object,
        base_xy: tuple[float, float],
        base_yaw: float,
        model_to_world_rotation: np.ndarray | None,
        model_to_world_translation: np.ndarray | None,
    ) -> tuple[object, hppfcl.Transform3f]:
        """Return one link's configured shape and its world transform."""
        position = np.asarray(pose.translation, dtype=float)
        if name in self.link_meshes:
            rotation = self._world_rotation(
                np.asarray(pose.rotation, dtype=float),
                base_yaw,
                model_to_world_rotation,
            )
            world = self._world_position(
                position,
                base_xy,
                base_yaw,
                model_to_world_rotation,
                model_to_world_translation,
            )
            return self.link_meshes[name], hppfcl.Transform3f(rotation, world)
        offset = self.link_offsets.get(name)
        if offset is not None:
            position = position + np.asarray(pose.rotation) @ np.asarray(offset, dtype=float)
        world = self._world_position(
            position,
            base_xy,
            base_yaw,
            model_to_world_rotation,
            model_to_world_translation,
        )
        return hppfcl.Sphere(float(self.link_radii[name])), hppfcl.Transform3f(world)

    def is_collision_free(
        self,
        q_arm: np.ndarray,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
        *,
        model_to_world_rotation: np.ndarray | None = None,
        model_to_world_translation: np.ndarray | None = None,
    ) -> bool:
        """True if no link sphere intersects any inflated obstacle."""
        import pinocchio as pin

        q_full = self.kin._full_q(np.asarray(q_arm, dtype=float))
        pin.forwardKinematics(self.kin.model, self.kin.data, q_full)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        for name, radius in self.link_radii.items():
            pose = self.kin.data.oMf[self._frame_ids[name]]
            shape, shape_tf = self._link_shape_transform(
                name,
                pose,
                base_xy,
                base_yaw,
                model_to_world_rotation,
                model_to_world_translation,
            )
            for obs in self.obstacles:
                self._result.clear()
                if hppfcl.collide(obs.shape, obs.transform, shape, shape_tf, self._request, self._result):
                    return False
        return True

    def first_collision_link(
        self,
        q_arm: np.ndarray,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
        *,
        model_to_world_rotation: np.ndarray | None = None,
        model_to_world_translation: np.ndarray | None = None,
    ) -> str | None:
        """Name of the first colliding link (for diagnostics)."""
        import pinocchio as pin

        q_full = self.kin._full_q(np.asarray(q_arm, dtype=float))
        pin.forwardKinematics(self.kin.model, self.kin.data, q_full)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        for name, radius in self.link_radii.items():
            pose = self.kin.data.oMf[self._frame_ids[name]]
            shape, shape_tf = self._link_shape_transform(
                name,
                pose,
                base_xy,
                base_yaw,
                model_to_world_rotation,
                model_to_world_translation,
            )
            for obs in self.obstacles:
                self._result.clear()
                if hppfcl.collide(obs.shape, obs.transform, shape, shape_tf, self._request, self._result):
                    return name
        return None


def check_path(
    checker: CollisionChecker,
    waypoints: list[np.ndarray] | tuple[np.ndarray, ...],
    base_xy: tuple[float, float] = (0.0, 0.0),
    dense: int = 20,
    base_yaw: float = 0.0,
    *,
    model_to_world_rotation: np.ndarray | None = None,
    model_to_world_translation: np.ndarray | None = None,
) -> tuple[bool, int, str | None]:
    """Check a waypoint path (dense interpolation between consecutive waypoints)."""
    for i, (q0, q1) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        for step in range(dense + 1):
            t = step / dense
            q = np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * t
            if model_to_world_rotation is None and model_to_world_translation is None:
                free = checker.is_collision_free(q, base_xy, base_yaw)
            else:
                free = checker.is_collision_free(
                    q,
                    base_xy,
                    base_yaw,
                    model_to_world_rotation=model_to_world_rotation,
                    model_to_world_translation=model_to_world_translation,
                )
            if not free:
                if model_to_world_rotation is None and model_to_world_translation is None:
                    link = checker.first_collision_link(q, base_xy, base_yaw)
                else:
                    link = checker.first_collision_link(
                        q,
                        base_xy,
                        base_yaw,
                        model_to_world_rotation=model_to_world_rotation,
                        model_to_world_translation=model_to_world_translation,
                    )
                return False, i, link
    return True, len(waypoints) - 1, None


def carried_object_path_free(
    kin,
    q_path: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
    scene: SceneModel,
    grasp_context,
    *,
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_yaw: float = 0.0,
    exclude: tuple[str, ...] = (),
) -> tuple[bool, dict[str, object]]:
    """Check a conservative swept proxy for an object held by the gripper."""
    if not hasattr(kin, "grasp_center_fk"):
        return True, {"checked": False, "reason": "kinematics has no grasp_center_fk"}
    try:
        object_model = scene.object(grasp_context.object_name)
    except (AttributeError, KeyError, ValueError) as exc:
        return False, {"checked": False, "reason": f"held object is missing from scene: {exc}"}
    if object_model.type is ObjectType.CYLINDER:
        proxy_radius = float(np.hypot(object_model.radius, object_model.height * 0.5))
    else:
        proxy_radius = float(np.linalg.norm(np.asarray(object_model.size, dtype=float)) * 0.5)
    margin = float(object_model.physics.planning_margin or DEFAULT_TABLE_MARGIN)
    proxy = hppfcl.Sphere(proxy_radius + margin)
    obstacles = obstacles_from_scene(scene, exclude=tuple(exclude), include_ground=True)
    offset_world = np.asarray(grasp_context.object_to_grasp_center_world, dtype=float)
    path = np.asarray(q_path, dtype=float)
    for index, q in enumerate(path):
        center_model, _ = kin.grasp_center_fk(q)
        center_world = CollisionChecker._world_position(center_model, base_xy, base_yaw)
        object_world = center_world - offset_world
        proxy_tf = hppfcl.Transform3f(object_world)
        for obstacle in obstacles:
            result = hppfcl.CollisionResult()
            if hppfcl.collide(
                obstacle.shape,
                obstacle.transform,
                proxy,
                proxy_tf,
                hppfcl.CollisionRequest(),
                result,
            ):
                return False, {
                    "checked": True,
                    "collision_index": index,
                    "object_name": grasp_context.object_name,
                    "object_position_world": object_world.tolist(),
                }
    return True, {
        "checked": True,
        "samples": int(len(path)),
        "proxy_radius_m": proxy_radius + margin,
        "object_name": grasp_context.object_name,
    }


__all__ = [
    "CollisionChecker",
    "LINK_SPHERE_RADII",
    "LINK_SPHERE_RADII_BY_SIDE",
    "LINK_SPHERE_OFFSETS_BY_SIDE",
    "Obstacle",
    "check_path",
    "carried_object_path_free",
    "collision_mesh_for_body",
    "ground_obstacle",
    "object_obstacle",
    "obstacles_from_scene",
]
