"""Safe, canonical facts exported from a SceneModel to a task planner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from r1pro_data_gen.domain import (
    ObjectCapability,
    ObjectType,
    SceneModel,
    object_xy_half_extents_m,
)
from r1pro_data_gen.robot.chassis import default_footprint_radius_m

from ..navigation.contract import (
    NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M,
    NAVIGATION_GRID_RESOLUTION_M,
    NAVIGATION_INFLATION_CLEARANCE_M,
)


# Three geometry-relative samples keep the planner from committing to the
# centre of a long obstacle while remaining independent of object dimensions.
_APPROACH_ALONG_FRACTIONS = (-1.0, 0.0, 1.0)
_CANDIDATE_EDGE_MARGIN_M = NAVIGATION_GRID_RESOLUTION_M

# Distance beyond which an IK probe is skipped and the target is marked
# unreachable.  The R1Pro left-arm workspace reaches about 1.02 m along its
# long axis, so anything beyond 1.4 m cannot possibly be reachable and probing
# would only burn planner budget on remote fence/wall candidates.
_MAX_IK_PROBE_DISTANCE_M = 1.4

# Candidate heading map so the base forward axis (+x) points at the obstacle.
# The base heading is the world yaw: yaw=0 faces +x, yaw=+pi/2 faces +y.  A
# west approach faces +x east toward the obstacle, an east approach faces -x
# west, a south approach faces +y north, and a north approach faces -y south.
# This is the v18-successful facing: the robot body faces its work position.
_APPROACH_YAW_BY_SIDE = {"west": 0.0, "east": math.pi, "south": math.pi / 2.0, "north": -math.pi / 2.0}

# Candidate shrink-to-reachability parameters.  Geometry-derived approach poses
# sit one footprint plus hard clearance outside the obstacle, plus one grid cell
# of numerical boundary margin.  That margin can push an otherwise reachable
# standoff past the arm's actual workspace boundary (the R1Pro left arm reaches
# far less with a fixed grasp orientation than its bare link reach), so when
# kinematics are available each candidate is re-probed along its approach axis
# and pulled toward the obstacle until the nearest dynamic target is reachable
# or the navigation-safe boundary stops the shrink.
_CANDIDATE_SHRINK_STEP_M = 0.01
_CANDIDATE_SHRINK_MAX_STEPS = 30
# IK probing is substantially more expensive than the geometry checks around
# it.  Probe every few centimetres first rather than solving 30 nearly
# identical poses per candidate; the published pose remains on a verified free
# cell and keeps the same obstacle-margin gate.  This bounds scene-fact
# generation even when a scene has many collision cuboids.
_CANDIDATE_SHRINK_PROBE_STRIDE = 4
# A shrunk stance must keep at least half a grid cell of clearance from every
# inflated obstacle box.  Stopping exactly on a cell boundary leaves the
# free/blocked verdict at the mercy of the floating base position after reset
# (a ~1 mm drift flips it), so the shrink refuses margin below this floor.
_CANDIDATE_SHRINK_MIN_MARGIN_M = 0.025
# A scene can contain many static obstacles, but only a small number of
# interaction stances need expensive IK. Probe one geometry-relative stance
# per side first and expand only when no side is reachable.
_MAX_IK_PROBES_PER_INTERACTION_OBSTACLE = 8
_SHRINK_DIRECTION_BY_SIDE = {
    "west": (1.0, 0.0),
    "east": (-1.0, 0.0),
    "south": (0.0, 1.0),
    "north": (0.0, -1.0),
}


def scene_to_facts(
    scene: SceneModel,
    kinematics: Any = None,
) -> dict[str, Any]:
    """Return a bounded JSON-compatible scene description.

    Only geometry and explicitly useful environment facts are exported. USD
    paths, adapter handles, sensor buffers and internal runtime state stay on
    the trusted side of the planning boundary.  ``kinematics`` is an optional
    robot model; when supplied, each navigation candidate is annotated with
    per-target IK reachability so the planner can pick an approach pose that
    the arm can actually reach.
    """
    robot_facts: dict[str, Any] = {
        "asset": scene.robot.asset,
        "init_pose": list(scene.robot.init_pose),
        "home_joint_names": sorted(scene.robot.home_joint_pos),
    }
    if scene.robot.navigation_footprint_radius_m is not None:
        robot_facts["navigation_footprint_radius_m"] = scene.robot.navigation_footprint_radius_m

    facts: dict[str, Any] = {
        "name": scene.name,
        "world": {
            "gravity": list(scene.world.gravity),
            "dt": scene.world.dt,
            "ground": scene.world.ground,
            "ground_size": list(scene.world.ground_size),
        },
        "robot": robot_facts,
        "objects": [_object_facts(obj) for obj in scene.objects],
        "cameras": [
            {
                "name": camera.name,
                "eye": list(camera.eye),
                "target": list(camera.target),
                "width": camera.width,
                "height": camera.height,
                "data_types": list(camera.data_types),
            }
            for camera in scene.cameras
        ],
        "contact_sensors": [
            {"name": sensor.name, "body": sensor.body, "filter": list(sensor.filter)}
            for sensor in scene.contact_sensors
        ],
        "collision_sensors": [
            {"name": sensor.name, "body": sensor.body, "filter": list(sensor.filter)}
            for sensor in scene.collision_sensors
        ],
    }
    navigation = _navigation_facts(scene, kinematics=kinematics)
    if navigation:
        facts["navigation"] = navigation
    return facts


def scene_facts_from_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Copy already canonical facts while rejecting mutable runtime objects."""
    import json

    try:
        copied = json.loads(json.dumps(data, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"scene facts must be JSON-compatible: {exc}") from exc
    if not isinstance(copied, dict):
        raise ValueError("scene facts must be a JSON object")
    return copied


def object_names(scene_facts: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable object names from canonical scene facts."""
    objects = scene_facts.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("scene facts objects must be an array")
    names: list[str] = []
    for item in objects:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise ValueError("each scene fact object requires a name")
        names.append(item["name"])
    return tuple(names)


def _object_facts(obj: Any) -> dict[str, Any]:
    surfaces = list(obj.surfaces)
    if obj.type is ObjectType.CUBOID and not any(
        surface.name == "top" for surface in surfaces
    ):
        surfaces.append(
            _implicit_top_surface(obj)
        )
    facts: dict[str, Any] = {
        "name": obj.name,
        "type": str(obj.type.value if isinstance(obj.type, ObjectType) else obj.type),
        "pos": list(obj.pos),
        "quat": list(obj.quat),
        "collision_enabled": obj.physics.collision_enabled,
        "kinematic": obj.physics.kinematic,
        "planning_margin": obj.physics.planning_margin,
        "semantic_class": obj.semantic_class,
        "aliases": list(obj.aliases),
        "capabilities": [capability.value for capability in obj.capabilities],
        "regions": [_region_facts(region) for region in obj.regions],
        "surfaces": [
            {
                "name": surface.name,
                "center": list(surface.center),
                "normal": list(surface.normal),
                "size": list(surface.size),
            }
            for surface in surfaces
        ],
    }
    if obj.type is ObjectType.CUBOID:
        facts["size"] = list(obj.size)
        facts["top_z"] = obj.top_z
    elif obj.type is ObjectType.CYLINDER:
        facts["radius"] = obj.radius
        facts["height"] = obj.height
        facts["top_z"] = obj.top_z
    return facts


def _implicit_top_surface(obj: Any) -> Any:
    from r1pro_data_gen.domain.scene import SurfaceModel

    half_x, half_y, half_z = (float(value) / 2.0 for value in obj.size)
    return SurfaceModel(
        name="top",
        center=(0.0, 0.0, half_z),
        normal=(0.0, 0.0, 1.0),
        size=(2.0 * half_x, 2.0 * half_y),
    )


def _region_facts(region: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "name": region.name,
        "shape": region.shape.value,
        "center": list(region.center),
    }
    if region.shape is ObjectType.CUBOID:
        facts["size"] = list(region.size)
    else:
        facts["radius"] = region.radius
        facts["height"] = region.height
    return facts


def _navigation_facts(scene: SceneModel, kinematics: Any = None) -> dict[str, Any]:
    """Expose execution-matched footprint facts and geometry-derived poses.

    Candidates are generated around every collision-enabled primitive.  The
    geometry is deliberately independent of task names or authored planning
    recipes: the runtime uses the same footprint radius and hard clearance
    when constructing its occupancy grid.  Each candidate heading is chosen so
    the arm's long axis (base-frame +y) points at the obstacle, and when
    ``kinematics`` is supplied every candidate is annotated with per-target
    IK reachability.
    """
    radius = scene.robot.navigation_footprint_radius_m
    authored_radius = radius is not None
    if radius is None:
        radius = default_footprint_radius_m()
    clearance = NAVIGATION_INFLATION_CLEARANCE_M
    candidates: list[dict[str, Any]] = []
    grasp_quat = _default_grasp_quat()
    interaction_obstacles = _interaction_obstacle_names(scene)
    ik_probe_counts: dict[str, int] = {}
    ik_reachable_obstacles: set[str] = set()
    for obj in scene.objects:
        if not obj.physics.collision_enabled:
            continue
        half_x, half_y = object_xy_half_extents_m(obj)
        along_y = _candidate_tangent_offsets(half_y)
        along_x = _candidate_tangent_offsets(half_x)
        poses = (
            ("west", (obj.pos[0] - half_x - radius - clearance, obj.pos[1], _APPROACH_YAW_BY_SIDE["west"]), along_y, 1),
            ("east", (obj.pos[0] + half_x + radius + clearance, obj.pos[1], _APPROACH_YAW_BY_SIDE["east"]), along_y, 1),
            ("south", (obj.pos[0], obj.pos[1] - half_y - radius - clearance, _APPROACH_YAW_BY_SIDE["south"]), along_x, 0),
            ("north", (obj.pos[0], obj.pos[1] + half_y + radius + clearance, _APPROACH_YAW_BY_SIDE["north"]), along_x, 0),
        )
        for side, anchor, offsets, varying_axis in poses:
            for offset in offsets:
                pose = [float(anchor[0]), float(anchor[1]), float(anchor[2])]
                pose[varying_axis] += float(offset)
                if side in {"west", "east"}:
                    pose[0] += (
                        -NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M
                        if side == "west"
                        else NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M
                    )
                else:
                    pose[1] += (
                        -NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M
                        if side == "south"
                        else NAVIGATION_CANDIDATE_EXTRA_CLEARANCE_M
                    )
                # The side heading is stable and points into the projected
                # obstacle footprint. Tangent samples are bounded by the
                # navigation grid below, so even the end candidates retain
                # a meaningful facing direction.
                yaw = float(pose[2])
                if _candidate_overlaps_other_obstacle(
                    scene,
                    candidate_name=obj.name,
                    pose=pose,
                    inflation=radius + clearance,
                ):
                    continue
                candidate: dict[str, Any] = {
                    "obstacle_name": obj.name,
                    "side": side,
                    "pose": [round(value, 4) for value in pose],
                    # The base heading is the world yaw: yaw=0 faces +x,
                    # yaw=+pi/2 faces +y.  ``facing`` is that heading as a
                    # unit vector so a planner can check a candidate aims a
                    # direction without converting the angle.
                    "facing": [round(math.cos(yaw), 4), round(math.sin(yaw), 4)],
                    "footprint_radius_m": float(radius),
                    "inflation_clearance_m": clearance,
                }
                should_probe = (
                    kinematics is not None
                    and _has_ik_probe_interface(kinematics)
                    and obj.name in interaction_obstacles
                    and obj.name not in ik_reachable_obstacles
                    and ik_probe_counts.get(obj.name, 0)
                    < _MAX_IK_PROBES_PER_INTERACTION_OBSTACLE
                )
                if should_probe:
                    annotations = _probe_reachability(
                        scene,
                        candidate,
                        kinematics,
                        grasp_quat,
                    )
                    ik_probe_counts[obj.name] = ik_probe_counts.get(obj.name, 0) + 1
                    if any(item.get("reachable") is True for item in annotations):
                        ik_reachable_obstacles.add(obj.name)
                    else:
                        candidate = _shrink_candidate_to_reachable(
                            scene,
                            candidate,
                            kinematics,
                            grasp_quat,
                            float(radius),
                        )
                        if candidate.get("pose") != [round(value, 4) for value in pose]:
                            annotations = _probe_reachability(
                                scene,
                                candidate,
                                kinematics,
                                grasp_quat,
                            )
                            if any(item.get("reachable") is True for item in annotations):
                                ik_reachable_obstacles.add(obj.name)
                    candidate["ik_reachability"] = annotations
                candidates.append(candidate)
    return {
        "footprint_radius_m": float(radius),
        "footprint_radius_source": "scene" if authored_radius else "robot_default",
        "inflation_clearance_m": clearance,
        "approach_candidates": candidates,
    }


def _shrink_candidate_to_reachable(
    scene: SceneModel,
    candidate: Mapping[str, Any],
    kinematics: Any,
    grasp_quat: tuple[float, float, float, float],
    footprint_radius: float,
) -> dict[str, Any]:
    """Pull a geometry-derived candidate toward its obstacle until the nearest
    dynamic target is IK-reachable, staying inside the navigation-safe cells.

    Approach poses are pure obstacle geometry (footprint + clearance + one
    grid-cell boundary margin).  That margin can push an otherwise reachable
    standoff outside the arm workspace once a fixed grasp orientation is
    required, so when kinematics are available the candidate is re-probed
    along its approach axis.  Each shrink step keeps the pose in a
    navigation-free cell and outside every other collision cuboid, matching
    the runtime ``base_navigate_to`` contract.  If no reachable pose exists
    within the safe boundary the original candidate (and its unreachable
    annotation) is returned unchanged.
    """
    direction = _SHRINK_DIRECTION_BY_SIDE.get(candidate.get("side"))
    if direction is None or not _has_ik_probe_interface(kinematics):
        return dict(candidate)
    target = _nearest_dynamic_target(scene, candidate["pose"])
    if target is None:
        return dict(candidate)
    if not _shrink_direction_approaches(direction, candidate["pose"], target.pos):
        return dict(candidate)
    if _target_reachable_at(candidate["pose"], target, grasp_quat, kinematics):
        return dict(candidate)
    pose = [float(value) for value in candidate["pose"]]
    obstacle_name = candidate.get("obstacle_name")
    best_pose: list[float] | None = None
    best_margin = -1.0
    probe_step_m = _CANDIDATE_SHRINK_STEP_M * _CANDIDATE_SHRINK_PROBE_STRIDE
    probe_count = math.ceil(_CANDIDATE_SHRINK_MAX_STEPS / _CANDIDATE_SHRINK_PROBE_STRIDE)
    for _ in range(probe_count):
        pose[0] += direction[0] * probe_step_m
        pose[1] += direction[1] * probe_step_m
        if not _nav_cell_is_free(scene, pose, footprint_radius):
            break
        if _candidate_overlaps_other_obstacle(
            scene,
            candidate_name=obstacle_name,
            pose=pose,
            inflation=footprint_radius + NAVIGATION_INFLATION_CLEARANCE_M,
        ):
            continue
        if not _target_reachable_at(pose, target, grasp_quat, kinematics):
            continue
        margin = _candidate_margin_to_obstacles(scene, pose, footprint_radius)
        if margin < _CANDIDATE_SHRINK_MIN_MARGIN_M:
            continue
        if margin > best_margin:
            best_margin = margin
            best_pose = list(pose)
    if best_pose is not None:
        shrunk = dict(candidate)
        shrunk["pose"] = [round(value, 4) for value in best_pose]
        yaw = float(best_pose[2])
        shrunk["facing"] = [round(math.cos(yaw), 4), round(math.sin(yaw), 4)]
        return shrunk
    return dict(candidate)


def _candidate_margin_to_obstacles(
    scene: SceneModel,
    pose: Sequence[float],
    footprint_radius: float,
) -> float:
    """Minimum distance from a pose to every inflated collision box.

    Used by the shrink loop to refuse stances that sit within half a grid cell
    of an inflated obstacle: those cells flip free/blocked on sub-millimetre
    floating-base drift, so a publishable candidate must keep real clearance.
    """
    inflate = footprint_radius + NAVIGATION_INFLATION_CLEARANCE_M
    min_margin = float("inf")
    for obj in scene.objects:
        if not obj.physics.collision_enabled:
            continue
        hx, hy = object_xy_half_extents_m(obj)
        box = (
            obj.pos[0] - hx - inflate,
            obj.pos[1] - hy - inflate,
            obj.pos[0] + hx + inflate,
            obj.pos[1] + hy + inflate,
        )
        dx = max(box[0] - pose[0], 0.0, pose[0] - box[2])
        dy = max(box[1] - pose[1], 0.0, pose[1] - box[3])
        min_margin = min(min_margin, math.hypot(dx, dy))
    return min_margin


def _nearest_dynamic_target(scene: SceneModel, pose: Sequence[float]):
    """Nearest non-kinematic, collision-enabled scene object to a pose."""
    best = None
    best_distance = float("inf")
    for obj in scene.objects:
        if (
            obj.physics.kinematic
            or not obj.physics.collision_enabled
            or not _is_grasp_target(obj)
        ):
            continue
        distance = math.hypot(obj.pos[0] - pose[0], obj.pos[1] - pose[1])
        if distance < best_distance:
            best, best_distance = obj, distance
    return best


def _shrink_direction_approaches(
    direction: tuple[float, float],
    pose: Sequence[float],
    target: Sequence[float],
) -> bool:
    """True when moving along the approach axis reduces the target distance.

    Candidates around distant obstacles (fences, room walls) would otherwise
    burn the whole shrink budget moving away from the actual interaction
    target; this dot-product gate skips them before any IK probe.
    """
    dot = direction[0] * (target[0] - pose[0]) + direction[1] * (target[1] - pose[1])
    return dot > 0.0


def _target_reachable_at(pose: Sequence[float], target: Any, grasp_quat, kinematics: Any) -> bool:
    """IK-reachability probe for one dynamic target from one base pose."""
    from ..navigation.reachability import assess_interaction_target

    horizontal = math.hypot(target.pos[0] - pose[0], target.pos[1] - pose[1])
    if horizontal > _MAX_IK_PROBE_DISTANCE_M:
        return False
    report = assess_interaction_target(
        candidate_pose_world=pose,
        target_position_world=target.pos,
        target_quaternion=grasp_quat,
        target_frame="grasp_center",
        kinematics=kinematics,
    )
    return bool(report.target_reachable)


def _nav_cell_is_free(
    scene: SceneModel,
    pose: Sequence[float],
    footprint_radius: float,
    resolution: float = NAVIGATION_GRID_RESOLUTION_M,
) -> bool:
    """Goal-cell free check matching the runtime ``base_navigate_to`` grid.

    ``base_navigate_to`` builds its occupancy grid from the live start pose;
    the obstacle boxes are absolute so the authored start pose is a safe proxy
    for whether a world position sits in a free cell.  Keeping this in the
    facts layer means published candidates are rasterization-consistent with
    the skill that consumes them.
    """
    start = scene.robot.init_pose
    pad = 1.5
    xmin = min(float(start[0]), float(pose[0])) - pad
    xmax = max(float(start[0]), float(pose[0])) + pad
    ymin = min(float(start[1]), float(pose[1])) - pad
    ymax = max(float(start[1]), float(pose[1])) + pad
    rows = max(2, int(math.ceil((ymax - ymin) / resolution)))
    cols = max(2, int(math.ceil((xmax - xmin) / resolution)))
    inflate = footprint_radius + NAVIGATION_INFLATION_CLEARANCE_M
    row = int((float(pose[1]) - ymin) / resolution)
    col = int((float(pose[0]) - xmin) / resolution)
    if not (0 <= row < rows and 0 <= col < cols):
        return False
    for obj in scene.objects:
        if not obj.physics.collision_enabled:
            continue
        hx, hy = object_xy_half_extents_m(obj)
        box_xmin = int((obj.pos[0] - hx - inflate - xmin) / resolution)
        box_xmax = int((obj.pos[0] + hx + inflate - xmin) / resolution)
        box_ymin = int((obj.pos[1] - hy - inflate - ymin) / resolution)
        box_ymax = int((obj.pos[1] + hy + inflate - ymin) / resolution)
        if box_xmin <= col <= box_xmax and box_ymin <= row <= box_ymax:
            return False
    return True


def _default_grasp_quat() -> tuple[float, float, float, float]:
    """Left-arm default parallel-jaw grasp orientation (w, x, y, z)."""
    from r1pro_data_gen.robot.robot_config import R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE

    return tuple(float(v) for v in R1PRO_DEFAULT_GRASP_ORIENTATION_BY_SIDE["left"])


def _has_ik_probe_interface(kinematics: Any) -> bool:
    """Duck-type the minimal interface the reachability probe needs.

    Tests inject placeholder kinematics objects; an object without the IK
    entry points simply skips the reachability annotation rather than crashing
    the planner request.
    """
    return hasattr(kinematics, "ik_candidates") and hasattr(
        kinematics, "ee_target_from_grasp_center"
    )


def _probe_reachability(
    scene: SceneModel,
    candidate: Mapping[str, Any],
    kinematics: Any,
    grasp_quat: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Annotate one candidate with per-target IK reachability.

    Only dynamic, collision-enabled objects are considered interaction targets
    (kinematic cuboids are obstacles, not things the arm grasps).  A coarse
    distance pre-filter skips remote targets before the (expensive) IK probe.
    """
    from ..navigation.reachability import assess_interaction_target

    x, y, yaw = (float(value) for value in candidate["pose"])
    c, s = math.cos(yaw), math.sin(yaw)
    results: list[dict[str, Any]] = []
    for obj in scene.objects:
        if (
            obj.physics.kinematic
            or not obj.physics.collision_enabled
            or not _is_grasp_target(obj)
        ):
            continue
        dx, dy = float(obj.pos[0] - x), float(obj.pos[1] - y)
        base_x, base_y = c * dx + s * dy, -s * dx + c * dy
        horizontal_m = math.hypot(base_x, base_y)
        if horizontal_m > _MAX_IK_PROBE_DISTANCE_M:
            results.append(
                {
                    "name": obj.name,
                    "reachable": False,
                    "distance_m": round(horizontal_m, 3),
                }
            )
            continue
        report = assess_interaction_target(
            candidate_pose_world=candidate["pose"],
            target_position_world=obj.pos,
            target_quaternion=grasp_quat,
            target_frame="grasp_center",
            kinematics=kinematics,
        )
        results.append(
            {
                "name": obj.name,
                "reachable": bool(report.target_reachable),
                "distance_m": round(horizontal_m, 3),
                "target_position_base": (
                    [round(float(v), 4) for v in report.target_position_base]
                    if not report.target_reachable
                    else None
                ),
            }
        )
    return results


def _candidate_overlaps_other_obstacle(
    scene: SceneModel,
    *,
    candidate_name: str,
    pose: list[float],
    inflation: float,
) -> bool:
    """Reject a geometry-derived approach pose occupied by another obstacle."""
    x, y = float(pose[0]), float(pose[1])
    for obstacle in scene.objects:
        if obstacle.name == candidate_name or not obstacle.physics.collision_enabled:
            continue
        half_x, half_y = object_xy_half_extents_m(obstacle)
        if (
            obstacle.pos[0] - half_x - inflation <= x <= obstacle.pos[0] + half_x + inflation
            and obstacle.pos[1] - half_y - inflation <= y <= obstacle.pos[1] + half_y + inflation
        ):
            return True
    return False


def _candidate_tangent_offsets(half_extent: float) -> tuple[float, ...]:
    """Return dimension-relative samples that stay inside an obstacle edge."""
    margin = min(float(_CANDIDATE_EDGE_MARGIN_M), float(half_extent))
    usable = min(
        max(0.0, float(half_extent) - margin),
        4.0 * float(NAVIGATION_GRID_RESOLUTION_M),
    )
    values = tuple(
        float(fraction) * usable
        for fraction in (0.0, -1.0, 1.0)
        if fraction in _APPROACH_ALONG_FRACTIONS
    )
    return tuple(dict.fromkeys(round(value, 6) for value in values))


def _is_grasp_target(obj: Any) -> bool:
    """Whether a dynamic object should receive grasp IK annotations.

    Empty capabilities retain legacy scene compatibility. Once a scene
    declares capabilities, only an explicit ``graspable`` object is treated
    as an arm-interaction target; push-only objects must not distort grasp
    candidate ranking.
    """
    capabilities = tuple(getattr(obj, "capabilities", ()))
    return not capabilities or ObjectCapability.GRASPABLE in capabilities


def _interaction_obstacle_names(scene: SceneModel) -> set[str]:
    """Return geometry around which grasp IK is useful.

    IK is an interaction annotation, not a generic property of every wall.
    Restricting it to declared grasp targets and their containing support
    surfaces keeps facts generation bounded for large scenes without using
    benchmark names or a task-specific obstacle list.
    """
    targets = [
        obj
        for obj in scene.objects
        if (
            not obj.physics.kinematic
            and obj.physics.collision_enabled
            and _is_grasp_target(obj)
        )
    ]
    names: set[str] = {obj.name for obj in targets}
    for target in targets:
        target_x, target_y, target_z = (float(value) for value in target.pos)
        for support in scene.objects:
            if (
                support.name == target.name
                or not support.physics.kinematic
                or not support.physics.collision_enabled
            ):
                continue
            half_x, half_y = object_xy_half_extents_m(support)
            inside = (
                abs(target_x - float(support.pos[0])) <= half_x + 0.03
                and abs(target_y - float(support.pos[1])) <= half_y + 0.03
            )
            vertical_gap = target_z - float(support.top_z)
            if inside and -0.08 <= vertical_gap <= 0.20:
                names.add(support.name)
    return names


__all__ = ["object_names", "scene_facts_from_mapping", "scene_to_facts"]
