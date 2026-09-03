"""Whole-body feasibility methods for semantic manipulation skills.

The ordinary arm collision checker deliberately models the arm chain only.
That is sufficient for a local end-effector move, but it is not sufficient for
changing the torso while the shoulders and torso sweep through a scene.  This
module adds a conservative, robot-level swept check for the torso, both arm
bases, and the selected arm links.  It also checks a held object's swept
proxy.  No task entity or scene coordinate is encoded here; callers provide
the live joint samples and the scene-derived obstacle exclusions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import hppfcl
import numpy as np

from r1pro_data_gen.domain import ObjectType, SceneModel

from ..collision import (
    LINK_SPHERE_OFFSETS_BY_SIDE,
    LINK_SPHERE_RADII_BY_SIDE,
    CollisionChecker,
    obstacles_from_scene,
)


# Conservative bounding spheres for non-arm robot links.  These describe the
# supplied R1Pro asset, not a particular task.  The arm link radii are reused
# from the regular checker so both planners share one collision approximation.
WHOLE_BODY_FRAME_RADII_BY_SIDE: dict[str, dict[str, float]] = {}
WHOLE_BODY_FRAME_OFFSETS_BY_SIDE: dict[str, dict[str, tuple[float, float, float]]] = {}
for _side in ("left", "right"):
    _radii = {
        "base_link": 0.30,
        "torso_link1": 0.13,
        "torso_link2": 0.14,
        "torso_link3": 0.14,
        "torso_link4": 0.12,
        f"{_side}_arm_base_link": 0.07,
    }
    _radii.update(LINK_SPHERE_RADII_BY_SIDE[_side])
    _other = "right" if _side == "left" else "left"
    _radii[f"{_other}_arm_base_link"] = 0.07
    _radii.update(LINK_SPHERE_RADII_BY_SIDE[_other])
    WHOLE_BODY_FRAME_RADII_BY_SIDE[_side] = _radii
    _offsets = {}
    _offsets.update(LINK_SPHERE_OFFSETS_BY_SIDE[_side])
    _other = "right" if _side == "left" else "left"
    _offsets.update(LINK_SPHERE_OFFSETS_BY_SIDE[_other])
    WHOLE_BODY_FRAME_OFFSETS_BY_SIDE[_side] = _offsets


def _set_torso_auxiliary_q(kin: Any, torso_q: Sequence[float]) -> None:
    if not hasattr(kin, "set_auxiliary_q"):
        return
    values = tuple(float(value) for value in torso_q)
    if len(values) != 4:
        raise ValueError("torso_q must contain four joint positions")
    kin.set_auxiliary_q(
        {f"torso_joint{index}": values[index - 1] for index in range(1, 5)}
    )


class WholeBodyCollisionChecker:
    """Check a robot configuration including torso and both shoulder chains."""

    def __init__(
        self,
        kin: Any,
        obstacles: list[Any] | None = None,
        *,
        side: str = "left",
        frame_radii: dict[str, float] | None = None,
    ) -> None:
        if side not in WHOLE_BODY_FRAME_RADII_BY_SIDE:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.kin = kin
        self.obstacles = list(obstacles or ())
        defaults = WHOLE_BODY_FRAME_RADII_BY_SIDE[side]
        requested = defaults if frame_radii is None else frame_radii
        self.frame_radii = {
            name: float(radius)
            for name, radius in requested.items()
            if _frame_exists(kin, name)
        }
        self.frame_offsets = WHOLE_BODY_FRAME_OFFSETS_BY_SIDE[side]
        self._frame_ids = {name: kin.model.getFrameId(name) for name in self.frame_radii}
        self._request = hppfcl.CollisionRequest()
        self._result = hppfcl.CollisionResult()

    def first_collision_frame(
        self,
        q_arm: np.ndarray,
        torso_q: Sequence[float],
        *,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
        model_to_world_rotation: np.ndarray | None = None,
        model_to_world_translation: np.ndarray | None = None,
    ) -> str | None:
        """Return the first colliding whole-body frame, if any."""
        import pinocchio as pin

        _set_torso_auxiliary_q(self.kin, torso_q)
        q_full = self.kin._full_q(np.asarray(q_arm, dtype=float))
        pin.forwardKinematics(self.kin.model, self.kin.data, q_full)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        for name, radius in self.frame_radii.items():
            pose = self.kin.data.oMf[self._frame_ids[name]]
            position = np.asarray(pose.translation)
            offset = self.frame_offsets.get(name)
            if offset is not None:
                position = position + np.asarray(pose.rotation) @ np.asarray(
                    offset, dtype=float
                )
            world = CollisionChecker._world_position(
                position,
                base_xy,
                base_yaw,
                model_to_world_rotation,
                model_to_world_translation,
            )
            sphere = hppfcl.Sphere(radius)
            sphere_tf = hppfcl.Transform3f(world)
            for obstacle in self.obstacles:
                self._result.clear()
                if hppfcl.collide(
                    obstacle.shape,
                    obstacle.transform,
                    sphere,
                    sphere_tf,
                    self._request,
                    self._result,
                ):
                    return name
        return None

    def is_collision_free(
        self,
        q_arm: np.ndarray,
        torso_q: Sequence[float],
        *,
        base_xy: tuple[float, float] = (0.0, 0.0),
        base_yaw: float = 0.0,
        model_to_world_rotation: np.ndarray | None = None,
        model_to_world_translation: np.ndarray | None = None,
    ) -> bool:
        return self.first_collision_frame(
            q_arm,
            torso_q,
            base_xy=base_xy,
            base_yaw=base_yaw,
            model_to_world_rotation=model_to_world_rotation,
            model_to_world_translation=model_to_world_translation,
        ) is None


def whole_body_path_free(
    checker: WholeBodyCollisionChecker,
    states: Sequence[tuple[np.ndarray, Sequence[float]]],
    *,
    base_xy: tuple[float, float] = (0.0, 0.0),
    base_yaw: float = 0.0,
    model_to_world_rotation: np.ndarray | None = None,
    model_to_world_translation: np.ndarray | None = None,
    dense: int = 12,
    budget_check: Callable[[], None] | None = None,
) -> tuple[bool, dict[str, object]]:
    """Check a swept sequence of ``(arm_q, torso_q)`` states.

    The interpolation is deliberately performed before execution.  A final
    pose that is collision-free does not make a torso sweep safe, so a caller
    must certify every segment before sending joint targets to the simulator.
    """
    if len(states) < 1:
        return False, {"reason": "whole-body path has no states"}
    samples_per_edge = max(1, int(dense))
    for edge, (start, end) in enumerate(zip(states[:-1], states[1:])):
        if budget_check is not None:
            budget_check()
        q0, torso0 = np.asarray(start[0], dtype=float), np.asarray(start[1], dtype=float)
        q1, torso1 = np.asarray(end[0], dtype=float), np.asarray(end[1], dtype=float)
        if q0.shape != q1.shape or torso0.shape != (4,) or torso1.shape != (4,):
            return False, {"reason": "whole-body path state shapes do not match", "edge": edge}
        for sample in range(samples_per_edge + 1):
            if budget_check is not None:
                budget_check()
            alpha = sample / samples_per_edge
            q = q0 + (q1 - q0) * alpha
            torso = torso0 + (torso1 - torso0) * alpha
            collision = checker.first_collision_frame(
                q,
                torso,
                base_xy=base_xy,
                base_yaw=base_yaw,
                model_to_world_rotation=model_to_world_rotation,
                model_to_world_translation=model_to_world_translation,
            )
            if collision is not None:
                return False, {
                    "reason": "whole-body collision",
                    "edge": edge,
                    "sample": sample,
                    "collision_frame": collision,
                    "torso_q": torso.round(5).tolist(),
                    "arm_q": q.round(5).tolist(),
                }
    return True, {"checked": True, "edges": max(0, len(states) - 1), "dense": samples_per_edge}


def held_object_configuration_free(
    scene: SceneModel,
    object_name: str,
    object_position_world: Sequence[float],
    *,
    exclude: Sequence[str] = (),
    include_ground: bool = True,
) -> tuple[bool, dict[str, object]]:
    """Check a conservative sphere proxy for a live held object."""
    try:
        object_model = scene.object(object_name)
    except (AttributeError, KeyError, ValueError) as exc:
        return False, {"reason": "held object is missing from scene", "error": str(exc)}
    position = np.asarray(object_position_world, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        return False, {"reason": "held object position is invalid"}
    if object_model.type is ObjectType.CYLINDER:
        proxy_radius = float(np.hypot(object_model.radius, object_model.height * 0.5))
    else:
        proxy_radius = float(np.linalg.norm(np.asarray(object_model.size, dtype=float)) * 0.5)
    margin = float(object_model.physics.planning_margin or 0.05)
    proxy = hppfcl.Sphere(proxy_radius + margin)
    excluded = tuple(dict.fromkeys((object_name, *[str(name) for name in exclude])))
    obstacles = obstacles_from_scene(scene, exclude=excluded, include_ground=include_ground)
    proxy_tf = hppfcl.Transform3f(position)
    request = hppfcl.CollisionRequest()
    for obstacle in obstacles:
        result = hppfcl.CollisionResult()
        if hppfcl.collide(obstacle.shape, obstacle.transform, proxy, proxy_tf, request, result):
            return False, {
                "reason": "held object collides with obstacle",
                "object_name": object_name,
                "object_position_world": position.round(5).tolist(),
            }
    return True, {
        "checked": True,
        "object_name": object_name,
        "proxy_radius_m": proxy_radius + margin,
        "object_position_world": position.round(5).tolist(),
    }


def _frame_exists(kin: Any, name: str) -> bool:
    try:
        return bool(kin.model.existFrame(name))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        try:
            kin.model.getFrameId(name)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False
        return True


__all__ = [
    "WHOLE_BODY_FRAME_RADII_BY_SIDE",
    "WHOLE_BODY_FRAME_OFFSETS_BY_SIDE",
    "WholeBodyCollisionChecker",
    "held_object_configuration_free",
    "whole_body_path_free",
]
