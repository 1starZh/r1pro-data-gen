"""Task-space interpolant stays on the live IK branch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from r1pro_data_gen.methods.manipulation.taskspace import (
    plan_certified_task_path,
    plan_task_path,
)


class _Kin:
    lower = np.full(7, -2.0)
    upper = np.full(7, 2.0)

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        return q[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def _ik_once(self, pos, quat, q_init, **_kwargs):
        del quat
        q = np.asarray(q_init, dtype=float).copy()
        q[:3] = np.asarray(pos, dtype=float)
        return SimpleNamespace(
            success=True,
            q_arm=q,
            position_error=0.0,
            rotation_error=0.0,
        )

    def ik(self, pos, quat, q_init=None):
        return self._ik_once(pos, quat, np.zeros(7) if q_init is None else q_init)


class _JumpKin(_Kin):
    def _ik_once(self, pos, quat, q_init, **_kwargs):
        del pos, quat, q_init
        return SimpleNamespace(
            success=True,
            q_arm=np.full(7, 1.4),
            position_error=0.0,
            rotation_error=0.0,
        )

    def ik(self, pos, quat, q_init=None):
        return self._ik_once(pos, quat, q_init)


def test_plan_task_path_screw_interpolates_coupled_pose():
    from r1pro_data_gen.methods.manipulation.taskspace import _screw_pose

    pos0 = np.array([0.0, 0.0, 0.0])
    quat0 = np.array([1.0, 0.0, 0.0, 0.0])
    pos1 = np.array([0.0, 0.0, 0.2])
    quat1 = np.array([0.70710678, 0.70710678, 0.0, 0.0])
    mid_pos, mid_quat = _screw_pose(pos0, quat0, pos1, quat1, 0.5)

    assert mid_pos[2] == pytest.approx(0.1, abs=1e-6)
    assert abs(float(np.linalg.norm(mid_quat)) - 1.0) < 1e-9


def test_plan_task_path_keeps_chained_ik_continuous():
    start = np.zeros(7)
    planned = plan_task_path(
        _Kin(),
        np.array([0.12, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        start,
    )

    assert planned.success
    assert len(planned.waypoints) >= 3
    deltas = [
        float(np.max(np.abs(planned.waypoints[index] - planned.waypoints[index - 1])))
        for index in range(1, len(planned.waypoints))
    ]
    assert max(deltas) < 0.05
    assert np.allclose(planned.waypoints[-1][:3], [0.12, 0.0, 0.0], atol=1e-9)


def test_plan_task_path_rejects_ik_branch_jump():
    planned = plan_task_path(
        _JumpKin(),
        np.array([0.12, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(7),
    )

    assert not planned.success
    assert "branch jump" in planned.notes


def test_plan_certified_task_path_retiming_uses_straight_interpolant(monkeypatch):
    captured = []

    def certify(_planner, geometric, _scene, **_kwargs):
        captured.append(np.asarray(geometric, dtype=float))
        path = np.asarray(geometric, dtype=float)
        return {
            "success": True,
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "duration": 1.0,
            "dt": 0.5,
            "winding": 1.0,
            "ee_winding": 1.0,
            "status": "SequenceVerified",
            "reason": None,
        }

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.retime_and_validate_path",
        certify,
    )
    result = plan_certified_task_path(
        object(),
        _Kin(),
        np.zeros(7),
        np.array([0.08, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        object(),
        base_xy=(0.0, 0.0),
        base_yaw=0.0,
        full_q_current=np.zeros(22),
        speed_scale=0.2,
        side="left",
    )

    assert result["success"]
    assert result["status"] == "TaskSpaceVerified"
    assert len(captured) == 1
    assert captured[0][0, 0] == 0.0
    assert captured[0][-1, 0] == 0.08


def test_plan_certified_task_path_retries_via_when_straight_collides(monkeypatch):
    attempts = []

    def certify(_planner, geometric, _scene, **_kwargs):
        path = np.asarray(geometric, dtype=float)
        attempts.append(path[-1, 2])
        if len(attempts) == 1:
            return {
                "success": False,
                "status": "collision",
                "reason": "table",
                "failure_stage": "sequence_hppfcl_collision",
            }
        return {
            "success": True,
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "duration": 1.0,
            "dt": 0.5,
            "winding": 1.0,
            "ee_winding": 1.0,
            "status": "SequenceVerified",
            "reason": None,
        }

    monkeypatch.setattr(
        "r1pro_data_gen.methods.manipulation.mplib_path.retime_and_validate_path",
        certify,
    )
    result = plan_certified_task_path(
        object(),
        _Kin(),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.20, 0.0, 0.80]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        object(),
        base_xy=(0.0, 0.0),
        base_yaw=0.0,
        full_q_current=np.zeros(22),
        speed_scale=0.2,
        side="left",
    )

    assert result["success"]
    assert result["status"] == "TaskSpaceVerified"
    assert len(attempts) == 2
