from __future__ import annotations

from types import SimpleNamespace

import pytest

from r1pro_data_gen.skills import SkillResult
from r1pro_data_gen.skills.manipulation.release import ReleaseObject


class _Adapter:
    def __init__(self):
        self.steps = 0

    def lock_joint_mask(self, **kwargs):
        del kwargs

    def step(self):
        self.steps += 1

    def read_observation(self, timestamp):
        del timestamp
        return SimpleNamespace(base_pose=(0.0, 0.0, 0.0), joint_positions={})

    def gripper_object_alignment(self, object_name, side="left"):
        del object_name, side
        return {"finger_midpoint": [0.4, 0.1, 1.05]}


class _Open:
    def execute(self, adapter, **params):
        del adapter
        assert params["object_name"] == "item"
        assert params["open_value"] > 0
        return SkillResult(True, "gripper_set", metrics={"detached": True})


class _Move:
    def __init__(self, success: bool = True):
        self.targets = []
        self.success = success

    def execute(self, adapter, **params):
        del adapter
        self.targets.append(list(params["target_pos"]))
        assert params["target_frame"] == "grasp_center"
        assert "item" in params["exclude_objects"]
        return SkillResult(self.success, "arm_move_to")


class _Directional:
    def __init__(self):
        self.calls = 0
        self.direction = None
        self.distance = None

    def execute(self, adapter, **params):
        del adapter
        self.calls += 1
        self.direction = list(params["direction"])
        self.distance = params["distance"]
        assert params.get("object_name") is None
        assert params.get("until_contact") is False
        return SkillResult(True, "arm_move_directional")


class _Scene:
    def object(self, name):
        return SimpleNamespace(name=name)


def test_release_object_opens_and_settles() -> None:
    adapter = _Adapter()
    result = ReleaseObject(_Open()).execute(
        adapter, object_name="item", side="left", settle_steps=4
    )
    assert result.success
    assert adapter.steps == 4
    assert result.details["object_name"] == "item"
    assert result.metrics["lifted"] == 0.0


def test_release_object_lifts_hand_after_opening() -> None:
    adapter = _Adapter()
    move = _Move()
    result = ReleaseObject(_Open(), move).execute(
        adapter,
        scene=_Scene(),
        object_name="item",
        side="left",
        settle_steps=4,
    )
    assert result.success
    assert adapter.steps == 4
    assert move.targets
    assert move.targets[0][0] == pytest.approx(0.4)
    assert move.targets[0][1] == pytest.approx(0.1)
    assert move.targets[0][2] == pytest.approx(1.15)
    assert result.metrics["lifted"] == 1.0


def test_release_object_prefers_cartesian_lift_over_planned_move() -> None:
    adapter = _Adapter()
    move = _Move(success=False)
    directional = _Directional()
    result = ReleaseObject(_Open(), move, directional).execute(
        adapter,
        scene=_Scene(),
        object_name="item",
        side="left",
        settle_steps=4,
    )
    assert result.success
    assert directional.calls == 1
    assert directional.direction == [0.0, 0.0, 1.0]
    assert directional.distance == pytest.approx(0.10)
    assert not move.targets
    assert result.metrics["lifted"] == 1.0
