from __future__ import annotations

from types import SimpleNamespace

from r1pro_data_gen.skills import SkillResult
from r1pro_data_gen.skills.posture.workspace import PrepareWorkspace


class _Torso:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, adapter, scene=None, **params):
        del adapter, scene
        self.calls.append(params)
        return SkillResult(True, "torso_move_to", metrics={"final_error_rad": 0.0})


class _WholeBody:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, adapter, scene=None, **params):
        del adapter, scene
        self.calls.append(params)
        return SkillResult(True, "whole_body_pregrasp_transition")


def test_tabletop_profile_skips_when_torso_already_standing() -> None:
    from r1pro_data_gen.skills.posture.torso import TORSO_JOINTS

    class _Standing:
        def read_observation(self, timestamp):
            del timestamp
            return SimpleNamespace(
                joint_positions={name: 0.0 for name in TORSO_JOINTS}
            )

    torso = _Torso()
    skill = PrepareWorkspace(torso)
    result = skill.execute(_Standing(), profile="tabletop")
    assert result.success
    assert result.details.get("already_prepared") is True
    assert torso.calls == []


def test_tabletop_profile_moves_standing_torso() -> None:
    torso = _Torso()
    skill = PrepareWorkspace(torso)
    result = skill.execute(object(), profile="tabletop")
    assert result.success
    assert result.skill == "prepare_workspace"
    assert result.details["profile"] == "tabletop"
    assert torso.calls[0]["target_q"] == [0.0, 0.0, 0.0, 0.0]


def test_floor_profile_uses_whole_body_backend() -> None:
    torso = _Torso()
    whole_body = _WholeBody()
    skill = PrepareWorkspace(torso, whole_body_pregrasp=whole_body)
    scene = SimpleNamespace(
        objects=(
            SimpleNamespace(
                name="floor_cube",
                capabilities=("movable", "graspable"),
                pos=(0.4, 0.0, 0.05),
            ),
        )
    )
    result = skill.execute(object(), scene=scene, profile="floor")
    assert result.success
    assert result.details["profile"] == "floor"
    assert result.details["object_name"] == "floor_cube"
    assert whole_body.calls
    assert not torso.calls


def test_floor_profile_fails_without_object_or_backend() -> None:
    skill = PrepareWorkspace(_Torso())
    result = skill.execute(object(), scene=SimpleNamespace(objects=()), profile="floor")
    assert not result.success
    assert result.details["failure_code"] == "floor_target_unavailable"
