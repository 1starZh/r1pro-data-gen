from __future__ import annotations

from types import SimpleNamespace

from r1pro_data_gen.skills import SkillResult
from r1pro_data_gen.skills.manipulation.transfer import TransferObjectBetweenSupports


class _Scene:
    def __init__(self):
        self._objects = {
            "item": SimpleNamespace(name="item", capabilities=("movable", "graspable")),
            "table": SimpleNamespace(
                name="table", capabilities=("supports_objects",), size=(1.0, 1.0, 0.1)
            ),
            "target": SimpleNamespace(
                name="target", capabilities=("contains_objects",), size=(0.2, 0.2, 0.01), regions=("region",)
            ),
        }

    def object(self, name):
        return self._objects[name]


class _Phase:
    def __init__(self, name: str, success: bool = True):
        self.name = name
        self.success = success
        self.calls = []

    def execute(self, adapter, **params):
        self.calls.append(params)
        return SkillResult(self.success, self.name, details={"phase": self.name})


def test_transfer_executes_complete_semantic_sequence() -> None:
    grasp = _Phase("grasp_object")
    carry = _Phase("arm_carry_object_to")
    release = _Phase("release_object")
    skill = TransferObjectBetweenSupports(grasp, carry, release)

    result = skill.execute(
        object(),
        scene=_Scene(),
        object_name="item",
        target_region_name="target",
        support_surface_name="table",
        side="right",
        settle_steps=20,
    )

    assert result.success
    assert [item["name"] for item in result.details["phases"]] == [
        "grasp",
        "carry_and_place",
        "release_and_settle",
    ]
    assert carry.calls[0]["target_region_name"] == "target"
    assert carry.calls[0]["support_surface_name"] == "table"
    assert carry.calls[0]["side"] == "right"
    assert release.calls[0]["settle_steps"] == 20


def test_transfer_stops_before_carry_when_grasp_fails() -> None:
    grasp = _Phase("grasp_object", success=False)
    carry = _Phase("arm_carry_object_to")
    release = _Phase("release_object")
    result = TransferObjectBetweenSupports(grasp, carry, release).execute(
        object(),
        scene=_Scene(),
        object_name="item",
        target_region_name="target",
        support_surface_name="table",
    )

    assert not result.success
    assert result.details["failure_code"] == "grasp_phase_failed"
    assert not carry.calls
    assert not release.calls


def test_transfer_accepts_legacy_scenes_without_capability_annotations() -> None:
    scene = _Scene()
    for model in scene._objects.values():
        model.capabilities = ()
    grasp = _Phase("grasp_object")
    carry = _Phase("arm_carry_object_to")
    release = _Phase("release_object")
    result = TransferObjectBetweenSupports(grasp, carry, release).execute(
        object(),
        scene=scene,
        object_name="item",
        target_region_name="target",
        support_surface_name="table",
    )
    assert result.success
