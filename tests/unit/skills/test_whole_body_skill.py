from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from r1pro_data_gen.skills import SkillResult
from r1pro_data_gen.skills.manipulation.whole_body import (
    WholeBodyPregraspTransition,
    WholeBodyTransferObjectBetweenSupports,
    _staging_candidates,
)


class _Scene:
    def __init__(self) -> None:
        self.robot = SimpleNamespace(navigation_footprint_radius_m=0.25)
        self._objects = {
            "item": SimpleNamespace(
                name="item",
                pos=(1.2, 0.8, 0.05),
                radius=0.025,
                height=0.10,
                size=None,
                capabilities=("movable", "graspable"),
            ),
            "floor": SimpleNamespace(
                name="floor",
                pos=(1.2, 0.8, 0.0),
                size=(0.8, 0.8, 0.1),
                capabilities=("supports_objects",),
                top_z=0.05,
            ),
            "table": SimpleNamespace(
                name="table",
                pos=(1.8, 0.8, 1.0),
                size=(0.5, 0.5, 0.1),
                capabilities=("supports_objects",),
                top_z=1.05,
            ),
            "target": SimpleNamespace(
                name="target",
                pos=(1.8, 0.8, 1.056),
                size=(0.16, 0.16, 0.01),
                regions=("place_region",),
                capabilities=("contains_objects",),
            ),
        }

    def object(self, name: str):
        return self._objects[name]

    @property
    def objects(self):
        return tuple(self._objects.values())


class _Adapter:
    def __init__(self) -> None:
        self._object_position = (1.2, 0.8, 0.05)

    def object_position(self, name: str):
        assert name == "item"
        return self._object_position

    def read_observation(self, timestamp):
        del timestamp
        return SimpleNamespace(base_pose=(0.0, 0.0, 0.0))


def test_staging_candidates_are_derived_from_live_object_geometry() -> None:
    candidates = _staging_candidates(_Adapter(), _Scene(), "item")

    assert candidates
    assert all(len(candidate) == 3 for candidate in candidates)
    assert all(np.linalg.norm(np.asarray(candidate[:2]) - np.asarray((1.2, 0.8))) > 0.5 for candidate in candidates)
    # The candidate ring is object-relative; it must not depend on a benchmark
    # coordinate or on the adapter's initial scene object location.
    moved = _Adapter()
    moved._object_position = (3.2, -1.4, 0.05)
    moved_candidates = _staging_candidates(moved, _Scene(), "item")
    assert moved_candidates[0][:2] != candidates[0][:2]


def test_pregrasp_fails_closed_without_selected_kinematics() -> None:
    result = WholeBodyPregraspTransition(None).execute(
        _Adapter(),
        scene=_Scene(),
        object_name="item",
    )

    assert not result.success
    assert result.skill == "whole_body_pregrasp_transition"
    assert result.details["failure_code"] == "kinematics_unavailable"


def test_pregrasp_executes_a_generic_coordinated_posture_on_clear_scene() -> None:
    from r1pro_data_gen.domain import SceneModel
    from r1pro_data_gen.robot.kinematics import R1ProKinematics

    scene = SceneModel.from_dict(
        {
            "name": "clear_pregrasp",
            "world": {"ground": False},
            "robot": {"asset": "asset/r1pro/r1pro.usda"},
            "objects": [
                {
                    "name": "item",
                    "type": "cylinder",
                    "radius": 0.025,
                    "height": 0.10,
                    "pos": [3.0, 3.0, 0.05],
                    "collision_enabled": False,
                    "capabilities": ["movable", "graspable"],
                }
            ],
        }
    )
    kin = R1ProKinematics("asset/r1pro/mplib/robot.urdf", side="left")

    class _KinematicAdapter:
        dt = 1.0 / 60.0

        def __init__(self) -> None:
            self.arm = np.zeros(7, dtype=float)
            self.torso = np.zeros(4, dtype=float)

        def read_observation(self, timestamp):
            del timestamp
            return SimpleNamespace(
                base_pose=(0.0, 0.0, 0.0),
                joint_positions={
                    **{f"left_arm_joint{i}": float(self.arm[i - 1]) for i in range(1, 8)},
                    **{f"torso_joint{i}": float(self.torso[i - 1]) for i in range(1, 5)},
                },
            )

        def body_position(self, name):
            kin.set_auxiliary_q(
                {f"torso_joint{i}": float(self.torso[i - 1]) for i in range(1, 5)}
            )
            return tuple(kin.frame_positions(self.arm, (name,))[0])

        def object_position(self, name):
            assert name == "item"
            return (3.0, 3.0, 0.05)

        def set_targets(self, position, velocity=None):
            del velocity
            for index in range(1, 8):
                name = f"left_arm_joint{index}"
                if name in position:
                    self.arm[index - 1] = float(position[name])
            for index in range(1, 5):
                name = f"torso_joint{index}"
                if name in position:
                    self.torso[index - 1] = float(position[name])

        def step(self):
            return None

    adapter = _KinematicAdapter()
    result = WholeBodyPregraspTransition(kin).execute(
        adapter,
        scene=scene,
        object_name="item",
        speed_scale=1.0,
        settle_steps=0,
    )

    assert result.success, result.details
    assert result.metrics["iterations"] > 0


class _Phase:
    def __init__(self, name: str, success: bool = True) -> None:
        self.name = name
        self.success = success
        self.calls: list[dict[str, object]] = []

    def execute(self, adapter, **params):
        del adapter
        self.calls.append(params)
        return SkillResult(self.success, self.name, details={"phase": self.name})


def test_whole_body_transfer_keeps_complete_phase_and_handoff_contract() -> None:
    grasp = _Phase("grasp_object")
    carry = _Phase("arm_carry_object_to")
    release = _Phase("release_object")
    handoff = _Phase("whole_body_hold_transition")
    skill = WholeBodyTransferObjectBetweenSupports(grasp, carry, release, handoff)

    result = skill.execute(
        _Adapter(),
        scene=_Scene(),
        object_name="item",
        target_region_name="target",
        support_surface_name="table",
        settle_steps=20,
    )

    assert result.success
    assert [phase["name"] for phase in result.details["phases"]] == [
        "grasp",
        "whole_body_handoff",
        "carry_and_place",
        "release_and_settle",
    ]
    assert handoff.calls[0]["target_height_m"] == 1.23
    assert carry.calls[0]["skip_lift"] is True
    assert release.calls[0]["settle_steps"] == 20


class _RetryCarry(_Phase):
    def __init__(self) -> None:
        super().__init__("arm_carry_object_to")
        self._attempt = 0

    def execute(self, adapter, **params):
        del adapter
        self.calls.append(params)
        self._attempt += 1
        if self._attempt == 1:
            return SkillResult(
                False,
                self.name,
                details={
                    "reason": "destination is unreachable from the current manipulation stance",
                    "grasp_context": {"attached": True},
                },
            )
        return SkillResult(True, self.name, details={"phase": self.name})


def test_whole_body_transfer_can_reposition_before_retrying_carry() -> None:
    grasp = _Phase("grasp_object")
    carry = _RetryCarry()
    release = _Phase("release_object")
    handoff = _Phase("whole_body_hold_transition")
    reposition = _Phase("base_navigate_to")

    result = WholeBodyTransferObjectBetweenSupports(
        grasp,
        carry,
        release,
        handoff,
        base_reposition=reposition,
    ).execute(
        _Adapter(),
        scene=_Scene(),
        object_name="item",
        target_region_name="target",
        support_surface_name="table",
    )

    assert result.success
    assert len(reposition.calls) == 1
    assert reposition.calls[0]["purpose"] == "dropoff"
    assert reposition.calls[0]["target_ref"] == "scene://table"
    assert carry.calls[1]["skip_lift"] is True
    assert result.details["phases"][-1]["name"] == "release_and_settle"
