from types import SimpleNamespace

import pytest

from r1pro_data_gen.agent.observation import build_agent_observation
from r1pro_data_gen.skills import SkillResult


class _Adapter:
    def read_observation(self, timestamp):
        del timestamp
        return SimpleNamespace(base_pose=(0.0, 0.0, 0.0))

    def finger_contact_forces(self, side="left"):
        if side == "left":
            return (0.1, 0.2)
        if side == "right":
            return (0.3, 0.4)
        raise KeyError(side)


def test_agent_observation_reports_both_gripper_contacts() -> None:
    payload = build_agent_observation(
        adapter=_Adapter(),
        remaining_actions=4,
    )
    live = payload["live"]
    assert live["contacts_left"] == [0.1, 0.2]
    assert live["contacts_right"] == [0.3, 0.4]
    assert live["contacts"] == {"left": [0.1, 0.2], "right": [0.3, 0.4]}


def test_agent_observation_does_not_include_a_plan_skeleton() -> None:
    payload = build_agent_observation(
        adapter=_Adapter(),
        remaining_actions=4,
        skill_catalogue=[{"name": "grasp_object"}],
    )
    assert "plan_skeleton" not in payload
    assert "candidate_skills" not in payload
    assert "skill_catalogue" not in payload


def test_failed_grasp_carries_a_family_recovery_hint() -> None:
    payload = build_agent_observation(
        adapter=_Adapter(),
        remaining_actions=3,
        last_skill="grasp_object",
        last_parameters={"object_name": "item"},
        last_result=SkillResult(
            False,
            "grasp_object",
            details={
                "failure_code": "workspace_not_prepared",
                "reason": "object is below the current torso workspace",
            },
        ),
    )
    action = payload["last_action"]
    assert action["failure_code"] == "workspace_not_prepared"
    assert "prepare_workspace" in action["recovery_hint"]
    assert "pick_cylinder" not in action["recovery_hint"]


def test_live_objects_report_size_support_and_current_stance() -> None:
    table = SimpleNamespace(
        name="support_a",
        pos=(1.0, 0.0, 0.375),
        size=(0.8, 0.8, 0.75),
        radius=None,
        height=None,
        capabilities=("supports_objects",),
        surfaces=("top",),
        top_z=0.75,
        physics=SimpleNamespace(kinematic=True),
        regions=(),
    )
    item = SimpleNamespace(
        name="movable",
        pos=(1.0, 0.0, 0.85),
        size=None,
        radius=0.03,
        height=0.12,
        capabilities=("movable", "graspable"),
        surfaces=(),
        physics=SimpleNamespace(kinematic=False),
        regions=(),
    )

    class _PoseAdapter(_Adapter):
        def object_position(self, name):
            if name == "movable":
                return (1.0, 0.0, 0.85)
            if name == "support_a":
                return (1.0, 0.0, 0.375)
            raise KeyError(name)

    payload = build_agent_observation(
        adapter=_PoseAdapter(),
        scene=SimpleNamespace(objects=(table, item)),
        scene_facts={
            "navigation": {
                "footprint_radius_m": 0.25,
                "inflation_clearance_m": 0.05,
                "approach_candidates": [
                    {
                        "obstacle_name": "movable",
                        "side": "west",
                        "pose": [0.0, 0.0, 0.0],
                        "ik_reachability": [{"reachable": True, "distance_m": 0.4}],
                    }
                ],
            }
        },
        remaining_actions=3,
    )
    record = payload["live"]["objects"]["movable"]
    assert record["position"] == [1.0, 0.0, 0.85]
    assert record["size"] == {"shape": "cylinder", "radius": 0.03, "height": 0.12}
    assert "graspable" in record["capabilities"]
    assert record["on_support"] == "support_a"
    assert record["planar_distance_m"] == pytest.approx(1.0)
    assert record["reachable_from_here"] is True
    assert "skill_catalogue" not in payload
    assert "pick_cylinder" not in str(payload["live"]["objects"])


def test_reachable_from_here_is_false_when_stance_is_far_from_ik_candidate() -> None:
    item = SimpleNamespace(
        name="movable",
        pos=(3.0, 0.0, 0.1),
        size=(0.05, 0.05, 0.1),
        radius=None,
        height=None,
        capabilities=("movable", "pushable"),
        surfaces=(),
        physics=SimpleNamespace(kinematic=False),
        regions=(),
    )

    class _PoseAdapter(_Adapter):
        def object_position(self, name):
            return (3.0, 0.0, 0.1) if name == "movable" else (_ for _ in ()).throw(KeyError(name))

    payload = build_agent_observation(
        adapter=_PoseAdapter(),
        scene=SimpleNamespace(objects=(item,)),
        scene_facts={
            "navigation": {
                "footprint_radius_m": 0.25,
                "inflation_clearance_m": 0.05,
                "approach_candidates": [
                    {
                        "obstacle_name": "movable",
                        "side": "west",
                        "pose": [2.5, 0.0, 0.0],
                        "ik_reachability": [{"reachable": True}],
                    }
                ],
            }
        },
        remaining_actions=2,
    )
    record = payload["live"]["objects"]["movable"]
    assert record["reachable_from_here"] is False
    assert "pushable" in record["capabilities"]
    assert record["on_support"] is None
