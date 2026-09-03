from __future__ import annotations

from collections import OrderedDict

import pytest

from r1pro_data_gen.domain import (
    GoalPredicate,
    GoalSpec,
    ObjectModel,
    ObjectType,
    RobotModel,
    SceneModel,
    WorldModel,
    goal_spec_sha256,
    goal_spec_to_dict,
    parse_goal_spec,
)


def _scene() -> SceneModel:
    return SceneModel(
        name="goal_contract_scene",
        world=WorldModel(),
        robot=RobotModel(asset="asset/robot.usda"),
        objects=(
            ObjectModel(
                name="item",
                type=ObjectType.CYLINDER,
                pos=(0.0, 0.0, 0.1),
                radius=0.03,
                height=0.2,
            ),
            ObjectModel(
                name="destination",
                type=ObjectType.CUBOID,
                pos=(1.0, 0.0, 0.4),
                size=(0.5, 0.5, 0.8),
            ),
        ),
    )


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "bindings": {
            "item": "scene://item",
            "destination": "scene://destination",
        },
        "required": [
            {
                "predicate": "on_support",
                "arguments": {
                    "subject": "item",
                    "support": "destination",
                    "surface": {"center": [0.0, 0.0, 0.4], "size": [0.5, 0.5]},
                    "subject_half_height_m": 0.1,
                },
            },
            {
                "predicate": "released",
                "arguments": {"subject": "item"},
            },
            {
                "predicate": "settled",
                "arguments": {"subject": "item"},
            },
        ],
        "invariants": [
            {
                "predicate": "collision_free",
                "arguments": {"subject": "robot"},
            }
        ],
    }


def test_parse_goal_spec_rejects_noncanonical_entity_argument_names() -> None:
    payload = _valid_payload()
    payload["required"] = [
        {
            "predicate": "on_support",
            "arguments": {"object": "item", "support": "destination"},
        },
        {"predicate": "released", "arguments": {"object": "item"}},
    ]

    with pytest.raises(ValueError, match="on_support.*subject"):
        parse_goal_spec(payload, _scene())


def test_parse_goal_spec_rejects_base_vocabulary_as_attachment_effector() -> None:
    payload = _valid_payload()
    payload["required"][1]["arguments"]["effector"] = "robot"

    with pytest.raises(ValueError, match="effector.*robot base"):
        parse_goal_spec(payload, _scene())


def test_parse_goal_spec_allows_effector_agnostic_attachment() -> None:
    payload = _valid_payload()
    payload["required"] = [
        {"predicate": "attached", "arguments": {"subject": "item"}}
    ]

    spec = parse_goal_spec(payload, _scene())

    assert spec.required[0].arguments == {"subject": "item"}


def test_parse_goal_spec_rejects_string_support_surface() -> None:
    payload = _valid_payload()
    payload["required"] = [
        {
            "predicate": "on_support",
            "arguments": {
                "subject": "item",
                "support": "destination",
                "surface": "top",
                "subject_half_height_m": 0.1,
            },
        }
    ]

    with pytest.raises(TypeError, match="surface must be an object"):
        parse_goal_spec(payload, _scene())


def test_parse_goal_spec_rejects_skill_and_evaluator_fields() -> None:
    for field, value in (
        ("skill", "gripper_grasp"),
        ("evaluator", "PickPlaceEvaluator"),
        ("failure_rule", "lower_then_grasp"),
    ):
        payload = _valid_payload()
        payload[field] = value

        with pytest.raises(ValueError, match="unknown fields"):
            parse_goal_spec(payload, _scene())


def test_parse_goal_spec_rejects_binding_to_unknown_scene_object() -> None:
    payload = _valid_payload()
    payload["bindings"] = {"item": "scene://missing"}

    with pytest.raises(ValueError, match="unknown scene object"):
        parse_goal_spec(payload, _scene())


def test_parse_goal_spec_normalizes_scene_qualified_binding() -> None:
    payload = _valid_payload()
    payload["bindings"] = {
        "item": "scene://goal_contract_scene/item",
        "destination": "scene://goal_contract_scene/destination",
    }

    spec = parse_goal_spec(payload, _scene())

    assert dict(spec.bindings) == {
        "item": "scene://item",
        "destination": "scene://destination",
    }


def test_goal_spec_requires_at_least_one_required_predicate() -> None:
    payload = _valid_payload()
    payload["required"] = []

    with pytest.raises(ValueError, match="required predicates"):
        parse_goal_spec(payload, _scene())


def test_goal_spec_round_trip_uses_closed_public_shape() -> None:
    spec = parse_goal_spec(_valid_payload(), _scene())

    assert isinstance(spec, GoalSpec)
    assert spec.required[0] == GoalPredicate(
        predicate="on_support",
        arguments={
            "subject": "item",
            "support": "destination",
            "surface": {"center": [0.0, 0.0, 0.4], "size": [0.5, 0.5]},
            "subject_half_height_m": 0.1,
        },
    )
    assert goal_spec_to_dict(spec) == _valid_payload()


def test_goal_spec_hash_is_stable_across_mapping_order() -> None:
    first = parse_goal_spec(_valid_payload(), _scene())
    payload = _valid_payload()
    payload["bindings"] = OrderedDict(
        (("destination", "scene://destination"), ("item", "scene://item"))
    )
    second = parse_goal_spec(payload, _scene())

    assert goal_spec_sha256(first) == goal_spec_sha256(second)
    assert len(goal_spec_sha256(first)) == 64


def test_parse_goal_spec_accepts_direct_scene_reference_in_arguments() -> None:
    """A provider may repeat the full scene:// URI as an entity argument; the
    parser must resolve it against the bound scene objects."""
    payload = _valid_payload()
    payload["required"][0]["arguments"]["subject"] = "scene://item"
    payload["required"][1]["arguments"]["subject"] = "scene://item"
    spec = parse_goal_spec(payload, _scene())
    assert spec.required[0].arguments["subject"] == "scene://item"
    assert spec.required[1].arguments["subject"] == "scene://item"


def test_parse_goal_spec_rejects_direct_scene_reference_to_unbound_object() -> None:
    """A scene:// URI naming an object that is not bound is still rejected."""
    payload = _valid_payload()
    payload["required"][0]["arguments"]["subject"] = "scene://missing"
    with pytest.raises(ValueError, match="unknown binding"):
        parse_goal_spec(payload, _scene())


def test_parse_goal_spec_rejects_inside_region_with_string_region() -> None:
    """A provider emitting ``region: "top"`` (a name) must be rejected so the
    goal planner can repair the response instead of yielding an UNKNOWN
    predicate that can never verify."""
    payload = _valid_payload()
    payload["required"].insert(
        0,
        {
            "predicate": "inside_region",
            "arguments": {"subject": "item", "reference": "destination", "region": "top"},
        },
    )
    with pytest.raises(ValueError, match="region must be an object"):
        parse_goal_spec(payload, _scene())


def test_parse_goal_spec_accepts_valid_cuboid_region_object() -> None:
    """A well-formed cuboid region object passes and is verified."""
    payload = _valid_payload()
    payload["required"].insert(
        0,
        {
            "predicate": "inside_region",
            "arguments": {
                "subject": "item",
                "reference": "destination",
                "region": {"shape": "cuboid", "center": [0.0, 0.0, 0.5], "size": [0.5, 0.5, 0.3]},
            },
        },
    )
    spec = parse_goal_spec(payload, _scene())
    assert any(p.predicate == "inside_region" for p in spec.required)


def test_base_at_pose_accepts_explicit_robot_subject_for_provider_compatibility() -> None:
    payload = {
        "schema_version": 1,
        "bindings": {},
        "required": [
            {
                "predicate": "base_at_pose",
                "arguments": {
                    "subject": "robot",
                    "pose": [1.0, 2.0, 0.5],
                },
            }
        ],
        "invariants": [],
    }

    spec = parse_goal_spec(payload, _scene())

    assert spec.required[0].arguments["subject"] == "robot"


def test_base_at_pose_rejects_non_robot_subject() -> None:
    with pytest.raises(ValueError, match="robot base"):
        GoalPredicate(
            predicate="base_at_pose",
            arguments={"subject": "item", "pose": [1.0, 2.0, 0.5]},
        )
