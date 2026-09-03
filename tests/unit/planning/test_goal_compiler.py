"""Deterministic Goal Compiler contract tests."""

from __future__ import annotations

import json

import pytest

from r1pro_data_gen.domain import parse_goal_spec
from r1pro_data_gen.planning.goals.compiler import GoalCompileError, GoalCompiler
from r1pro_data_gen.data.scenes import load_scene_data
from r1pro_data_gen.tasks import load_task_spec
from tests.support import PROJECT_ROOT


ROOT = PROJECT_ROOT
ROOM = load_task_spec("pickplace.tabletop").scene
V5 = ROOT / "tests/fixtures/goal_spec_loop_v5.json"
V7 = ROOT / "tests/fixtures/goal_spec_loop_v7.json"
PUSH = load_task_spec("push.box_to_region").scene


def _goal(path: str):
    scene = load_scene_data(ROOM)
    with open(path, encoding="utf-8") as handle:
        return scene, parse_goal_spec(json.load(handle), scene)


def test_compiler_materializes_geometry_and_observation_requirements():
    scene, spec = _goal(V5)
    # The historical collision invariant is intentionally removed here so the
    # remaining support/place contract can be inspected independently.
    from r1pro_data_gen.domain import GoalSpec

    spec = GoalSpec(
        schema_version=spec.schema_version,
        bindings=spec.bindings,
        required=spec.required,
        invariants=(),
    )
    compiled = GoalCompiler().compile(spec, scene)

    assert compiled.contract_hash
    assert compiled.goal_spec_hash
    assert "entity_states" in compiled.required_observations
    assert "settled_windows" in compiled.required_observations
    support = next(item for item in compiled.predicate_contracts if item["predicate"] == "on_support")
    assert support["evaluation"] == "geometry_plus_settled"


def test_compiler_rejects_region_in_wrong_frame():
    scene, spec = _goal(V7)

    with pytest.raises(GoalCompileError, match="does not match") as error:
        GoalCompiler().compile(spec, scene)

    assert error.value.code == "REGION_GEOMETRY_MISMATCH"


def test_compiler_rejects_collision_goal_without_collision_telemetry():
    scene, spec = _goal(V5)

    with pytest.raises(GoalCompileError, match="no declared collision telemetry") as error:
        GoalCompiler().compile(spec, scene)

    assert error.value.code == "COLLISION_OBSERVATION_UNAVAILABLE"


def test_compiler_resolves_robot_contact_alias_to_declared_base_sensor():
    scene = load_scene_data(PUSH)
    payload = {
        "schema_version": 1,
        "bindings": {
            "object": "scene://push_box",
            "target": "scene://push_goal",
        },
        "required": [
            {
                "predicate": "contact",
                "arguments": {"entity_a": "robot", "entity_b": "object"},
            },
            {
                "predicate": "inside_region",
                "arguments": {
                    "subject": "object",
                    "reference": "target",
                    "region": {
                        "shape": "cuboid",
                        "center": [0.0, 0.0, 0.06],
                        "size": [0.30, 0.30, 0.16],
                    },
                },
            },
            {"predicate": "settled", "arguments": {"subject": "object"}},
        ],
        "invariants": [],
    }

    compiled = GoalCompiler().compile(parse_goal_spec(payload, scene), scene)

    contact = next(
        item for item in compiled.predicate_contracts if item["predicate"] == "contact"
    )
    assert contact["entity_a"] == "base_link"
    assert contact["entity_b"] == "push_box"
