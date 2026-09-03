from __future__ import annotations

import importlib.util
import json
import sys
from types import MappingProxyType

from tests.support import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "tasks" / "run_plan.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_plan_entrypoint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_plan_entrypoint"] = module
    spec.loader.exec_module(module)
    return module


def test_json_safe_converts_mappingproxy_nested_in_result():
    module = _load_module()

    converted = module._json_safe(
        {"evaluation": [{"observed": MappingProxyType({"value": 1})}]}
    )

    assert converted == {"evaluation": [{"observed": {"value": 1}}]}
    json.dumps(converted)

def test_replay_entrypoint_requires_task_spec_goal_spec_and_plan():
    module = _load_module()
    actions = {option: action for action in module.build_parser()._actions for option in action.option_strings}

    assert "--goal-spec" in actions
    assert actions["--goal-spec"].required is False
    assert "--task" in actions
    assert actions["--task"].required is False
    assert "--plan" in actions
    assert actions["--plan"].required is False
    assert "--scene" not in actions
    assert "--scene-path" not in actions
    assert "--evidence-hz" in actions
    assert actions["--evidence-hz"].default == 10.0
    assert "--max-action-seconds" in actions
    assert actions["--max-action-seconds"].default == 600.0


def test_generic_entrypoint_loads_goal_spec_against_scene(tmp_path):
    module = _load_module()
    scene_file = tmp_path / "scene.yaml"
    scene_file.write_text(
        "name: scene\n"
        "robot:\n"
        "  asset: robot.usd\n"
        "world: {}\n"
        "objects:\n"
        "  - name: object\n"
        "    type: cuboid\n"
        "    pos: [0.0, 0.0, 0.5]\n"
        "    size: [0.1, 0.1, 0.1]\n",
        encoding="utf-8",
    )
    goal_file = tmp_path / "goal.json"
    goal_file.write_text(
        '{"schema_version":1,"bindings":{"subject":"scene://object"},'
        '"required":[{"predicate":"settled","arguments":{"subject":"subject"}}],'
        '"invariants":[]}',
        encoding="utf-8",
    )

    from r1pro_data_gen.data.scenes import load_scene_yaml

    spec = module.load_goal_spec(goal_file, load_scene_yaml(scene_file))

    assert spec.bindings["subject"] == "scene://object"
    assert module.goal_spec_sha256(spec)
