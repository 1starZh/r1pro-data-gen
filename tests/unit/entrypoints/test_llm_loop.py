"""Unit tests for the bounded external-LLM failure feedback loop.

The loop script (``scripts/planning/run_llm_loop.py``) is loaded via ``importlib.util``
because ``scripts/`` is not a package and the script itself is the module-safe
state machine. Tests inject fake planners and replay callbacks and never import
``run_plan.py`` (which imports Isaac AppLauncher) or load a real URDF.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from r1pro_data_gen.domain import Plan, PlanStage
from r1pro_data_gen.agent import Feedback
from r1pro_data_gen.planning import TaskPlanningResult
from tests.support import PROJECT_ROOT

_SCRIPT = PROJECT_ROOT / "scripts" / "planning" / "run_llm_loop.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_llm_loop", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass processing can resolve the module.
    sys.modules["run_llm_loop"] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()

_SCENE_YAML = (
    "name: test_scene\n"
    "robot:\n"
    "  asset: test_robot\n"
    "world: {}\n"
    "objects: []\n"
)


def config_for(tmp_path, **overrides):
    scene = tmp_path / "scene.yaml"
    scene.write_text(_SCENE_YAML, encoding="utf-8")
    scene_data = yaml.safe_load(_SCENE_YAML)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("placeholder", encoding="utf-8")
    defaults = dict(
        scene=scene,
        task="Move the object to the target.",
        urdf=urdf,
        output_dir=tmp_path / "out",
    )
    defaults.update(overrides)
    task_spec_path = defaults.pop("task_spec_path", None)
    if task_spec_path is None:
        task_spec_path = tmp_path / "task.yaml"
        task_spec_path.write_text(
            "\n".join(
                (
                    "schema_version: task_spec.v2",
                    "id: test.move_object",
                    "family: manipulation",
                    "scene_human_verified: true",
                    "scene:",
                    "  name: test_scene",
                    "  robot:",
                    "    asset: test_robot",
                    "  world: {}",
                    "  objects: []",
                    f"instruction: {defaults['task']!r}",
                    "tags: [test]",
                    "",
                )
            ),
            encoding="utf-8",
        )
    defaults["scene"] = scene_data
    defaults["task_spec_path"] = task_spec_path
    return module.LoopConfig(**defaults)


def _generic_config_for(tmp_path, **overrides):
    config = config_for(tmp_path, **overrides)
    scene_data = yaml.safe_load(
        """name: test_scene
robot:
  asset: test_robot
world: {}
objects:
  - name: item
    type: cuboid
    pos: [0.0, 0.0, 0.05]
    size: [0.1, 0.1, 0.1]
"""
    )
    task_payload = yaml.safe_load(config.task_spec_path.read_text(encoding="utf-8"))
    task_payload["scene"] = scene_data
    config.task_spec_path.write_text(yaml.safe_dump(task_payload, sort_keys=False), encoding="utf-8")
    goal_spec = tmp_path / "goal.json"
    goal_spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bindings": {"item": "scene://item"},
                "required": [
                    {"predicate": "settled", "arguments": {"subject": "item"}}
                ],
                "invariants": [],
            }
        ),
        encoding="utf-8",
    )
    return module.LoopConfig(
        scene=scene_data,
        task=config.task,
        urdf=config.urdf,
        output_dir=config.output_dir,
        task_spec_path=config.task_spec_path,
        goal_spec=goal_spec,
        max_attempts=config.max_attempts,
        feedback_window=config.feedback_window,
        fps=config.fps,
        width=config.width,
        height=config.height,
        physical_gpu_id=config.physical_gpu_id,
        device=config.device,
        stream_replay_logs=config.stream_replay_logs,
    )


def generic_failed_replay(_plan, _attempt_dir, config):
    goal_payload = json.loads(config.goal_spec.read_text(encoding="utf-8"))
    goal_hash = module.goal_spec_sha256(
        module.parse_goal_spec(goal_payload, module.load_scene_data(config.scene))
    )
    return module.ReplayOutcome(
        available=True,
        result={
            "result": "failed",
            "goal_spec_hash": goal_hash,
            "execution": {"failed": "s1", "failure_reason": "motion failed"},
            "evaluation": {"status": "failed"},
        },
    )


def test_generic_replay_failure_feedback_preserves_frozen_goal_spec_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "R1ProKinematics", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "build_default_registry",
        lambda *_args: type("Registry", (), {"llm_descriptions": lambda self: []})(),
    )
    config = _generic_config_for(tmp_path, max_attempts=1)
    planner = FakePlanner([generic_planned_result(config)])
    module.run_loop(
        config,
        planner=planner,
        replay_runner=generic_failed_replay,
        skill_catalog=(),
    )

    feedback = json.loads((tmp_path / "out" / "attempt_01" / "feedback.json").read_text())
    goal_spec = json.loads(config.goal_spec.read_text(encoding="utf-8"))
    expected_hash = module.goal_spec_sha256(
        module.parse_goal_spec(goal_spec, module.load_scene_data(config.scene))
    )
    assert feedback["goal_spec_hash"] == expected_hash


def test_generic_initial_feedback_hash_mismatch_aborts_before_planning(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "R1ProKinematics", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "build_default_registry",
        lambda *_args: type("Registry", (), {"llm_descriptions": lambda self: []})(),
    )
    config = _generic_config_for(tmp_path, max_attempts=1)
    feedback_path = tmp_path / "initial_feedback.json"
    feedback_path.write_text(
        json.dumps(
            Feedback(
                attempt=1,
                failed_stage="s1",
                skill=None,
                request={},
                observations={"failure_type": "gpu", "reason": "failed"},
                discrepancies=(),
                completed_prefix=(),
                goal_spec_hash="b" * 64,
            ).to_json()
        ),
        encoding="utf-8",
    )
    config = module.LoopConfig(
        scene=config.scene,
        task=config.task,
        urdf=config.urdf,
        output_dir=config.output_dir,
        task_spec_path=config.task_spec_path,
        goal_spec=config.goal_spec,
        initial_feedback=feedback_path,
        max_attempts=config.max_attempts,
        feedback_window=config.feedback_window,
        fps=config.fps,
        width=config.width,
        height=config.height,
        physical_gpu_id=config.physical_gpu_id,
        device=config.device,
        stream_replay_logs=config.stream_replay_logs,
    )

    with pytest.raises(ValueError, match="initial feedback goal_spec_hash"):
        module.run_loop(config, planner=FakePlanner([]), replay_runner=never_called, skill_catalog=())


def test_generic_replay_hash_mismatch_aborts_without_replanning(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "R1ProKinematics", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "build_default_registry",
        lambda *_args: type("Registry", (), {"llm_descriptions": lambda self: []})(),
    )
    config = _generic_config_for(tmp_path, max_attempts=2)
    planner = FakePlanner([generic_planned_result(config)])

    def wrong_hash_replay(_plan, _attempt_dir, _config):
        return module.ReplayOutcome(
            available=True,
            result={
                "result": "failed",
                "goal_spec_hash": "b" * 64,
                "evaluation": {"status": "failed"},
            },
        )

    with pytest.raises(ValueError, match="replay goal_spec_hash"):
        module.run_loop(
            config,
            planner=planner,
            replay_runner=wrong_hash_replay,
            skill_catalog=(),
        )
    assert len(planner.calls) == 1


def test_generic_loop_result_carries_frozen_goal_spec_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "R1ProKinematics", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "build_default_registry",
        lambda *_args: type("Registry", (), {"llm_descriptions": lambda self: []})(),
    )
    config = _generic_config_for(tmp_path, max_attempts=1)
    planner = FakePlanner([generic_planned_result(config)])
    def passed_with_frozen_hash(_plan, _attempt_dir, replay_config):
        goal_payload = json.loads(replay_config.goal_spec.read_text(encoding="utf-8"))
        expected_hash = module.goal_spec_sha256(
            module.parse_goal_spec(
                goal_payload,
                module.load_scene_data(replay_config.scene),
            )
        )
        return module.ReplayOutcome(
            available=True,
            result={
                "result": "passed",
                "evaluation_mode": "goal_spec",
                "goal_spec_hash": expected_hash,
                "evaluation": {"status": "succeeded"},
            },
        )

    module.run_loop(
        config,
        planner=planner,
        replay_runner=passed_with_frozen_hash,
        skill_catalog=(),
    )
    payload = json.loads((config.output_dir / "loop_result.json").read_text())
    assert len(payload["goal_spec_hash"]) == 64


    monkeypatch.setattr(module, "R1ProKinematics", lambda *_args: object())
    monkeypatch.setattr(
        module,
        "build_default_registry",
        lambda *_args: type("Registry", (), {"llm_descriptions": lambda self: []})(),
    )
    planner = FakePlanner([failed_result("schema invalid")])
    config = _generic_config_for(tmp_path, max_attempts=1)
    module.run_loop(config, planner=planner, replay_runner=never_called, skill_catalog=())

    feedback = json.loads((tmp_path / "out" / "attempt_01" / "feedback.json").read_text())
    assert len(feedback["goal_spec_hash"]) == 64


class FakePlanner:
    name = "fake"
    model = "fake-model"

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def plan(self, request):
        self.calls.append(request)
        if not self.results:
            raise AssertionError("FakePlanner exhausted its result queue")
        return self.results.pop(0)


def planned_result():
    stage = PlanStage(
        name="s1",
        goal="observe the target",
        depends_on=(),
        parameters={"skill": "query_object_pose", "object_name": "obj"},
    )
    return TaskPlanningResult(
        status="planned",
        plan=Plan(task_name="pick", stages=(stage,)),
        provider="fake",
        model="fake-model",
    )


def generic_planned_result(config):
    result = planned_result()
    goal_payload = json.loads(config.goal_spec.read_text(encoding="utf-8"))
    goal_hash = module.goal_spec_sha256(
        module.parse_goal_spec(goal_payload, module.load_scene_data(config.scene))
    )
    return TaskPlanningResult(
        status=result.status,
        plan=Plan(
            task_name=result.plan.task_name,
            stages=result.plan.stages,
            metadata={"goal_spec_hash": goal_hash},
        ),
        provider=result.provider,
        model=result.model,
    )

def failed_result(reason="schema invalid"):
    return TaskPlanningResult(
        status="failed", reason=reason, provider="fake", model="fake-model"
    )


def unsupported_result():
    return TaskPlanningResult(
        status="unsupported", reason="task not supported", provider="fake", model="fake-model"
    )


def passed_replay(_plan, _attempt_dir, _config):
    return module.ReplayOutcome(
        available=True,
        result={"result": "passed", "evaluation": {"status": "succeeded"}},
    )


def evaluator_failed_replay(_plan, _attempt_dir, _config):
    return module.ReplayOutcome(
        available=True,
        result={
            "result": "failed",
            "execution": {"failed": "s1", "failure_reason": "gripper missed"},
            "evaluation": {
                "status": "failed",
                "failure": {"category": "grasp", "stage": "s1", "reason": "gripper missed"},
            },
        },
    )


def unavailable_replay(_plan, _attempt_dir, _config):
    return module.ReplayOutcome(
        available=False, runtime_error="replay runtime crashed", returncode=1
    )


def never_called(*_args, **_kwargs):
    raise AssertionError("replay must not run on this branch")




def test_loop_parser_accepts_only_task_spec_selector():
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--task", "pickplace.tabletop_complete",
            "--goal-spec", "goal.json",
            "--urdf", "robot.urdf",
        ]
    )
    assert args.goal_spec == Path("goal.json")
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--scene" not in options
    assert "--scene-path" not in options
    assert "--task-module" not in options


def test_planner_request_artifact_contains_frozen_goal_spec(tmp_path):
    request = module.build_request(
        task_description="move the object",
        scene_facts={},
        skill_catalog=(),
        feedbacks=(),
        metadata={},
        feedback_window=3,
        goal_spec={"schema_version": 1, "required": []},
        goal_spec_hash="a" * 64,
    )
    module._write_planner_request(tmp_path, request)
    payload = json.loads((tmp_path / "planner_request.json").read_text())
    assert payload["goal_spec"]["schema_version"] == 1
    assert payload["goal_spec_hash"] == "a" * 64


    command = module.build_replay_command(
        plan_path=tmp_path / "plan.json",
        goal_spec_path=tmp_path / "goal.json",
        config=config_for(tmp_path),
        attempt_dir=tmp_path / "attempt",
    )
    assert "--goal-spec" in command
    assert command[command.index("--task") + 1] == str(config_for(tmp_path).task_spec_path)
    assert "--scene-path" not in command
    assert "--task-module" not in command




def test_product_entrypoint_accepts_task_spec_without_legacy_task_options():
    entrypoint = PROJECT_ROOT / "scripts" / "tasks" / "run_task.py"
    spec = importlib.util.spec_from_file_location("run_task_product", entrypoint)
    product = importlib.util.module_from_spec(spec)
    sys.modules["run_task_product"] = product
    spec.loader.exec_module(product)
    options = {
        option
        for action in product.build_parser()._actions
        for option in action.option_strings
    }
    assert "--task" in options
    assert "--scene" not in options
    assert "--scene-path" not in options
    assert "--instruction" not in options
    assert "--instruction-file" not in options
    assert "--task-module" not in options


def test_build_request_serializes_feedback_objects_to_markdown():
    feedback = Feedback(
        attempt=1,
        failed_stage="s1",
        skill="arm_move_to",
        request={"target_pos": [1.0, 2.0, 3.0]},
        observations={"failure_type": "gpu", "reason": "hit an obstacle"},
        discrepancies=(),
        completed_prefix=(),
    )
    request = module.build_request(
        task_description="pick object",
        scene_facts={},
        skill_catalog=(),
        feedbacks=[feedback],
        metadata={},
        feedback_window=3,
    )
    payload = json.loads(request.constraints["failure_feedback"][0])
    assert payload["schema_version"] == "fact_feedback.v1"
    assert payload["failed_stage"] == "s1"


def test_build_request_carries_last_validated_plan_for_minimal_repair():
    previous = {
        "task_name": "move_object",
        "stages": [
            {
                "name": "standoff",
                "goal": "approach safely",
                "depends_on": [],
                "parameters": {"skill": "arm_move_to", "target_pos": [0.0, 0.0, 0.2]},
                "outputs": [],
                "preconditions": [],
                "postconditions": [],
            }
        ],
        "metadata": {},
    }
    request = module.build_request(
        task_description="pick an object",
        scene_facts={},
        skill_catalog=(),
        feedbacks=(),
        metadata={},
        feedback_window=3,
        previous_plan=previous,
    )
    assert request.constraints["previous_plan"] == previous


def test_success_stops_without_next_planner_call(tmp_path):
    planner = FakePlanner([planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=passed_replay,
        skill_catalog=(),
    )
    assert outcome.status == "succeeded"
    assert outcome.success_attempt == 1
    assert len(planner.calls) == 1


def test_provider_failures_stop_on_second_consecutive_failure(tmp_path):
    planner = FakePlanner(
        [
            TaskPlanningResult(status="failed", reason="timeout", provider="fake", model="m"),
            TaskPlanningResult(status="failed", reason="HTTP 503", provider="fake", model="m"),
        ]
    )
    outcome = module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=never_called,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    assert outcome.reason == "provider unavailable"
    assert outcome.attempts == 2


def test_provider_counter_resets_after_non_provider_round(tmp_path):
    planner = FakePlanner(
        [
            failed_result("DeepSeek timeout"),       # provider: 1
            failed_result("schema invalid"),          # validator: reset to 0
            failed_result("HTTP 500"),                # provider: 1
            failed_result("connection reset"),        # provider: 2 -> stop
        ]
    )
    outcome = module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=never_called,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    assert outcome.reason == "provider unavailable"
    assert outcome.attempts == 4


def test_max_attempts_exhausted_writes_every_failure_round(tmp_path):
    planner = FakePlanner([failed_result("schema invalid") for _ in range(3)])
    outcome = module.run_loop(
        config_for(tmp_path, max_attempts=3),
        planner=planner,
        replay_runner=never_called,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    assert outcome.reason == "max attempts exhausted"
    assert outcome.attempts == 3
    for index in range(1, 4):
        attempt_dir = tmp_path / "out" / f"attempt_{index:02d}"
        assert (attempt_dir / "feedback.json").is_file()
        assert (attempt_dir / "feedback.md").is_file()


def test_validator_failure_then_success(tmp_path):
    planner = FakePlanner([failed_result("schema invalid"), planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=passed_replay,
        skill_catalog=(),
    )
    assert outcome.status == "succeeded"
    assert outcome.success_attempt == 2
    feedback = json.loads((tmp_path / "out" / "attempt_01" / "feedback.json").read_text())
    assert feedback["observations"]["failure_type"] == "validator"


def test_runtime_replan_receives_the_exact_previous_validated_plan(tmp_path):
    planner = FakePlanner([planned_result(), planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path, max_attempts=2),
        planner=planner,
        replay_runner=evaluator_failed_replay,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    assert len(planner.calls) == 2
    previous = planner.calls[1].constraints["previous_plan"]
    assert previous["task_name"] == "pick"
    assert previous["stages"][0]["name"] == "s1"


def test_initial_plan_seeds_a_continuation_request(tmp_path):
    initial_plan = tmp_path / "initial_plan.json"
    initial_plan.write_text(
        json.dumps(module.plan_to_dict(planned_result().plan)),
        encoding="utf-8",
    )
    planner = FakePlanner([failed_result("schema invalid")])
    outcome = module.run_loop(
        config_for(tmp_path, max_attempts=1, initial_plan=initial_plan),
        planner=planner,
        replay_runner=never_called,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    assert planner.calls[0].constraints["previous_plan"]["task_name"] == "pick"


def test_unsupported_round_does_not_stop_loop(tmp_path):
    planner = FakePlanner([unsupported_result(), planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=passed_replay,
        skill_catalog=(),
    )
    assert outcome.status == "succeeded"
    assert outcome.success_attempt == 2
    feedback = json.loads((tmp_path / "out" / "attempt_01" / "feedback.json").read_text())
    assert feedback["observations"]["failure_type"] == "unsupported"


def test_runtime_replay_failures_stop_on_second_consecutive(tmp_path):
    planner = FakePlanner([planned_result(), planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=unavailable_replay,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    assert outcome.reason == "replay unavailable"
    assert outcome.attempts == 2
    for index in (1, 2):
        feedback = json.loads(
            (tmp_path / "out" / f"attempt_{index:02d}" / "feedback.json").read_text()
        )
        assert feedback["observations"]["failure_type"] == "gpu"


def test_replay_runtime_failure_then_recovery(tmp_path):
    calls = {"count": 0}

    def flaky_replay(_plan, _attempt_dir, _config):
        calls["count"] += 1
        if calls["count"] == 1:
            return unavailable_replay(_plan, _attempt_dir, _config)
        return passed_replay(_plan, _attempt_dir, _config)

    planner = FakePlanner([planned_result(), planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=flaky_replay,
        skill_catalog=(),
    )
    assert outcome.status == "succeeded"
    assert outcome.success_attempt == 2


def test_feedback_window_keeps_only_most_recent(tmp_path):
    planner = FakePlanner([failed_result("schema invalid") for _ in range(5)])
    module.run_loop(
        config_for(tmp_path, max_attempts=5),
        planner=planner,
        replay_runner=never_called,
        skill_catalog=(),
    )
    recent = planner.calls[-1].constraints["failure_feedback"]
    assert len(recent) == 3
    assert json.loads(recent[0])["attempt"] == 2
    assert json.loads(recent[-1])["attempt"] == 4


def test_latest_physical_feedback_survives_validator_churn(tmp_path):
    planner = FakePlanner(
        [planned_result(), *[failed_result("schema invalid") for _ in range(4)]]
    )
    module.run_loop(
        config_for(tmp_path, max_attempts=5),
        planner=planner,
        replay_runner=evaluator_failed_replay,
        skill_catalog=(),
    )
    # The bounded conversational window has dropped attempt 1 by the fifth
    # request, but the last physical failure remains an active repair channel.
    assert "active_runtime_feedback" in planner.calls[-1].constraints
    active = planner.calls[-1].constraints["active_runtime_feedback"]
    assert len(active) == 1
    assert json.loads(active[0])["attempt"] == 1


def test_initial_feedback_seeds_first_request_and_active_runtime_channel(tmp_path):
    feedback_path = tmp_path / "prior_feedback.json"
    feedback_path.write_text(
        json.dumps(
            {
                "schema_version": "fact_feedback.v1",
                "attempt": 8,
                "failed_stage": "grasp_cylinder",
                "skill": "gripper_grasp",
                "request": {"object_name": "cylinder"},
                "observations": {
                    "failure_type": "gpu",
                    "reason": "skill returned failure",
                    "measured_lowering_distance_m": 0.0746,
                },
                "discrepancies": [],
                "completed_prefix": [],
                "goal_spec_hash": "",
                "evidence_refs": [],
            }
        ),
        encoding="utf-8",
    )
    planner = FakePlanner([planned_result()])

    module.run_loop(
        config_for(tmp_path, initial_feedback=feedback_path),
        planner=planner,
        replay_runner=passed_replay,
        skill_catalog=(),
    )

    request = planner.calls[0]
    recent = request.constraints["failure_feedback"]
    active = request.constraints["active_runtime_feedback"]
    assert len(recent) == 1
    assert len(active) == 1
    assert json.loads(recent[0])["attempt"] == 8
    assert json.loads(active[0])["failed_stage"] == "grasp_cylinder"


def test_replay_failure_archives_feedback(tmp_path):
    planner = FakePlanner([planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path, max_attempts=1),
        planner=planner,
        replay_runner=evaluator_failed_replay,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    assert outcome.reason == "max attempts exhausted"
    assert outcome.attempts == 1
    feedback = json.loads((tmp_path / "out" / "attempt_01" / "feedback.json").read_text())
    assert feedback["observations"]["failure_type"] == "gpu"
    assert feedback["failed_stage"] == "s1"


def test_replay_failure_without_signal_degrades_gracefully(tmp_path):
    def signal_missing_replay(_plan, _attempt_dir, _config):
        return module.ReplayOutcome(available=True, result={"result": "failed"})

    planner = FakePlanner([planned_result()])
    outcome = module.run_loop(
        config_for(tmp_path, max_attempts=1),
        planner=planner,
        replay_runner=signal_missing_replay,
        skill_catalog=(),
    )
    assert outcome.status == "failed"
    feedback = json.loads((tmp_path / "out" / "attempt_01" / "feedback.json").read_text())
    assert feedback["observations"]["failure_type"] == "gpu"


def test_planned_branch_writes_plan_and_provenance(tmp_path):
    planner = FakePlanner([planned_result()])
    module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=passed_replay,
        skill_catalog=(),
    )
    attempt_dir = tmp_path / "out" / "attempt_01"
    plan = json.loads((attempt_dir / "plan.json").read_text())
    assert plan["task_name"] == "pick"
    raw = json.loads((attempt_dir / "plan.raw.json").read_text())
    assert raw["status"] == "planned"
    assert raw["provider"] == "fake"
    assert set(raw) >= {"status", "provider", "model", "usage", "raw_response"}


def test_planned_branch_archives_exact_planner_request(tmp_path):
    planner = FakePlanner([planned_result()])
    config = config_for(tmp_path, task="导航到桌前抓取圆柱并放到 [1.9, 2.05, 1.056]")
    module.run_loop(
        config,
        planner=planner,
        replay_runner=passed_replay,
        skill_catalog=(),
    )
    request = json.loads(
        (tmp_path / "out" / "attempt_01" / "planner_request.json").read_text()
    )
    assert request["task_description"] == config.task
    assert request["constraints"]["failure_feedback"] == []


def test_success_writes_loop_result(tmp_path):
    planner = FakePlanner([planned_result()])
    module.run_loop(
        config_for(tmp_path),
        planner=planner,
        replay_runner=passed_replay,
        skill_catalog=(),
    )
    payload = json.loads((tmp_path / "out" / "loop_result.json").read_text())
    assert payload["status"] == "succeeded"
    assert payload["success_attempt"] == 1


def test_exhausted_writes_loop_result(tmp_path):
    planner = FakePlanner([failed_result("schema invalid")])
    module.run_loop(
        config_for(tmp_path, max_attempts=1),
        planner=planner,
        replay_runner=never_called,
        skill_catalog=(),
    )
    payload = json.loads((tmp_path / "out" / "loop_result.json").read_text())
    assert payload["status"] == "failed"
    assert payload["reason"] == "max attempts exhausted"


def test_loop_config_rejects_invalid_values(tmp_path):
    base = config_for(tmp_path)
    for field_name in ("max_attempts", "feedback_window", "fps", "width", "height"):
        kwargs = dict(
            scene=base.scene,
            task=base.task,
            urdf=base.urdf,
            output_dir=base.output_dir,
            task_spec_path=base.task_spec_path,
        )
        kwargs[field_name] = 0
        with pytest.raises(ValueError):
            module.LoopConfig(**kwargs)
    with pytest.raises(ValueError):
        module.LoopConfig(
            scene=base.scene, task=base.task, urdf=base.urdf,
            output_dir=base.output_dir, task_spec_path=base.task_spec_path,
            physical_gpu_id=-1,
        )
    with pytest.raises(ValueError):
        module.LoopConfig(
            scene=base.scene, task="  ", urdf=base.urdf,
            output_dir=base.output_dir, task_spec_path=base.task_spec_path,
        )


def test_loop_config_requires_task_spec_identity(tmp_path):
    base = config_for(tmp_path)
    with pytest.raises(ValueError, match="task_spec_path"):
        module.LoopConfig(
            scene=base.scene,
            task=base.task,
            urdf=base.urdf,
            output_dir=base.output_dir,
        )

    other_scene = yaml.safe_load(_SCENE_YAML)
    other_scene["name"] = "other_scene"
    with pytest.raises(ValueError, match="scene data"):
        module.LoopConfig(
            scene=other_scene,
            task=base.task,
            urdf=base.urdf,
            output_dir=base.output_dir,
            task_spec_path=base.task_spec_path,
        )


def test_loop_outcome_to_json_shape():
    outcome = module.LoopOutcome(
        status="succeeded", attempts=2, success_attempt=2, reason="replay passed"
    )
    payload = outcome.to_json()
    assert payload["status"] == "succeeded"
    assert payload["attempts"] == 2
    assert payload["success_attempt"] == 2
    assert payload["last_failure"] is None


def test_loop_outcome_validation():
    with pytest.raises(ValueError):
        module.LoopOutcome(
            status="succeeded", attempts=1, success_attempt=None, reason="missing attempt"
        )
    with pytest.raises(ValueError):
        module.LoopOutcome(
            status="weird", attempts=1, success_attempt=None, reason="bad status"
        )
    with pytest.raises(ValueError):
        module.LoopOutcome(
            status="failed", attempts=-1, success_attempt=None, reason="negative attempts"
        )
    with pytest.raises(ValueError):
        module.LoopOutcome(
            status="failed", attempts=1, success_attempt=None, reason=""
        )


def test_replay_outcome_requires_result_when_available():
    with pytest.raises(TypeError):
        module.ReplayOutcome(available=True)
    with pytest.raises(TypeError):
        module.ReplayOutcome(available="yes")
    with pytest.raises(TypeError):
        module.ReplayOutcome(available=False, returncode="one")


def test_write_loop_result_helper(tmp_path):
    outcome = module.LoopOutcome(
        status="failed", attempts=1, success_attempt=None, reason="provider unavailable"
    )
    module.write_loop_result(tmp_path / "out", outcome)
    payload = json.loads((tmp_path / "out" / "loop_result.json").read_text())
    assert payload["reason"] == "provider unavailable"


def test_run_replay_runner_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    outcome = module.run_replay(tmp_path / "plan.json", tmp_path / "out", config_for(tmp_path))
    assert outcome.available is False
    assert "run_plan.py" in (outcome.runtime_error or "")
    assert outcome.returncode is None


def test_run_replay_parses_result_with_marker(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".run_plan_result_written.json").write_text(
            '{"marker": "result_written"}', encoding="utf-8"
        )
        (out_dir / "result.json").write_text(
            json.dumps({"result": "failed", "evaluation": {"status": "failed"}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    attempt_dir = tmp_path / "out" / "attempt_01"
    config = config_for(tmp_path, goal_spec=tmp_path / "goal.json")
    outcome = module.run_replay(tmp_path / "plan.json", attempt_dir, config)
    assert outcome.available is True
    assert outcome.returncode == 1
    assert outcome.result is not None and outcome.result["result"] == "failed"
    assert captured["cmd"][0] == sys.executable
    assert captured["cmd"][1] == str(module.PROJECT_ROOT / "scripts" / "tasks" / "run_plan.py")
    assert captured["cwd"] == module.PROJECT_ROOT


def test_run_replay_subprocess_oserror(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise OSError("simulator launch failed")

    monkeypatch.setattr(subprocess, "run", boom)
    attempt_dir = tmp_path / "out" / "attempt_01"
    outcome = module.run_replay(
        tmp_path / "plan.json",
        attempt_dir,
        config_for(tmp_path, goal_spec=tmp_path / "goal.json"),
    )
    assert outcome.available is False
    assert "simulator launch failed" in (outcome.runtime_error or "")
    assert outcome.returncode is None


def test_run_replay_builds_exact_command(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = config_for(tmp_path, goal_spec=tmp_path / "goal.json")
    attempt_dir = tmp_path / "out" / "attempt_01"
    plan_path = attempt_dir / "plan.json"
    module.run_replay(plan_path, attempt_dir, config)
    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("scripts/tasks/run_plan.py")
    assert "--external-llm-plan" in cmd
    pairs = [
        ("--task", str(config.task_spec_path)),
        ("--plan", str(plan_path)),
        ("--output-dir", str(attempt_dir)),
        ("--physical-gpu-id", "6"),
        ("--device", "cuda:0"),
        ("--fps", "30"),
        ("--width", "960"),
        ("--height", "544"),
    ]
    for flag, value in pairs:
        assert flag in cmd
        assert cmd[cmd.index(flag) + 1] == value
    assert "--scene-path" not in cmd
    assert "--instruction" not in cmd
    assert "--task-module" not in cmd


def test_run_replay_missing_marker_yields_bounded_runtime_error(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom: " + "x" * 5000)

    monkeypatch.setattr(subprocess, "run", fake_run)
    attempt_dir = tmp_path / "out" / "attempt_01"
    outcome = module.run_replay(
        tmp_path / "plan.json",
        attempt_dir,
        config_for(tmp_path, goal_spec=tmp_path / "goal.json"),
    )
    assert outcome.available is False
    assert outcome.returncode == 2
    assert "boom" in (outcome.runtime_error or "")
    assert len(outcome.runtime_error or "") <= 2500


def test_run_replay_writes_stage_calls_from_result(tmp_path, monkeypatch):
    result_payload = {
        "result": "failed",
        "execution": {
            "stage_results": {
                "s1": {
                    "skill": "arm_move_to",
                    "success": False,
                    "call": {"params": {"target_pos": [1, 2, 3]}},
                }
            }
        },
    }

    def fake_run(cmd, **kwargs):
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".run_plan_result_written.json").write_text("{}", encoding="utf-8")
        (out_dir / "result.json").write_text(json.dumps(result_payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    attempt_dir = tmp_path / "out" / "attempt_01"
    outcome = module.run_replay(
        tmp_path / "plan.json",
        attempt_dir,
        config_for(tmp_path, goal_spec=tmp_path / "goal.json"),
    )
    assert outcome.available is True
    calls = json.loads((attempt_dir / "stage_calls.json").read_text())
    assert calls["s1"]["params"]["target_pos"] == [1, 2, 3]


def test_raw_provenance_never_contains_api_key(tmp_path):
    result = TaskPlanningResult(
        status="planned",
        plan=Plan(
            task_name="pick",
            stages=(PlanStage(name="s1", goal="g", parameters={"skill": "query_object_pose"}),),
        ),
        provider="fake",
        model="fake-model",
        raw_response={
            "status": "planned",
            "api_key": "sk-TOPSECRET0123456789",
            "usage": {"total_tokens": 3},
        },
    )
    module._write_raw_provenance(tmp_path, result)
    text = (tmp_path / "plan.raw.json").read_text()
    assert '"provider": "fake"' in text
    assert '"model": "fake-model"' in text
    assert "raw_response" in text
    assert "sk-TOPSECRET0123456789" not in text


def test_failed_round_writes_planner_result_not_plan(tmp_path):
    planner = FakePlanner([failed_result("DeepSeek timeout")])
    module.run_loop(
        config_for(tmp_path, max_attempts=1),
        planner=planner,
        replay_runner=never_called,
        skill_catalog=(),
    )
    attempt_dir = tmp_path / "out" / "attempt_01"
    payload = json.loads((attempt_dir / "planner_result.json").read_text())
    assert payload["status"] == "failed"
    assert payload["provider"] == "fake"
    assert payload["model"] == "fake-model"
    assert "reason" in payload
    assert not (attempt_dir / "plan.json").exists()


def test_planner_result_reason_is_secret_stripped(tmp_path):
    result = TaskPlanningResult(
        status="failed",
        plan=None,
        reason="ProviderError: api_key sk-TOPSECRET0123456789 rejected",
        provider="fake",
        model="fake-model",
        raw_response=None,
        usage={},
    )
    feedback = Feedback(
        attempt=1,
        failed_stage=None,
        skill=None,
        request={},
        observations={
            "failure_type": "provider",
            "reason": "provider error",
            "raw_error": "provider error",
        },
        discrepancies=(),
        completed_prefix=(),
    )
    module._write_planner_result(tmp_path, result, feedback)
    text = (tmp_path / "planner_result.json").read_text()
    assert "sk-TOPSECRET0123456789" not in text
    assert "<redacted>" in text


def test_keyboard_interrupt_at_cli_boundary(tmp_path, monkeypatch):
    def interrupted(config):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "run_loop", interrupted)
    scene = tmp_path / "scene.yaml"
    scene.write_text(_SCENE_YAML, encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("placeholder", encoding="utf-8")
    task_spec = config_for(tmp_path).task_spec_path
    rc = module.main(
        [
            "--task", str(task_spec),
            "--goal-spec", str(tmp_path / "goal.json"),
            "--urdf", str(urdf),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 130


def test_parser_defaults():
    args = module.build_parser().parse_args(
        ["--task", "pickplace.tabletop_complete", "--urdf", "robot.urdf"]
    )
    assert args.max_attempts == 10
    assert args.feedback_window == 3
    assert args.fps == 30
    assert args.width == 960
    assert args.height == 544
    assert args.physical_gpu_id == 6
    assert args.device == "cuda:0"
    assert args.output_dir is None
    assert args.initial_feedback is None
    assert args.initial_plan is None


def test_parser_has_no_scene_or_task_package_selector():
    options = {
        option
        for action in module.build_parser()._actions
        for option in action.option_strings
    }
    assert "--scene" not in options
    assert "--scene-path" not in options
    assert "--task-module" not in options


def test_main_maps_outcome_to_exit_codes(tmp_path, monkeypatch, capsys):
    scene = tmp_path / "scene.yaml"
    scene.write_text(_SCENE_YAML, encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("placeholder", encoding="utf-8")
    task_spec = config_for(tmp_path).task_spec_path
    args = [
        "--task", str(task_spec),
        "--goal-spec", str(tmp_path / "goal.json"),
        "--urdf", str(urdf),
        "--output-dir", str(tmp_path / "out"),
    ]

    monkeypatch.setattr(
        module,
        "run_loop",
        lambda config: module.LoopOutcome(
            status="succeeded",
            attempts=2,
            success_attempt=2,
            reason="replay passed",
            last_failure=None,
        ),
    )
    rc = module.main(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "replay passed" in out
    assert "DEEPSEEK" not in out

    monkeypatch.setattr(
        module,
        "run_loop",
        lambda config: module.LoopOutcome(
            status="failed",
            attempts=10,
            success_attempt=None,
            reason="max attempts exhausted",
            last_failure=None,
        ),
    )
    rc = module.main(args)
    assert rc == 1


def test_main_rejects_nonpositive_loop_bounds(tmp_path, monkeypatch):
    scene = tmp_path / "scene.yaml"
    scene.write_text(_SCENE_YAML, encoding="utf-8")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("placeholder", encoding="utf-8")
    task_spec = config_for(tmp_path).task_spec_path
    monkeypatch.setattr(module, "run_loop", lambda config: (_ for _ in ()).throw(
        AssertionError("planner must not be constructed for invalid bounds")
    ))
    for flag in ("--max-attempts", "--feedback-window"):
        rc = module.main(
            [
                "--task", str(task_spec),
                "--goal-spec", str(tmp_path / "goal.json"),
                "--urdf", str(urdf),
                "--output-dir", str(tmp_path / "out"),
                flag, "0",
            ]
        )
        assert rc == 2


def test_interrupt_writes_interrupted_record(tmp_path):
    class InterruptPlanner:
        name = "fake"
        model = "fake-model"

        def plan(self, request):
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        module.run_loop(
            config_for(tmp_path),
            planner=InterruptPlanner(),
            replay_runner=never_called,
            skill_catalog=(),
        )
    data = json.loads((tmp_path / "out" / "loop_result.json").read_text())
    assert data["status"] == "interrupted"
    assert data["attempts"] == 1
    assert data["reason"] == "interrupted"
