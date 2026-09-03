from __future__ import annotations

import time
from types import SimpleNamespace

from r1pro_data_gen.domain import Observation, Plan, PlanStage
from r1pro_data_gen.execution import Orchestrator
from r1pro_data_gen.skills import SkillResult


class _Adapter:
    def __init__(self):
        self.calls = []
        self.steps = 0

    def read_observation(self, timestamp):
        return Observation(timestamp=timestamp, base_pose=(1.0, 2.0, 0.0))

    def step(self):
        self.steps += 1


class _Skill:
    def __init__(self, name, result=None):
        self.name = name
        self.result = result or SkillResult(True, name, details={"position": [3.0, 4.0, 0.5]})

    def execute(self, adapter, scene=None, step_hook=None, **params):
        adapter.calls.append((self.name, params))
        return self.result


class _Registry(dict):
    def get(self, name):
        return super().get(name)

    def execute(self, name, adapter, scene=None, step_hook=None, **params):
        return self[name].execute(adapter, scene=scene, step_hook=step_hook, **params)


class _RaisingSkill(_Skill):
    def execute(self, adapter, scene=None, step_hook=None, **params):
        raise RuntimeError("simulator unavailable")


class _LongSkill(_Skill):
    def execute(self, adapter, scene=None, step_hook=None, **params):
        for _ in range(3):
            adapter.step()
        return SkillResult(True, self.name)


class _PlanningBusySkill(_Skill):
    def execute(self, adapter, scene=None, step_hook=None, **params):
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            pass
        return SkillResult(True, self.name)


def test_orchestrator_notifies_current_stage_before_each_skill():
    adapter = _Adapter()
    registry = _Registry(one=_Skill("one"), two=_Skill("two"))
    stages = []
    plan = Plan(
        "generic",
        (
            PlanStage("first", "first", parameters={"skill": "one"}),
            PlanStage("second", "second", parameters={"skill": "two"}),
        ),
    )

    execution = Orchestrator(
        adapter,
        registry,
        stage_hook=stages.append,
    ).run_plan(plan)

    assert execution.success
    assert stages == ["first", "second"]


    adapter = _Adapter()
    registry = _Registry(query=_Skill("query"), move=_Skill("move", SkillResult(True, "move")))
    plan = Plan(
        "generic",
        (
            PlanStage(
                "observe",
                "observe target",
                parameters={"skill": "query"},
                outputs=("position",),
            ),
            PlanStage(
                "move",
                "move to target",
                depends_on=("observe",),
                parameters={
                    "skill": "move",
                    "target": {
                        "ref": "stage.observe.details.position",
                        "value_type": "array",
                        "shape": [3],
                    },
                },
            ),
        ),
    )
    execution = Orchestrator(adapter, registry).run_plan(plan)
    assert execution.success
    assert adapter.calls == [("query", {}), ("move", {"target": [3.0, 4.0, 0.5]})]
    assert execution.stage_calls["move"].raw_parameters["target"]["ref"] == "stage.observe.details.position"
    assert execution.stage_calls["move"].resolved_parameters["target"] == [3.0, 4.0, 0.5]


def test_orchestrator_notifies_stage_end_for_failed_stage():
    adapter = _Adapter()
    registry = _Registry(
        fail=_Skill("fail", SkillResult(False, "fail", details={"reason": "blocked"}))
    )
    events = []
    plan = Plan(
        "generic",
        (PlanStage("fail", "fail", parameters={"skill": "fail"}),),
    )

    execution = Orchestrator(
        adapter,
        registry,
        stage_end_hook=lambda name, success: events.append((name, success)),
    ).run_plan(plan)

    assert not execution.success
    assert events == [("fail", False)]


def test_orchestrator_notifies_stage_end_when_dependency_is_missing():
    adapter = _Adapter()
    registry = _Registry(later=_Skill("later"))
    events = []
    plan = Plan(
        "generic",
        (
            PlanStage(
                "later",
                "later",
                depends_on=("first",),
                parameters={"skill": "later"},
            ),
            PlanStage("first", "first", parameters={"skill": "first"}),
        ),
    )

    execution = Orchestrator(
        adapter,
        registry,
        stage_end_hook=lambda name, success: events.append((name, success)),
    ).run_plan(plan)

    assert not execution.success
    assert execution.failed == "later"
    assert events == [("later", False)]


def test_orchestrator_notifies_stage_end_before_propagating_runtime_failure():
    adapter = _Adapter()
    registry = _Registry(run=_RaisingSkill("run"))
    events = []
    plan = Plan(
        "generic",
        (PlanStage("run", "run", parameters={"skill": "run"}),),
    )

    try:
        Orchestrator(
            adapter,
            registry,
            stage_end_hook=lambda name, success: events.append((name, success)),
        ).run_plan(plan)
    except RuntimeError as exc:
        assert str(exc) == "simulator unavailable"
    else:
        raise AssertionError("runtime failure must propagate")

    assert events == [("run", False)]


    adapter = _Adapter()
    registry = _Registry(
        fail=_Skill("fail", SkillResult(False, "fail", details={"reason": "blocked"})),
        later=_Skill("later"),
    )
    plan = Plan(
        "generic",
        (
            PlanStage("fail", "fail", parameters={"skill": "fail"}),
            PlanStage("later", "later", depends_on=("fail",), parameters={"skill": "later"}),
        ),
    )
    execution = Orchestrator(adapter, registry).run_plan(plan)
    assert not execution.success
    assert execution.failed == "fail"
    assert "later" not in execution.stage_results
    assert adapter.calls == [("fail", {})]


def test_orchestrator_execute_skill_runs_one_call():
    adapter = _Adapter()
    registry = _Registry(grasp_object=_Skill("grasp_object"))
    result, call = Orchestrator(adapter, registry).execute_skill(
        "grasp_object",
        {"object_name": "item"},
        stage_name="step_01_grasp_object",
    )
    assert result.success
    assert call.skill == "grasp_object"
    assert call.success
    assert adapter.calls == [("grasp_object", {"object_name": "item"})]


def test_orchestrator_converts_physical_action_budget_to_structured_failure():
    adapter = _Adapter()
    registry = _Registry(long=_LongSkill("long"))
    result, call = Orchestrator(
        adapter,
        registry,
        max_action_physics_steps=2,
        max_action_seconds=None,
    ).execute_skill("long", stage_name="step_01_long")

    assert result.success is False
    assert result.details["failure_code"] == "action_budget_exceeded"
    assert result.metrics["physics_steps"] == 2
    assert call.success is False
    assert adapter.steps == 2


def test_orchestrator_bounds_planning_without_physics_steps():
    adapter = _Adapter()
    registry = _Registry(plan=_PlanningBusySkill("plan"))
    result, call = Orchestrator(
        adapter,
        registry,
        max_action_physics_steps=None,
        max_action_seconds=0.01,
    ).execute_skill("plan", stage_name="step_01_plan")

    assert result.success is False
    assert result.details["failure_code"] == "action_budget_exceeded"
    assert result.metrics["budget_phase"] == "planning_or_execution"
    assert call.success is False
    assert adapter.steps == 0
