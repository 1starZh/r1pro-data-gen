from __future__ import annotations

import json

from r1pro_data_gen.agent import AgentLoop
from r1pro_data_gen.evaluation.predicates import (
    PredicateEvaluation,
    PredicateStatus,
    VerificationReport,
    VerificationStatus,
)
from r1pro_data_gen.agent.contracts import AGENT_SCHEMA_VERSION
from r1pro_data_gen.planning.llm.providers.protocol import ProviderResponse
from r1pro_data_gen.skills import SkillResult


class _Adapter:
    def read_observation(self, timestamp):
        del timestamp
        return type("Obs", (), {"base_pose": (0.0, 0.0, 0.0)})()


class _Orchestrator:
    def __init__(self):
        self.adapter = _Adapter()
        self.scene = None
        self.calls = []

    def execute_skill(self, skill, parameters, stage_name=None):
        self.calls.append((skill, dict(parameters), stage_name))
        return SkillResult(True, skill, details={"failure_code": None}), None


class _Provider:
    name = "fake"
    model = "fake"

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.seen = []

    def complete(self, *, system: str, user: str) -> ProviderResponse:
        self.seen.append((system, user))
        text = self.payloads.pop(0)
        return ProviderResponse(text=text, provider=self.name, model=self.model)


def _act(skill: str, **parameters) -> str:
    return json.dumps(
        {
            "schema_version": AGENT_SCHEMA_VERSION,
            "status": "act",
            "reason": "",
            "action": {"skill": skill, "parameters": parameters},
        }
    )


def test_agent_loop_stops_when_progress_reports_success() -> None:
    reports = [
        VerificationReport(
            status=VerificationStatus.INCOMPLETE,
            predicates=(
                PredicateEvaluation(
                    predicate="settled",
                    status=PredicateStatus.VIOLATED,
                ),
            ),
            evidence_complete=False,
        ),
        VerificationReport(
            status=VerificationStatus.INCOMPLETE,
            predicates=(
                PredicateEvaluation(
                    predicate="settled",
                    status=PredicateStatus.VIOLATED,
                ),
            ),
            evidence_complete=False,
        ),
        VerificationReport(
            status=VerificationStatus.SUCCEEDED,
            predicates=(
                PredicateEvaluation(
                    predicate="settled",
                    status=PredicateStatus.SATISFIED,
                ),
            ),
            evidence_complete=True,
        ),
    ]

    def progress():
        return reports.pop(0)

    catalog = [
        {
            "name": "grasp_object",
            "parameters": {"object_name": {"type": "string", "required": True}},
        }
    ]
    orchestrator = _Orchestrator()
    provider = _Provider([_act("grasp_object", object_name="item")])
    episode = AgentLoop(
        provider,
        orchestrator,
        catalog=catalog,
        scene_object_names=("item",),
        max_actions=4,
        progress_fn=progress,
    ).run(task_description="Make the item stable.")

    assert episode.status == "succeeded"
    assert episode.success_action == 1
    assert orchestrator.calls[0][0] == "grasp_object"
    plan = episode.to_plan_dict()
    assert plan["stages"][0]["parameters"]["skill"] == "grasp_object"


def test_agent_loop_rejects_micro_skill_without_executing() -> None:
    catalog = [
        {
            "name": "grasp_object",
            "parameters": {"object_name": {"type": "string", "required": True}},
        }
    ]
    orchestrator = _Orchestrator()
    provider = _Provider([_act("arm_move_to", target_pos=[0.1, 0.0, 1.0])])
    episode = AgentLoop(
        provider,
        orchestrator,
        catalog=catalog,
        scene_object_names=("item",),
        max_actions=1,
        max_consecutive_failures=1,
    ).run(task_description="Pick the item.")
    assert episode.status == "failed"
    assert "invalid agent action" in episode.reason
    assert orchestrator.calls == []
    assert episode.steps[0].skill == "invalid_action"


def test_invalid_action_is_fed_back_and_loop_continues() -> None:
    catalog = [
        {
            "name": "grasp_object",
            "parameters": {"object_name": {"type": "string", "required": True}},
        }
    ]
    incomplete = VerificationReport(
        status=VerificationStatus.INCOMPLETE, predicates=(), evidence_complete=False
    )
    reports = [
        incomplete,
        incomplete,
        incomplete,
        VerificationReport(
            status=VerificationStatus.SUCCEEDED,
            predicates=(),
            evidence_complete=True,
        ),
    ]

    def progress():
        return reports.pop(0)

    orchestrator = _Orchestrator()
    provider = _Provider(
        [
            _act("arm_move_to", target_pos=[0.1, 0.0, 1.0]),
            _act("grasp_object", object_name="item"),
        ]
    )
    episode = AgentLoop(
        provider,
        orchestrator,
        catalog=catalog,
        scene_object_names=("item",),
        max_actions=4,
        progress_fn=progress,
    ).run(task_description="Pick the item.")
    assert episode.status == "succeeded"
    assert orchestrator.calls == [("grasp_object", {"object_name": "item"}, "step_02_grasp_object")]
    assert episode.steps[0].skill == "invalid_action"
    assert episode.steps[1].skill == "grasp_object"


def test_agent_trace_preserves_action_diagnostics_for_replanning() -> None:
    catalog = [
        {
            "name": "grasp_object",
            "parameters": {"object_name": {"type": "string", "required": True}},
        }
    ]
    episode = AgentLoop(
        _Provider([_act("grasp_object", object_name="item")]),
        _Orchestrator(),
        catalog=catalog,
        scene_object_names=("item",),
        max_actions=1,
    ).run(task_description="Pick the item.")

    diagnostics = episode.to_json()["steps"][0]["diagnostics"]
    assert diagnostics["details"]["failure_code"] is None
    assert diagnostics["metrics"] == {}


def test_agent_loop_freezes_simulation_clock_during_llm_planning() -> None:
    class _FrozenAdapter(_Adapter):
        def __init__(self):
            self.steps = 0
            self.frozen_during_complete = False
            self._frozen = False

        def freeze_simulation_clock(self):
            from contextlib import contextmanager

            @contextmanager
            def _guard():
                self._frozen = True
                try:
                    yield
                finally:
                    self._frozen = False

            return _guard()

        def step(self):
            if self._frozen:
                raise RuntimeError("physics stepped while the planning clock was frozen")
            self.steps += 1

    class _FrozenProvider(_Provider):
        def complete(self, *, system: str, user: str) -> ProviderResponse:
            assert adapter._frozen
            adapter.frozen_during_complete = True
            return super().complete(system=system, user=user)

    adapter = _FrozenAdapter()
    orchestrator = _Orchestrator()
    orchestrator.adapter = adapter
    catalog = [
        {
            "name": "grasp_object",
            "parameters": {"object_name": {"type": "string", "required": True}},
        }
    ]
    episode = AgentLoop(
        _FrozenProvider([_act("grasp_object", object_name="item")]),
        orchestrator,
        catalog=catalog,
        scene_object_names=("item",),
        max_actions=1,
    ).run(task_description="Pick the item.")

    assert adapter.frozen_during_complete
    assert adapter.steps == 0
    assert episode.steps[0].skill == "grasp_object"


def test_agent_loop_emits_non_authoritative_action_checkpoints() -> None:
    catalog = [
        {
            "name": "grasp_object",
            "parameters": {"object_name": {"type": "string", "required": True}},
        }
    ]
    events = []
    episode = AgentLoop(
        _Provider([_act("grasp_object", object_name="item")]),
        _Orchestrator(),
        catalog=catalog,
        scene_object_names=("item",),
        max_actions=1,
        checkpoint_fn=events.append,
    ).run(task_description="Pick the item.")

    assert episode.status == "failed"
    assert [event["phase"] for event in events] == [
        "episode_started",
        "planning",
        "action_started",
        "action_completed",
        "episode_finished",
    ]
    assert events[2]["skill"] == "grasp_object"
    assert events[3]["success"] is True
