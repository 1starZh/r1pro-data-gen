"""Template planner: task name -> fixed skill sequence.

Until the LLM planner is connected (``planning.llm`` placeholder), the
template planner provides the deterministic baseline every task can fall back
to. It turns a task name into a :class:`Plan` whose stages are skill calls --
each stage's parameters carry the skill name plus skill arguments.

Task templates are defined per task (in ``tasks/<task_name>``) and injected
here, keeping the planner mechanism generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from r1pro_data_gen.domain import Plan, PlanStage


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One skill call in a task template."""

    name: str
    skill: str
    goal: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


class TemplatePlanner:
    """Build a Plan from a task name using a fixed template."""

    def __init__(self, templates: Mapping[str, tuple[StageSpec, ...]]) -> None:
        if not templates:
            raise ValueError("templates must not be empty")
        self.templates = templates

    def plan(self, task_name: str) -> Plan:
        """Return the fixed skill-sequence plan for ``task_name``."""
        specs = self.templates.get(task_name)
        if specs is None:
            raise KeyError(f"no template for task: {task_name}")
        stages = tuple(
            PlanStage(
                name=spec.name,
                goal=spec.goal,
                depends_on=spec.depends_on,
                parameters={"skill": spec.skill, **spec.parameters},
            )
            for spec in specs
        )
        return Plan(task_name=task_name, stages=stages)
