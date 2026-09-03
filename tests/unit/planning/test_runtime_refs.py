from __future__ import annotations

from types import SimpleNamespace

import pytest

from r1pro_data_gen.domain import Observation
from r1pro_data_gen.planning.context.runtime_refs import RuntimeReferenceError, resolve_parameters


class _Result:
    def __init__(self, details):
        self.details = details


class _Scene:
    def object(self, name):
        if name != "cube":
            raise KeyError(name)
        return SimpleNamespace(pos=(2.0, 3.0, 0.4), quat=(1.0, 0.0, 0.0, 0.0))


def test_resolves_stage_position_with_offset():
    value = resolve_parameters(
        {"target": {"ref": "stage.observe.details.position", "shape": [3], "offset": [0.0, 0.0, 0.1]}},
        stage_results={"observe": _Result({"position": [1.0, 2.0, 0.5]})},
        observation=Observation(0.0, base_pose=(0.0, 0.0, 0.0)),
        scene=_Scene(),
        current_stage="move",
        stage_outputs={"observe": {"position"}},
        stage_dependencies=("observe",),
    )
    assert value["target"] == [1.0, 2.0, 0.6]


def test_rejects_uncompleted_or_undeclared_stage_output():
    kwargs = dict(
        stage_results={"observe": _Result({"position": [1.0, 2.0, 0.5]})},
        observation=Observation(0.0, base_pose=(0.0, 0.0, 0.0)),
        scene=_Scene(),
        current_stage="move",
        stage_outputs={"observe": set()},
        stage_dependencies=("observe",),
    )
    with pytest.raises(RuntimeReferenceError, match="did not declare"):
        resolve_parameters({"x": {"ref": "stage.observe.details.position"}}, **kwargs)
    with pytest.raises(RuntimeReferenceError, match="not a dependency"):
        resolve_parameters(
            {"x": {"ref": "stage.other.details.position"}},
            **{**kwargs, "stage_results": {"other": _Result({"position": [1, 2, 3]})}},
        )


def test_rejects_expression_and_nonfinite_reference_result():
    kwargs = dict(
        stage_results={"observe": _Result({"position": [1.0, 2.0, 0.5]})},
        observation=Observation(0.0, base_pose=(0.0, 0.0, 0.0)),
        scene=_Scene(),
        current_stage="move",
    )
    with pytest.raises(RuntimeReferenceError, match="no detail"):
        resolve_parameters({"x": {"ref": "stage.observe.details.position[0]"}}, **kwargs)
    with pytest.raises(RuntimeReferenceError, match="finite"):
        resolve_parameters({"x": {"ref": "stage.observe.details.position", "offset": [0.0, 0.0, float("nan")]}}, **kwargs)
