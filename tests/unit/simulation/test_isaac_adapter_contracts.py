from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from r1pro_data_gen.simulation.isaac_sim.adapter import (
    R1ProSimAdapter,
    effort_within_runtime_limit,
)


def _adapter(sensor_map):
    adapter = object.__new__(R1ProSimAdapter)
    adapter.scene_model = SimpleNamespace(
        collision_sensors=(
            SimpleNamespace(name="collision", body="robot_link", filter=("item",)),
        )
    )
    adapter.scene = SimpleNamespace(sensors=sensor_map)
    return adapter


def _sensor(matrix):
    return SimpleNamespace(data=SimpleNamespace(force_matrix_w=matrix))


def test_collision_coverage_is_incomplete_when_declared_sensor_is_missing() -> None:
    adapter = _adapter({})

    assert not adapter.collision_observation_complete


def test_collision_coverage_is_incomplete_when_filter_telemetry_is_unavailable() -> None:
    adapter = _adapter({"collision": _sensor(None)})

    assert not adapter.collision_observation_complete


def test_collision_coverage_requires_all_declared_filter_slots() -> None:
    adapter = _adapter({"collision": _sensor(np.zeros((1, 1, 0, 3)))})

    assert not adapter.collision_observation_complete


def test_collision_coverage_is_complete_when_all_filter_slots_are_observed() -> None:
    adapter = _adapter({"collision": _sensor(np.zeros((1, 1, 1, 3)))})

    assert adapter.collision_observation_complete


def test_support_contact_uses_real_upward_force_and_ignores_lateral_force() -> None:
    adapter = _adapter({
        "support_contact_wheel1": _sensor(
            torch.tensor([[[3.0, 0.0, 0.0], [0.0, 0.0, 4.5]]])
        ),
    })

    assert adapter.support_contact_forces() == {"wheel1": 4.5}


def test_effort_gate_uses_current_reserve_not_a_recovered_peak() -> None:
    assert effort_within_runtime_limit(hard_exceeded=False, reserve_active_s=0.0)
    assert effort_within_runtime_limit(hard_exceeded=False, reserve_active_s=0.10)
    assert not effort_within_runtime_limit(hard_exceeded=False, reserve_active_s=0.50)
    assert not effort_within_runtime_limit(hard_exceeded=True, reserve_active_s=0.0)
