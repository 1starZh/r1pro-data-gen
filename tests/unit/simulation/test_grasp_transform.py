"""Pure transform contracts used by simulated grasp attachment."""

from __future__ import annotations

import numpy as np

from r1pro_data_gen.simulation.isaac_sim.adapter import _quat_rotate


def test_quaternion_rotation_preserves_vector_magnitude():
    angle = np.pi / 2.0
    quat = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    vector = np.array([0.02, 0.0, 0.0])

    rotated = _quat_rotate(quat, vector)

    assert np.allclose(rotated, [0.0, 0.02, 0.0], atol=1e-9)
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(vector))


def test_quaternion_rotation_keeps_zero_vector_zero():
    quat = np.array([0.70710678, 0.0, 0.70710678, 0.0])

    assert np.allclose(_quat_rotate(quat, np.zeros(3)), np.zeros(3))
