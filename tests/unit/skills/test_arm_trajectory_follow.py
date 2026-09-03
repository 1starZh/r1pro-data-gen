"""Pure-contract tests for generic arm tracking recovery."""

from __future__ import annotations

import numpy as np


def test_slow_retime_repeats_the_certified_path_without_changing_geometry():
    from r1pro_data_gen.skills.manipulation.arm_motion import _slow_retime_same_path

    path = np.zeros((3, 7), dtype=float)
    path[1, 3] = -0.4
    path[2, 3] = -1.2
    slowed = _slow_retime_same_path(path, factor=4)

    assert slowed.shape == (12, 7)
    assert np.allclose(slowed[::4], path)
    assert np.allclose(slowed[0], path[0])
    assert np.allclose(slowed[-1], path[-1])
    # Every recovery segment is a segment of the original certified path;
    # retiming must not invent a new waypoint or IK branch.
    assert np.unique(slowed, axis=0).shape[0] == path.shape[0]
