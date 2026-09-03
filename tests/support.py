"""Small fake adapters shared by pure skill unit tests.

These doubles intentionally model only the adapter contract used by the
tested skill. They are not part of the runtime robot framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from r1pro_data_gen.domain import Observation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCENES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "scenes"


def load_fixture_scene(name: str):
    """Load a test-only scene fixture by filename stem.

    Public task scenes must be loaded from TaskSpec.  This helper keeps
    low-level skill fixtures explicit and prevents tests from depending on a
    runtime repository-wide scene registry.
    """
    from r1pro_data_gen.data.scenes import load_scene_yaml

    return load_scene_yaml(FIXTURE_SCENES_DIR / f"{name}.yaml")


class TensorStub:
    """Tiny ndarray-like stand-in for Isaac tensor access chains."""

    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._values, dtype=float)


class FakeAdapter:
    """Minimal pure-Python stand-in for ``R1ProSimAdapter``."""

    class Robot:
        class Data:
            root_pos_w = [TensorStub([0.0, 0.0, 0.0])]
            root_quat_w = [TensorStub([1.0, 0.0, 0.0, 0.0])]

        data = Data()

    robot = Robot()

    def __init__(self, joint_positions=None, object_positions=None, contacts=(0.0, 0.0)):
        self._joint_positions = joint_positions or {}
        self._object_positions = object_positions or {}
        self._contacts = contacts
        self.steps = 0

    def read_observation(self, timestamp):
        return Observation(timestamp=timestamp, joint_positions=dict(self._joint_positions))

    def object_position(self, name):
        if name not in self._object_positions:
            raise RuntimeError(f"object {name!r} is not in the scene")
        return self._object_positions[name]

    def finger_contact_forces(self):
        return self._contacts

    def step(self, render=True):
        self.steps += 1

    def set_targets(self, position, velocity=None):
        self.targets = dict(position)


class FakeKinematics:
    """Fixed kinematics used by query and IK skill tests."""

    def fk(self, q_arm):
        return ([0.4, 0.0, 1.2], [1.0, 0.0, 0.0, 0.0])
