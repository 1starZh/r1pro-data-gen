"""Regression tests for startup/failure video capture cadence."""

from __future__ import annotations

import numpy as np
import pytest

from r1pro_data_gen.simulation.isaac_sim.video import VideoRecorder


class _ArrayView:
    def __init__(self, value: np.ndarray) -> None:
        self._value = value

    def detach(self) -> "_ArrayView":
        return self

    def cpu(self) -> "_ArrayView":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Camera:
    def __init__(self, frame: np.ndarray) -> None:
        self.data = type("CameraData", (), {"output": {"rgb": [_ArrayView(frame)]}})()


class _Adapter:
    def __init__(self, frame: np.ndarray) -> None:
        self.camera = _Camera(frame)


def test_step_hook_captures_startup_frames_before_first_stage() -> None:
    frame = np.full((2, 3, 3), 128, dtype=np.uint8)
    recorder = VideoRecorder(_Adapter(frame), "/tmp/unused-rollout.mp4", fps=30, sim_hz=60)

    for _ in range(8):
        recorder.step_hook()

    # At 60 Hz / 30 FPS, steps 0, 2, 4, and 6 are sampled. This is the evidence that
    # run_plan.py can retain a useful startup window even when stage 1 fails.
    assert len(recorder.frames) == 4
    assert all(np.array_equal(captured, frame) for captured in recorder.frames)


def test_video_recorder_rejects_nonstandard_fps() -> None:
    frame = np.full((2, 3, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="fixed at 30"):
        VideoRecorder(_Adapter(frame), "/tmp/unused-rollout.mp4", fps=15)
