"""Video recording and validation for headless runs.

The recorder samples RGB frames from the scene camera at a fixed cadence and
writes an MP4, then validates it by streaming the decoded frames for
statistics (mean/max RGB). The orchestrator installs the recorder's
``step_hook`` so every simulation step is captured exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS


@dataclass(slots=True)
class VideoRecorder:
    """Sample camera RGB frames and write/validate an MP4."""

    adapter: object
    output_path: Path
    fps: int = DEFAULT_VIDEO_FPS
    sim_hz: int = 60

    frames: list[np.ndarray] = field(default_factory=list, init=False)
    _step_counter: int = 0

    def __post_init__(self) -> None:
        if self.fps != DEFAULT_VIDEO_FPS:
            raise ValueError(
                f"video fps is fixed at {DEFAULT_VIDEO_FPS}; got {self.fps}"
            )

    def capture_frame(self) -> None:
        """Capture one frame outside the physics hook (startup/failure)."""
        rgb = self.adapter.camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[-1] == 3:
            self.frames.append(np.asarray(rgb, dtype=np.uint8).copy())

    def step_hook(self) -> None:
        """Call after every simulation step to sample a frame."""
        interval = max(1, round(self.sim_hz / self.fps))
        if self._step_counter % interval == 0:
            self.capture_frame()
        self._step_counter += 1

    def write_and_validate(self) -> dict[str, float]:
        """Write the MP4 and return frame statistics (streaming decode)."""
        import imageio.v2 as imageio

        if not self.frames:
            self.capture_frame()
        if not self.frames:
            raise RuntimeError("no RGB frames captured")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(self.output_path, self.frames, fps=self.fps, codec="libx264", quality=8)

        frame_count = 0
        mean_acc = 0.0
        max_rgb = 0.0
        with imageio.get_reader(self.output_path) as reader:
            for frame in reader:
                a = np.asarray(frame, dtype=np.float64)
                frame_count += 1
                mean_acc += float(a.mean())
                max_rgb = max(max_rgb, float(a.max()))
        mean_rgb = mean_acc / max(1, frame_count)
        return {
            "video_bytes": float(self.output_path.stat().st_size),
            "video_fps": float(self.fps),
            "video_frame_count": float(frame_count),
            "video_duration_s": float(frame_count) / float(self.fps),
            "video_mean_rgb": mean_rgb,
            "video_max_rgb": max_rgb,
            "video_rgb_valid": float(bool(frame_count > 0 and mean_rgb > 30.0 and max_rgb > 0.0)),
        }
