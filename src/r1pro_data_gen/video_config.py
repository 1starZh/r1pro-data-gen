"""Project-wide video output policy."""

from __future__ import annotations


# All product-generated videos use one fixed cadence so recordings are
# comparable across tasks, runners, and benchmark orchestration layers.
DEFAULT_VIDEO_FPS = 30
