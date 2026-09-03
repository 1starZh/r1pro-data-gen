"""Runtime infrastructure contracts used by the physical data pipeline."""

from .gpu_health import probe_gpu_health

__all__ = ["probe_gpu_health"]
