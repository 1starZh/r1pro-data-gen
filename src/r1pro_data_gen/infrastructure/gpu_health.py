"""Fail-closed health probe for the physical GPU used by Isaac Sim."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any


def probe_gpu_health(physical_gpu_id: int = 6, *, timeout_s: float = 10.0) -> dict[str, Any]:
    """Return a JSON-safe, fail-closed health report for one physical GPU."""
    report: dict[str, Any] = {
        "schema_version": "gpu_health.v1",
        "physical_gpu_id": int(physical_gpu_id),
        "healthy": False,
        "nvidia_smi": shutil.which("nvidia-smi"),
        "reason": None,
        "gpus": [],
    }
    if isinstance(physical_gpu_id, bool) or not isinstance(physical_gpu_id, int) or physical_gpu_id < 0:
        report["reason"] = "physical_gpu_id must be a non-negative integer"
        return report
    if report["nvidia_smi"] is None:
        report["reason"] = "nvidia-smi is not installed or not on PATH"
        return report
    command = [
        str(report["nvidia_smi"]),
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        report["reason"] = f"nvidia-smi probe failed: {exc}"
        return report
    report["returncode"] = int(completed.returncode)
    report["stdout"] = (completed.stdout or "").strip()
    report["stderr"] = (completed.stderr or "").strip()
    if completed.returncode != 0:
        report["reason"] = "nvidia-smi could not communicate with the NVIDIA driver"
        return report
    for line in (completed.stdout or "").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        report["gpus"].append(
            {
                "index": index,
                "name": fields[1],
                "driver_version": fields[2],
                "memory_total_mib": fields[3],
            }
        )
    selected = [gpu for gpu in report["gpus"] if gpu["index"] == physical_gpu_id]
    if not selected:
        report["reason"] = "requested physical GPU is not present in nvidia-smi inventory"
        return report
    report["healthy"] = True
    report["selected_gpu"] = selected[0]
    report["reason"] = "driver-backed GPU inventory is available"
    return report


__all__ = ["probe_gpu_health"]
