"""Run provenance models and JSON serialization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Minimum identity needed to reproduce a recorded run."""

    run_id: str
    task: str
    seed: int
    project_version: str
    planner: str
    controller: str

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.task.strip():
            raise ValueError("run_id and task must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_provenance(provenance: RunProvenance, path: str | Path) -> None:
    """Write deterministic, human-readable provenance JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(provenance.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
