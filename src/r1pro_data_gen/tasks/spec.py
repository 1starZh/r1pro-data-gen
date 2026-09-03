"""Data-driven task specifications and the repository task catalog.

``TaskSpec`` is the only task-definition contract used by product entrypoints.
An individual task is self-contained data: a stable id, an embedded scene
mapping, an explicit manual-verification status, and a natural-language
instruction. Planning, execution, observation, verification, and artifact
writing remain generic and must not be implemented in a task package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import yaml

from r1pro_data_gen.domain import SceneModel


TASK_SPEC_SCHEMA_VERSION = "task_spec.v2"
_TASK_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_TASK_FIELDS = {
    "schema_version",
    "id",
    "family",
    "scene",
    "scene_human_verified",
    "instruction",
    "tags",
}


def project_root() -> Path:
    """Return the repository root for the installed source tree."""
    return Path(__file__).resolve().parents[3]


def task_specs_dir() -> Path:
    """Return the root directory containing public TaskSpec YAML files."""
    return project_root() / "tasks"


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One complete, data-only task definition.

    ``scene`` is the canonical embedded scene mapping. ``source_path`` is
    optional for generated per-rollout specs and is never part of the public
    YAML schema. A scene marked as unverified may be prepared or planned, but
    physical execution entrypoints must call :meth:`require_human_verified`.
    """

    id: str
    family: str
    scene: dict[str, Any]
    scene_human_verified: bool
    instruction: str
    tags: tuple[str, ...] = ()
    schema_version: str = TASK_SPEC_SCHEMA_VERSION
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TASK_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported task schema_version: {self.schema_version!r}"
            )
        for value, label in ((self.id, "id"), (self.family, "family")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task {label} must be a non-empty string")
        if not _TASK_ID_RE.fullmatch(self.id):
            raise ValueError(
                "task id must contain lowercase letters, digits, '.', '_' or '-' "
                "and must start/end with a letter or digit"
            )
        if not _TASK_ID_RE.fullmatch(self.family):
            raise ValueError("task family must be a lowercase identifier")
        if not isinstance(self.scene, Mapping) or not self.scene:
            raise TypeError("task scene must be a non-empty mapping")
        try:
            SceneModel.from_dict(self.scene)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"task scene is invalid: {exc}") from exc
        if not isinstance(self.scene_human_verified, bool):
            raise TypeError("task scene_human_verified must be a bool")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("task instruction must be a non-empty string")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("task tags must contain non-empty strings")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("task tags must be unique")

    def require_human_verified(self) -> None:
        """Reject physical execution until a human has checked the scene."""
        if not self.scene_human_verified:
            raise ValueError(
                f"task {self.id!r} scene is not human-verified; set "
                "scene_human_verified: true only after manual scene inspection"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable public YAML/JSON representation."""
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "family": self.family,
            "scene_human_verified": self.scene_human_verified,
            "scene": deepcopy(self.scene),
            "instruction": self.instruction,
            "tags": list(self.tags),
        }


class TaskCatalog:
    """Resolve public TaskSpecs by id or explicit YAML path."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or task_specs_dir()).resolve()

    def paths(self) -> tuple[Path, ...]:
        if not self.root.is_dir():
            return ()
        return tuple(sorted(path for path in self.root.rglob("*.yaml") if path.is_file()))

    def load_all(self) -> tuple[TaskSpec, ...]:
        specs = tuple(load_task_spec(path, catalog_root=self.root) for path in self.paths())
        ids: dict[str, Path] = {}
        for spec in specs:
            if spec.id in ids:
                raise ValueError(
                    f"duplicate task id {spec.id!r}: {ids[spec.id]} and {spec.source_path}"
                )
            ids[spec.id] = spec.source_path or Path("<generated>")
        return specs

    def resolve(self, identifier: str | Path) -> TaskSpec:
        """Resolve an id or YAML path, with paths relative to this catalog."""
        value = str(identifier)
        if not value.strip():
            raise ValueError("task identifier must not be empty")
        candidates: list[Path] = []
        path = Path(value)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.extend(
                (
                    Path.cwd() / path,
                    project_root() / path,
                    self.root / path,
                )
            )
            # A dotted task id such as ``pickplace.tabletop_complete`` has a
            # suffix from pathlib's point of view, but it is not a filename.
            # Only real YAML paths should bypass the dotted-id expansion.
            if (
                "/" not in value
                and "\\" not in value
                and path.suffix not in {".yaml", ".yml"}
            ):
                dotted = self.root / Path(*value.split("."))
                candidates.append(dotted.with_suffix(".yaml"))
                candidates.append((self.root / value).with_suffix(".yaml"))
        for candidate in _unique_paths(candidates):
            if candidate.is_file():
                return load_task_spec(candidate, catalog_root=self.root)
        known = ", ".join(spec.id for spec in self.load_all())
        raise FileNotFoundError(
            f"task {value!r} was not found; expected a TaskSpec YAML path or one "
            f"of: {known or '<empty catalog>'}"
        )

    def ids(self) -> tuple[str, ...]:
        return tuple(spec.id for spec in self.load_all())


def load_task_spec(
    path_or_id: str | Path,
    *,
    catalog_root: Path | None = None,
) -> TaskSpec:
    """Load and strictly validate one TaskSpec YAML or catalog id."""
    if isinstance(path_or_id, Path) and path_or_id.is_file():
        path = path_or_id
    elif isinstance(path_or_id, str) and Path(path_or_id).is_file():
        path = Path(path_or_id)
    else:
        return TaskCatalog(catalog_root).resolve(path_or_id)
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"task YAML is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"task YAML must contain a mapping: {path}")
    unknown = set(payload) - _TASK_FIELDS
    missing = _TASK_FIELDS - set(payload) - {"tags"}
    if unknown:
        raise ValueError(f"task contains unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"task is missing fields: {sorted(missing)}")
    schema_version = payload["schema_version"]
    task_id = payload["id"]
    family = payload["family"]
    scene = payload["scene"]
    scene_human_verified = payload["scene_human_verified"]
    instruction = payload["instruction"]
    tags = payload.get("tags", [])
    if not isinstance(schema_version, str):
        raise TypeError("task schema_version must be a string")
    if not isinstance(task_id, str):
        raise TypeError("task id must be a string")
    if not isinstance(family, str):
        raise TypeError("task family must be a string")
    if not isinstance(scene, Mapping) or not scene:
        raise TypeError("task scene must be a non-empty mapping")
    if not isinstance(scene_human_verified, bool):
        raise TypeError("task scene_human_verified must be a bool")
    if not isinstance(instruction, str):
        raise TypeError("task instruction must be a string")
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise TypeError("task tags must be an array of strings")
    return TaskSpec(
        id=task_id,
        family=family,
        scene=deepcopy(dict(scene)),
        scene_human_verified=scene_human_verified,
        instruction=instruction,
        tags=tuple(tags),
        schema_version=schema_version,
        source_path=path,
    )


def write_task_spec(
    spec: TaskSpec,
    path: Path,
    *,
    scene: Mapping[str, Any] | None = None,
    scene_human_verified: bool | None = None,
) -> None:
    """Write a canonical embedded-scene TaskSpec.

    ``scene`` and ``scene_human_verified`` are override points for generated
    derived data. Randomized scenes must explicitly pass ``False`` because a
    transformed scene has not inherited the source scene's manual approval.
    """
    path = Path(path)
    scene_value = spec.scene if scene is None else scene
    verified_value = (
        spec.scene_human_verified
        if scene_human_verified is None
        else scene_human_verified
    )
    if not isinstance(scene_value, Mapping) or not scene_value:
        raise TypeError("task scene must be a non-empty mapping")
    try:
        SceneModel.from_dict(scene_value)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"task scene is invalid: {exc}") from exc
    if not isinstance(verified_value, bool):
        raise TypeError("task scene_human_verified must be a bool")
    payload = {
        "schema_version": spec.schema_version,
        "id": spec.id,
        "family": spec.family,
        "scene_human_verified": verified_value,
        "scene": deepcopy(dict(scene_value)),
        "instruction": spec.instruction,
        "tags": list(spec.tags),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            result.append(path)
            seen.add(key)
    return tuple(result)


__all__ = [
    "TASK_SPEC_SCHEMA_VERSION",
    "TaskCatalog",
    "TaskSpec",
    "load_task_spec",
    "project_root",
    "task_specs_dir",
    "write_task_spec",
]
