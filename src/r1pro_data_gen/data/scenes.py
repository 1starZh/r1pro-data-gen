"""Scene loading: YAML or embedded mappings -> :class:`SceneModel`.

Scene data is the environment-facts interface: the planner (Claude today, an
LLM later) reads it to understand an environment, the skills read the
resulting :class:`SceneModel` for collision/geometry, and the Isaac Sim adapter
builds the scene from it. Public TaskSpec v2 files embed this data; the
file-backed loader remains for generic low-level fixtures.
File-backed YAML is intentionally a generic fixture interface.  Public task
scenes are embedded in TaskSpec and should enter through
``load_scene_data(task_spec.scene)`` rather than through a repository-wide
scene-name registry.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from r1pro_data_gen.domain import SceneModel


def load_scene_yaml(path: str | Path) -> SceneModel:
    """Parse a scene YAML file into a validated :class:`SceneModel`."""
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"scene YAML must map to a dict: {path}")
    return load_scene_data(data, source=path)


def load_scene_data(
    data: Mapping[str, Any],
    *,
    source: str | Path = "<embedded scene>",
) -> SceneModel:
    """Parse an in-memory scene mapping into a validated :class:`SceneModel`.

    TaskSpec v2 embeds this mapping directly in each task file.  Keeping the
    mapping loader beside the YAML loader gives file-backed fixtures and
    embedded task scenes exactly the same validation path.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"scene data must map to a dict: {source}")
    try:
        return SceneModel.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid scene data from {source}: {exc}") from exc


def write_scene_yaml(data: Mapping[str, Any], path: str | Path) -> Path:
    """Materialize scene data as a reproducible YAML artifact."""
    import yaml

    target = Path(path)
    if not isinstance(data, Mapping):
        raise ValueError("scene data must map to a dict")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target


__all__ = [
    "load_scene_data",
    "load_scene_yaml",
    "write_scene_yaml",
]
