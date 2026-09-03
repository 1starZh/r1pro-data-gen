"""File-backed data and artifact boundaries.

The domain package owns validated in-memory models.  This package owns their
serialization, provenance records, and scene data transformations so file IO
does not leak into the domain layer.
"""

from .plan_io import (
    load_plan,
    plan_from_dict,
    plan_from_json,
    plan_to_dict,
    plan_to_json,
    save_plan,
)
from .provenance import RunProvenance, write_provenance
from .scenes import load_scene_data, load_scene_yaml, write_scene_yaml

__all__ = [
    "RunProvenance",
    "load_plan",
    "load_scene_data",
    "load_scene_yaml",
    "plan_from_dict",
    "plan_from_json",
    "plan_to_dict",
    "plan_to_json",
    "save_plan",
    "write_provenance",
    "write_scene_yaml",
]
