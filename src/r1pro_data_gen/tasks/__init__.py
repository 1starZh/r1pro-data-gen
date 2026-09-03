"""Data-driven TaskSpec v2 specification and catalog interfaces."""

from .spec import (
    TASK_SPEC_SCHEMA_VERSION,
    TaskCatalog,
    TaskSpec,
    load_task_spec,
    project_root,
    task_specs_dir,
    write_task_spec,
)

__all__ = [
    "TASK_SPEC_SCHEMA_VERSION",
    "TaskCatalog",
    "TaskSpec",
    "load_task_spec",
    "project_root",
    "task_specs_dir",
    "write_task_spec",
]
