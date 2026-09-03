"""Keep the handoff-facing scripts directory grouped by project responsibility."""

from __future__ import annotations

from pathlib import Path

from tests.support import PROJECT_ROOT


SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
EXPECTED_FILES = {
    "README.md",
    "tasks/run_task.py",
    "tasks/run_plan.py",
    "planning/generate_llm_plan.py",
    "planning/run_llm_loop.py",
    "benchmarks/run_benchmark_suite.py",
    "benchmarks/run_llm_random_rollouts.py",
    "infrastructure/check_gpu_health.py",
}
REMOVED_FILES = {
    "calibrate_grasp_center.py",
    "diag_nav_grid.py",
    "mplib_convert_meshes.py",
    "run_informative_loop.sh",
    "run_thinking_loop.sh",
    "thinking_preflight.sh",
}


def _relative_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }


def test_scripts_are_grouped_and_stage_wrappers_are_absent() -> None:
    assert _relative_files(SCRIPTS_ROOT) == EXPECTED_FILES
    assert not any((SCRIPTS_ROOT / name).exists() for name in REMOVED_FILES)
