from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tests.support import PROJECT_ROOT

_SCRIPT = PROJECT_ROOT / "scripts" / "benchmarks" / "run_benchmark_suite.py"
_SPEC = importlib.util.spec_from_file_location("run_benchmark_suite", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
sys.modules["run_benchmark_suite"] = module
_SPEC.loader.exec_module(module)


def _write_scene(path: Path) -> None:
    path.write_text(
        """name: suite_scene
world:
  ground_size: [8.0, 8.0]
robot:
  asset: asset/r1pro/r1pro.usda
  init_pose: [0.0, 0.0, 0.0]
objects: []
""",
        encoding="utf-8",
    )


def _write_task(scene: Path) -> Path:
    task = scene.with_name("task.yaml")
    task.write_text(
        yaml.safe_dump(
            {
                "schema_version": "task_spec.v2",
                "id": "test.suite_task",
                "family": "test",
                "scene_human_verified": True,
                "scene": yaml.safe_load(scene.read_text(encoding="utf-8")),
                "instruction": "Move the robot to the center of the room.",
                "tags": ["test"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return task


def _suite(scene: Path) -> dict:
    task = _write_task(scene)
    return {
        "schema_version": "benchmark_suite.v1",
        "name": "test_suite",
        "seed": 11,
        "min_rollouts_per_family": 1,
        "min_success_rate": 1.0,
        "families": [
            {
                "name": "generic_family",
                "cases": [
                    {
                        "id": "case_1",
                        "task": str(task),
                        "count": 1,
                        "randomization": {
                            "schema_version": "scene_randomization.v1",
                            "max_attempts": 8,
                            "robot": {"xy_radius_m": 0.1, "yaw_range_rad": 0.2},
                        },
                    }
                ],
            }
        ],
    }


def test_suite_requires_external_randomization_spec(tmp_path: Path) -> None:
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    suite = _suite(scene)
    del suite["families"][0]["cases"][0]["randomization"]
    path = tmp_path / "suite.yaml"
    import yaml

    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    with pytest.raises(ValueError, match="randomization"):
        module.load_suite(path)


def test_suite_rejects_stage_level_acceptance(tmp_path: Path) -> None:
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    suite = _suite(scene)
    suite["acceptance_unit"] = "stage"
    path = tmp_path / "suite.yaml"
    import yaml

    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    with pytest.raises(ValueError, match="complete_episode"):
        module.load_suite(path)


def test_prepare_only_runs_all_cases_and_writes_audit_report(tmp_path: Path) -> None:
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    suite_path = tmp_path / "suite.yaml"
    import yaml

    suite_path.write_text(yaml.safe_dump(_suite(scene)), encoding="utf-8")
    output = tmp_path / "output"
    args = module.build_parser().parse_args(
        [
            "--suite",
            str(suite_path),
            "--output-dir",
            str(output),
            "--prepare-only",
            "--urdf",
            str(PROJECT_ROOT / "asset/r1pro/r1_pro_with_gripper.urdf"),
        ]
    )

    assert module.run_suite(args) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["status"] == "prepared"
    assert report["acceptance_unit"] == "complete_episode"
    assert report["families"][0]["count"] == 1
    assert report["families"][0]["cases"][0]["invalid_randomizations"] == 0
    assert manifest["code"]["fingerprint_sha256"]
    assert manifest["acceptance_unit"] == "complete_episode"
    assert (output / "generic_family" / "case_1" / "rollouts" / "summary.json").is_file()


def test_physical_suite_rejects_unverified_task_before_launch(tmp_path: Path) -> None:
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    suite = _suite(scene)
    task_path = Path(suite["families"][0]["cases"][0]["task"])
    task_payload = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    task_payload["scene_human_verified"] = False
    task_path.write_text(yaml.safe_dump(task_payload, sort_keys=False), encoding="utf-8")
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    args = module.build_parser().parse_args(
        [
            "--suite",
            str(suite_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--urdf",
            str(PROJECT_ROOT / "asset/r1pro/r1_pro_with_gripper.urdf"),
        ]
    )

    with pytest.raises(ValueError, match="manually verified|unverified"):
        module.run_suite(args)


def test_holdout_families_are_distinct_and_run_only_after_primary_gate(tmp_path: Path, monkeypatch) -> None:
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    suite = _suite(scene)
    suite["holdout_families"] = [
        {
            "name": "unseen_family",
            "cases": [
                {
                    "id": "unseen_case",
                    "task": str(_write_task(scene)),
                    "count": 1,
                    "randomization": {
                        "schema_version": "scene_randomization.v1",
                        "max_attempts": 8,
                        "robot": {"xy_radius_m": 0.1, "yaw_range_rad": 0.2},
                    },
                }
            ],
        }
    ]
    suite_path = tmp_path / "suite.yaml"
    import yaml

    suite_path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    loaded = module.load_suite(suite_path)
    assert [item["name"] for item in loaded["holdout_families"]] == ["unseen_family"]

    calls = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "count": 1,
                    "succeeded": 0,
                    "failed": 1,
                    "success_rate": 0.0,
                    "rollouts": [
                        {
                            "rollout": 1,
                            "status": "failed",
                            "output_dir": str(output_dir / "rollout_01"),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, "", "physical gate failed")

    monkeypatch.setattr(module.subprocess, "run", fake_runner)
    output = tmp_path / "output"
    args = module.build_parser().parse_args(
        [
            "--suite",
            str(suite_path),
            "--output-dir",
            str(output),
            "--urdf",
            str(PROJECT_ROOT / "asset/r1pro/r1_pro_with_gripper.urdf"),
        ]
    )

    assert module.run_suite(args) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["families"][0]["passed"] is False
    assert report["holdout_status"] == "skipped_primary_gate"
    assert report["holdout_families"] == [
        {
            "name": "unseen_family",
            "holdout": True,
            "count": 0,
            "succeeded": 0,
            "failed": 0,
            "success_rate": 0.0,
            "threshold": None,
            "passed": False,
            "status": "primary_families_failed",
            "cases": [],
        }
    ]
    # The primary child ran once; the unseen holdout was not silently mixed
    # into the primary denominator or launched after a failed gate.
    assert len(calls) == 1


def test_family_gate_requires_eight_of_ten_formal_rollouts() -> None:
    case = {
        "count": 10,
        "succeeded": 8,
        "invalid_randomizations": 0,
        "rollout_manifest_paths": [f"manifest-{index}" for index in range(8)],
    }
    report = module.evaluate_family(
        "family",
        [case],
        min_rollouts=10,
        min_success_rate=0.8,
    )
    assert report["passed"] is True

    case["succeeded"] = 7
    case["rollout_manifest_paths"] = [f"manifest-{index}" for index in range(7)]
    report = module.evaluate_family(
        "family",
        [case],
        min_rollouts=10,
        min_success_rate=0.8,
    )
    assert report["passed"] is False
