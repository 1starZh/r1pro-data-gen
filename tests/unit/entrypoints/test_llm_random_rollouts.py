"""Unit tests for randomized external-LLM rollout orchestration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import yaml

from tests.support import PROJECT_ROOT
from r1pro_data_gen.tasks import load_task_spec

_SCRIPT = PROJECT_ROOT / "scripts" / "benchmarks" / "run_llm_random_rollouts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_llm_random_rollouts", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_llm_random_rollouts"] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _write_scene(path: Path) -> None:
    path.write_text(
        """name: generic_scene
world:
  ground_size: [8.0, 8.0]
robot:
  asset: asset/r1pro/r1pro.usda
  init_pose: [0.0, 0.0, 0.0]
objects: []
""",
        encoding="utf-8",
    )


def _write_task(scene: Path, instruction: str = "Move the robot to the center of the room.") -> Path:
    task_path = scene.with_name("task.yaml")
    task_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "task_spec.v2",
                "id": "test.move_center",
                "family": "navigation",
                "scene_human_verified": True,
                "scene": yaml.safe_load(scene.read_text(encoding="utf-8")),
                "instruction": instruction,
                "tags": ["test"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return task_path


def _write_formal_manifest(loop_dir: Path, scene: Path) -> None:
    """Create the product artifact contract expected by the batch gate."""
    loop_dir.mkdir(parents=True, exist_ok=True)
    attempt_result = loop_dir / "attempt_01" / "result.json"
    result_path = loop_dir / "result.json"
    if not result_path.is_file() and attempt_result.is_file():
        result_path.write_text(attempt_result.read_text(encoding="utf-8"), encoding="utf-8")
    if not (loop_dir / "loop_result.json").is_file():
        (loop_dir / "loop_result.json").write_text("{}", encoding="utf-8")
    (loop_dir / "goal_spec.json").write_text("{}", encoding="utf-8")
    (loop_dir / "goal_contract.json").write_text("{}", encoding="utf-8")
    for name in ("input.json", "evidence.json", "action_trace.json", "plan.json"):
        (loop_dir / name).write_text("{}", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "scene": {
            "source": "embedded_task_spec",
            "human_verified": True,
            "path": str(scene.resolve()),
        },
        "goal_spec": {
            "path": str((loop_dir / "goal_spec.json").resolve()),
            "sha256": "frozen-hash",
            "contract_sha256": "frozen-contract-hash",
        },
        "artifact_paths": {
            name: str((loop_dir / filename).resolve())
            for name, filename in {
                "input": "input.json",
                "goal_spec": "goal_spec.json",
                "goal_spec_provenance": "goal_spec.json.provenance.json",
                "goal_contract": "goal_contract.json",
                "evidence": "evidence.json",
                "action_trace": "action_trace.json",
                "plan": "plan.json",
                "result": "result.json",
                "loop_result": "loop_result.json",
                "video": "rollout.mp4",
            }.items()
        },
        "acceptance_status": "accepted",
    }
    (loop_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_random_rollout_uses_only_task_spec(tmp_path):
    output_dir = tmp_path / "rollouts"
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    task = _write_task(scene)
    args = module.build_parser().parse_args(
        [
            "--task",
            str(task),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
            "--prepare-only",
        ]
    )

    module.run_rollouts(args)
    command = json.loads(
        (output_dir / "rollout_01" / "command.txt").read_text(encoding="utf-8")
    )
    derived = load_task_spec(output_dir / "rollout_01" / "task.yaml")
    assert derived.scene_human_verified is False
    assert derived.scene == yaml.safe_load(
        (output_dir / "rollout_01" / "scene.yaml").read_text(encoding="utf-8")
    )
    assert "--task-module" not in command
    assert command[command.index("--task") + 1].endswith("task.yaml")
    assert "--scene-path" not in command
    assert "--instruction" not in command


def test_random_rollout_requires_explicit_task_spec(tmp_path):
    with __import__("pytest").raises(SystemExit):
        module.build_parser().parse_args(["--output-dir", str(tmp_path / "rollouts")])


def test_random_rollout_counts_only_goal_spec_verifier_success(tmp_path, monkeypatch):
    output_dir = tmp_path / "rollouts"
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    task = _write_task(scene)

    captured_env = {}

    def fake_run(command, **kwargs):
        captured_env.update(kwargs)
        loop_dir = Path(command[command.index("--output-dir") + 1])
        attempt_dir = loop_dir / "attempt_01"
        attempt_dir.mkdir(parents=True)
        (loop_dir / "goal_spec.json.provenance.json").write_text(
            json.dumps(
                {
                    "goal_spec_hash": "frozen-hash",
                    "goal_contract_hash": "frozen-contract-hash",
                }
            ),
            encoding="utf-8",
        )
        (loop_dir / "rollout.mp4").write_bytes(b"video")
        (loop_dir / "loop_result.json").write_text(
            json.dumps({"status": "succeeded", "attempts": 1, "success_attempt": 1}),
            encoding="utf-8",
        )
        (attempt_dir / "result.json").write_text(
            json.dumps(
                    {
                        "result": "passed",
                        "status": "succeeded",
                        "evaluation_mode": "goal_spec",
                    "goal_spec_hash": "frozen-hash",
                    "goal_contract_hash": "frozen-contract-hash",
                    "video": "rollout.mp4",
                    "video_rgb_valid": 1.0,
                    "video_frame_count": 10.0,
                    "video_duration_s": 1.0,
                    "acceptance": {
                        "status": "accepted",
                        "goal_satisfied": True,
                        "evidence_coverage_complete": True,
                        "artifact_valid": True,
                        "hashes_match": True,
                    },
                    "evaluation": {
                        "status": "succeeded",
                        "evidence_complete": True,
                        "evidence_coverage_complete": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        _write_formal_manifest(loop_dir, scene)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.build_parser().parse_args(
        [
            "--task",
            str(task),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
        ]
    )

    assert module.run_rollouts(args) == 0
    assert str(module.PROJECT_ROOT / "src") in captured_env["env"]["PYTHONPATH"].split(":")
    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["succeeded"] == 1


def test_random_rollout_counts_agent_loop_result_json(tmp_path, monkeypatch):
    output_dir = tmp_path / "rollouts"
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    task = _write_task(scene)

    def fake_run(command, **kwargs):
        del kwargs
        loop_dir = Path(command[command.index("--output-dir") + 1])
        loop_dir.mkdir(parents=True)
        (loop_dir / "goal_spec.json.provenance.json").write_text(
            json.dumps(
                {
                    "goal_spec_hash": "frozen-hash",
                    "goal_contract_hash": "frozen-contract-hash",
                }
            ),
            encoding="utf-8",
        )
        (loop_dir / "rollout.mp4").write_bytes(b"video")
        payload = {
            "result": "passed",
            "status": "succeeded",
            "evaluation_mode": "goal_spec",
            "goal_spec_hash": "frozen-hash",
            "goal_contract_hash": "frozen-contract-hash",
            "video": "rollout.mp4",
            "video_rgb_valid": 1.0,
            "video_frame_count": 10.0,
            "video_duration_s": 1.0,
            "acceptance": {
                "status": "accepted",
                "goal_satisfied": True,
                "evidence_coverage_complete": True,
                "artifact_valid": True,
                "hashes_match": True,
            },
            "evaluation": {
                "status": "succeeded",
                "evidence_complete": True,
                "evidence_coverage_complete": True,
            },
            "attempts": 4,
            "success_attempt": 4,
        }
        (loop_dir / "loop_result.json").write_text(json.dumps(payload), encoding="utf-8")
        (loop_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        _write_formal_manifest(loop_dir, scene)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.build_parser().parse_args(
        [
            "--task",
            str(task),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
        ]
    )

    assert module.run_rollouts(args) == 0
    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["succeeded"] == 1


def test_random_rollout_rejects_loop_status_without_goal_spec_result(tmp_path, monkeypatch):
    output_dir = tmp_path / "rollouts"
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    task = _write_task(scene)

    def fake_run(command, **kwargs):
        loop_dir = Path(command[command.index("--output-dir") + 1])
        loop_dir.mkdir(parents=True)
        (loop_dir / "loop_result.json").write_text(
            json.dumps({"status": "succeeded", "attempts": 1, "success_attempt": 1}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.build_parser().parse_args(
        [
            "--task",
            str(task),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
        ]
    )

    assert module.run_rollouts(args) == 1
    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["succeeded"] == 0


def test_random_rollout_archives_replay_timeout_in_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "rollouts"
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    task = _write_task(scene)

    def fake_timeout(command, **kwargs):
        del kwargs
        loop_dir = Path(command[command.index("--output-dir") + 1])
        loop_dir.mkdir(parents=True)
        raise subprocess.TimeoutExpired(command, timeout=3, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(module.subprocess, "run", fake_timeout)
    args = module.build_parser().parse_args(
        [
            "--task",
            str(task),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
        ]
    )

    assert module.run_rollouts(args) == 1
    report = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert report["rollouts"][0]["status"] == "timeout"
    assert report["rollouts"][0]["timed_out"] is True
    rollout = output_dir / "rollout_01"
    assert (rollout / "runner.stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (rollout / "runner.stderr.log").read_text(encoding="utf-8") == "partial stderr"


# Provider key lookup belongs to the generic loop/provider boundary, not this runner.
def test_random_rollout_defers_key_resolution_to_provider(tmp_path, monkeypatch):
    output_dir = tmp_path / "rollouts"
    scene = tmp_path / "scene.yaml"
    _write_scene(scene)
    task = _write_task(scene)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def fake_run(command, **kwargs):
        loop_dir = Path(command[command.index("--output-dir") + 1])
        loop_dir.mkdir(parents=True)
        (loop_dir / "loop_result.json").write_text(
            json.dumps({"status": "succeeded", "attempts": 1, "success_attempt": 1}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.build_parser().parse_args(
        [
            "--task",
            str(task),
            "--output-dir",
            str(output_dir),
            "--count",
            "1",
        ]
    )

    assert module.run_rollouts(args) == 1
