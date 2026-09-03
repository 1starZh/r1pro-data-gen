"""Run bounded TaskSpec rollouts over randomized robot poses.

Each rollout receives one self-contained TaskSpec and creates a derived
per-rollout TaskSpec with the randomized scene embedded in the task file. The
runner only prepares scene data and invokes the product ``run_task.py``
entrypoint; GoalSpec generation, planning, replay, and verification remain in
that generic product path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ENTRYPOINT = PROJECT_ROOT / "scripts" / "tasks" / "run_task.py"
DEFAULT_URDF = PROJECT_ROOT / "asset/r1pro/r1_pro_with_gripper.urdf"

# The runner is also imported directly by CPU tests.  Make the reusable scene
# randomizer available without requiring the caller to preconfigure PYTHONPATH.
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from r1pro_data_gen.data.randomization import (  # noqa: E402
    SceneRandomizationError,
    default_randomization_spec,
    randomize_scene_data,
)
from r1pro_data_gen.data.scenes import write_scene_yaml
from r1pro_data_gen.tasks import load_task_spec, write_task_spec
from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS


def _random_scene(
    base_data: dict[str, Any],
    rng: random.Random,
    spec: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample a valid scene through the task-independent randomizer."""
    return randomize_scene_data(
        base_data,
        rng,
        default_randomization_spec() if spec is None else spec,
    )


def _load_randomization_spec(path: Path | None) -> dict[str, Any]:
    if path is None:
        return default_randomization_spec()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"randomization spec is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("randomization spec YAML must contain a mapping")
    return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _load_replay_result(loop_dir: Path) -> dict[str, Any] | None:
    """Load the product result.json, falling back to the legacy attempt tree."""
    result = _load_json(loop_dir / "result.json")
    if isinstance(result, dict):
        return result
    loop_result = _load_json(loop_dir / "loop_result.json") or {}
    success_attempt = loop_result.get("success_attempt")
    if not isinstance(success_attempt, int) or success_attempt < 1:
        return None
    return _load_json(loop_dir / f"attempt_{success_attempt:02d}" / "result.json")


def _is_goal_spec_success(loop_dir: Path) -> bool:
    """Return true only when the complete product acceptance gate passes."""
    loop_result = _load_json(loop_dir / "loop_result.json") or {}
    result = _load_replay_result(loop_dir)
    frozen_hash = _load_goal_spec_hash(loop_dir)
    frozen_contract_hash = _load_goal_contract_hash(loop_dir)
    if (
        not isinstance(result, dict)
        or not frozen_hash
        or not frozen_contract_hash
        or not _manifest_is_complete(loop_dir)
    ):
        return False
    evaluation = result.get("evaluation")
    acceptance = result.get("acceptance")
    video_path = result.get("video")
    if isinstance(video_path, str):
        video = Path(video_path)
        if not video.is_absolute():
            video = loop_dir / video
    else:
        video = None
    return bool(
        isinstance(acceptance, dict)
        and acceptance.get("status") == "accepted"
        and acceptance.get("goal_satisfied") is True
        and acceptance.get("evidence_coverage_complete") is True
        and acceptance.get("artifact_valid") is True
        and acceptance.get("hashes_match") is True
        and loop_result.get("status") == "succeeded"
        and result.get("result") == "passed"
        and result.get("status") == "succeeded"
        and result.get("evaluation_mode") == "goal_spec"
        and isinstance(evaluation, dict)
        and evaluation.get("status") == "succeeded"
        and bool(evaluation.get("evidence_coverage_complete", evaluation.get("evidence_complete", False)))
        and bool(result.get("video_rgb_valid", False))
        and _positive_number(result.get("video_frame_count"))
        and _positive_number(result.get("video_duration_s"))
        and video is not None
        and video.is_file()
        and video.stat().st_size > 0
        and result.get("goal_spec_hash") == frozen_hash
        and result.get("goal_contract_hash") == frozen_contract_hash
    )


def _manifest_is_complete(loop_dir: Path) -> bool:
    """Require the formal product manifest and its core artifacts."""
    manifest = _load_json(loop_dir / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        return False
    scene = manifest.get("scene")
    goal_spec = manifest.get("goal_spec")
    artifact_paths = manifest.get("artifact_paths")
    if not isinstance(scene, dict) or not isinstance(goal_spec, dict):
        return False
    if scene.get("source") != "embedded_task_spec" or scene.get("human_verified") is not True:
        return False
    if not isinstance(scene.get("path"), str) or not scene.get("path"):
        return False
    if not isinstance(goal_spec.get("path"), str) or not goal_spec.get("path"):
        return False
    if goal_spec.get("sha256") is None or goal_spec.get("contract_sha256") is None:
        return False
    if not isinstance(artifact_paths, dict):
        return False
    for name in (
        "input",
        "goal_spec",
        "goal_spec_provenance",
        "goal_contract",
        "evidence",
        "action_trace",
        "plan",
        "result",
        "loop_result",
        "video",
    ):
        value = artifact_paths.get(name)
        if not isinstance(value, str) or not value:
            return False
        path = Path(value)
        if not path.is_absolute():
            path = loop_dir / path
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return manifest.get("acceptance_status") == "accepted"


def _load_goal_spec_hash(loop_dir: Path) -> str | None:
    provenance = _load_json(loop_dir / "goal_spec.json.provenance.json") or {}
    value = provenance.get("goal_spec_hash")
    return value if isinstance(value, str) and value else None


def _load_goal_contract_hash(loop_dir: Path) -> str | None:
    provenance = _load_json(loop_dir / "goal_spec.json.provenance.json") or {}
    value = provenance.get("goal_contract_hash")
    return value if isinstance(value, str) and value else None


def _positive_number(value: object) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def run_rollouts(args: argparse.Namespace) -> int:
    task_spec = load_task_spec(args.task)
    if isinstance(args.count, bool) or not isinstance(args.count, int) or args.count < 1:
        raise ValueError("count must be a positive integer")
    if isinstance(args.timeout_s, bool) or not isinstance(args.timeout_s, int) or args.timeout_s < 1:
        raise ValueError("timeout_s must be a positive integer")
    base_data = task_spec.scene
    if not isinstance(base_data, dict):
        raise ValueError(f"TaskSpec scene must contain a mapping: {task_spec.id}")
    if not args.prepare_only:
        task_spec.require_human_verified()
    randomization_spec = _load_randomization_spec(args.randomization_spec)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    summary: list[dict[str, Any]] = []

    for index in range(1, args.count + 1):
        rollout_dir = args.output_dir / f"rollout_{index:02d}"
        rollout_dir.mkdir()
        rollout_seed = args.seed + index - 1
        try:
            scene_data, randomization = _random_scene(base_data, rng, randomization_spec)
        except SceneRandomizationError as exc:
            randomization = {
                "schema_version": "scene_randomization.v1",
                "status": "invalid",
                "diagnostics": list(exc.diagnostics) or [str(exc)],
            }
            (rollout_dir / "randomization.json").write_text(
                json.dumps(randomization, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary.append(
                {
                    "rollout": index,
                    "seed": rollout_seed,
                    "randomization": randomization,
                    "task": task_spec.id,
                    "scene_human_verified": task_spec.scene_human_verified,
                    "instruction": task_spec.instruction,
                    "status": "invalid_randomization",
                    "success_attempt": None,
                    "attempts": 0,
                    "returncode": None,
                    "output_dir": str(rollout_dir.resolve()),
                }
            )
            print(f"[rollout {index}/{args.count}] invalid randomization", flush=True)
            continue
        scene_path = write_scene_yaml(scene_data, rollout_dir / "scene.yaml")
        (rollout_dir / "randomization.json").write_text(
            json.dumps(randomization, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rollout_task_path = rollout_dir / "task.yaml"
        write_task_spec(
            task_spec,
            rollout_task_path,
            scene=scene_data,
            scene_human_verified=False,
        )
        loop_dir = rollout_dir / "loop"
        command = [
            sys.executable,
            str(PRODUCT_ENTRYPOINT),
            "--task", str(rollout_task_path),
            "--urdf", str(args.urdf),
            "--output-dir", str(loop_dir),
            "--seed", str(rollout_seed),
            "--max-attempts", str(args.max_attempts),
            "--max-actions-per-attempt", str(args.max_actions_per_attempt),
            "--max-action-physics-steps", str(args.max_action_physics_steps),
            "--max-action-seconds", str(args.max_action_seconds),
            "--feedback-window", str(args.feedback_window),
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--physical-gpu-id", str(args.physical_gpu_id),
            "--device", args.device,
        ]
        if args.stream_logs:
            command.append("--stream-replay-logs")
        (rollout_dir / "command.txt").write_text(
            json.dumps(command, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.prepare_only:
            summary.append(
                {
                    "rollout": index,
                    "seed": rollout_seed,
                    "randomization": randomization,
                    "task": task_spec.id,
                    "scene_human_verified": False,
                    "instruction": task_spec.instruction,
                    "status": "prepared",
                    "success_attempt": None,
                    "attempts": 0,
                    "returncode": None,
                    "output_dir": str(rollout_dir.resolve()),
                }
            )
            print(f"[rollout {index}/{args.count}] prepared", flush=True)
            continue
        print(f"[rollout {index}/{args.count}] start seed={rollout_seed}", flush=True)
        timed_out = False
        # ``run_task.py`` is intentionally an import-light Isaac entrypoint
        # and does not mutate ``sys.path`` before bootstrapping SimulationApp.
        # Keep the batch runner self-contained by forwarding the repository's
        # source root to the child, regardless of the caller's working
        # directory or shell environment.
        child_env = dict(os.environ)
        source_root = str(SOURCE_ROOT)
        existing_pythonpath = child_env.get("PYTHONPATH", "")
        child_env["PYTHONPATH"] = (
            source_root
            if not existing_pythonpath
            else source_root + os.pathsep + existing_pythonpath
        )
        try:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=child_env,
                capture_output=not args.stream_logs,
                text=True,
                timeout=args.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            # ``subprocess.run`` has already terminated the child at this
            # boundary.  Preserve partial logs and emit a structured timeout
            # record so a batch remains auditable instead of crashing before
            # summary.json is written.
            timed_out = True
            completed = subprocess.CompletedProcess(
                command,
                124,
                _coerce_log_text(exc.stdout),
                _coerce_log_text(exc.stderr),
            )
        (rollout_dir / "runner.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
        (rollout_dir / "runner.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
        loop_result = _load_json(loop_dir / "loop_result.json") or {}
        result = _load_replay_result(loop_dir)
        success = _is_goal_spec_success(loop_dir)
        frozen_hash = _load_goal_spec_hash(loop_dir)
        result_hash = result.get("goal_spec_hash") if isinstance(result, dict) else None
        status = "timeout" if timed_out else ("succeeded" if success else "failed")
        record = {
            "rollout": index,
            "seed": rollout_seed,
            "randomization": randomization,
            "task": task_spec.id,
            "scene_human_verified": False,
            "instruction": task_spec.instruction,
            "status": status,
            "evaluation_mode": result.get("evaluation_mode") if isinstance(result, dict) else None,
            "evaluation_status": (
                result.get("evaluation", {}).get("status")
                if isinstance(result, dict) and isinstance(result.get("evaluation"), dict)
                else None
            ),
            "goal_spec_hash": frozen_hash,
            "result_goal_spec_hash": result_hash,
            "success_attempt": loop_result.get("success_attempt"),
            "attempts": loop_result.get("attempts"),
            "returncode": completed.returncode,
            "timed_out": timed_out,
            "output_dir": str(rollout_dir.resolve()),
        }
        summary.append(record)
        print(
            f"[rollout {index}/{args.count}] status={record['status']} "
            f"success_attempt={record['success_attempt']}",
            flush=True,
        )

    succeeded = sum(record["status"] == "succeeded" for record in summary)
    report = {
        "task": task_spec.id,
        "task_spec_path": str(task_spec.source_path) if task_spec.source_path else None,
        "scene_source": "embedded_task_spec",
        "scene_human_verified": task_spec.scene_human_verified,
        "seed": args.seed,
        "count": args.count,
        "succeeded": succeeded,
        "failed": args.count - succeeded,
        "success_rate": succeeded / args.count,
        "settings": {
            "max_attempts": args.max_attempts,
            "max_actions_per_attempt": args.max_actions_per_attempt,
            "feedback_window": args.feedback_window,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "physical_gpu_id": args.physical_gpu_id,
            "device": args.device,
            "randomization_spec": (
                str(args.randomization_spec.resolve())
                if args.randomization_spec is not None
                else None
            ),
        },
        "rollouts": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("count", "succeeded", "failed", "success_rate")}, ensure_ascii=False))
    if args.prepare_only:
        return 0 if all(record["status"] == "prepared" for record in summary) else 1
    return 0 if succeeded == args.count else 1


def _coerce_log_text(value: object) -> str:
    """Normalize TimeoutExpired partial output for durable UTF-8 logs."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        help="TaskSpec id or YAML path; randomized rollouts inherit its scene and instruction",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--randomization-spec",
        type=Path,
        default=None,
        help="YAML spec for generic robot/object/obstacle/physics perturbations",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Complete physical episodes per rollout; must remain 1",
    )
    parser.add_argument("--max-actions-per-attempt", type=int, default=24)
    parser.add_argument("--max-action-physics-steps", type=int, default=60000)
    parser.add_argument("--max-action-seconds", type=float, default=600.0)
    parser.add_argument("--feedback-window", type=int, default=3)
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        choices=(DEFAULT_VIDEO_FPS,),
        help=f"Video fps (fixed: {DEFAULT_VIDEO_FPS})",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--physical-gpu-id", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=2700,
        help="Hard wall-clock timeout for one rollout (default: 45 minutes)",
    )
    parser.add_argument(
        "--stream-logs",
        action="store_true",
        help="Show per-attempt planning and Isaac Sim replay logs in the terminal",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only generate randomized scenes and manifests; do not call the product entrypoint",
    )
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    raise SystemExit(run_rollouts(parsed_args))
