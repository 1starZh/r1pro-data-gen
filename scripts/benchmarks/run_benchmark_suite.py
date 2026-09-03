"""Run a YAML-defined complete-task generalization benchmark.

The suite is deliberately an orchestration layer.  It does not contain task
policies, object-specific evaluators, or action sequences; every case is still
sent through ``run_llm_random_rollouts.py`` and the product ``run_task.py``
entrypoint.  Each rollout is one complete task episode: all subgoals in the
instruction must be satisfied before its single final acceptance decision is
counted.  Intermediate stage results are retained for diagnosis only.  A
family is only a taxonomy/grouping and passes when its configured minimum
number of complete episodes reaches the configured acceptance rate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "scripts" / "benchmarks" / "run_llm_random_rollouts.py"
DEFAULT_URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"
SUITE_SCHEMA_VERSION = "benchmark_suite.v1"
COMPLETE_EPISODE_ACCEPTANCE_UNIT = "complete_episode"

# Permit direct execution from the repository root without requiring callers
# to remember a separate PYTHONPATH for this import-light orchestration layer.
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from r1pro_data_gen.tasks import TaskSpec, load_task_spec  # noqa: E402
from r1pro_data_gen.data.scenes import write_scene_yaml  # noqa: E402
from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS  # noqa: E402


def load_suite(path: Path) -> dict[str, Any]:
    """Load and validate one external benchmark-suite YAML."""
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"suite YAML is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("suite YAML must contain a mapping")
    allowed = {
        "schema_version",
        "name",
        "seed",
        "min_rollouts_per_family",
        "min_success_rate",
        "acceptance_unit",
        "families",
        "holdout_families",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"suite contains unknown fields: {sorted(unknown)}")
    if payload.get("schema_version", SUITE_SCHEMA_VERSION) != SUITE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported suite schema_version: {payload.get('schema_version')!r}"
        )
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("suite name must be a non-empty string")
    min_rollouts = payload.get("min_rollouts_per_family", 10)
    if isinstance(min_rollouts, bool) or not isinstance(min_rollouts, int) or min_rollouts < 1:
        raise ValueError("min_rollouts_per_family must be a positive integer")
    min_rate = _number(payload.get("min_success_rate", 0.8), "min_success_rate")
    if not 0.0 <= min_rate <= 1.0:
        raise ValueError("min_success_rate must be between 0 and 1")
    acceptance_unit = payload.get(
        "acceptance_unit", COMPLETE_EPISODE_ACCEPTANCE_UNIT
    )
    if acceptance_unit != COMPLETE_EPISODE_ACCEPTANCE_UNIT:
        raise ValueError(
            "benchmark acceptance_unit must be 'complete_episode'; "
            "stage-level acceptance is not supported"
        )
    seed = payload.get("seed", 20260828)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("suite seed must be an integer")
    normalized_families, family_names = _normalize_families(
        payload.get("families"),
        field_name="families",
        existing_names=set(),
        required=True,
    )
    normalized_holdouts, _ = _normalize_families(
        payload.get("holdout_families", []),
        field_name="holdout_families",
        existing_names=family_names,
        required=False,
    )
    for family in (*normalized_families, *normalized_holdouts):
        for case in family["cases"]:
            _load_task_spec_reference(case["task"], suite_dir=path.parent)
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "name": name,
        "seed": seed,
        "min_rollouts_per_family": min_rollouts,
        "min_success_rate": min_rate,
        "acceptance_unit": acceptance_unit,
        "families": normalized_families,
        "holdout_families": normalized_holdouts,
    }


def run_suite(args: argparse.Namespace) -> int:
    suite_path = args.suite.resolve()
    suite = load_suite(suite_path)
    if not args.prepare_only:
        unverified = []
        for family in (*suite["families"], *suite["holdout_families"]):
            for case in family["cases"]:
                task_spec = _load_task_spec_reference(
                    case["task"], suite_dir=suite_path.parent
                )
                if not task_spec.scene_human_verified:
                    unverified.append(task_spec.id)
        if unverified:
            raise ValueError(
                "benchmark physical execution requires manually verified scenes; "
                f"unverified TaskSpecs: {sorted(set(unverified))}"
            )
    if args.max_attempts != 1:
        raise ValueError(
            "benchmark suite requires one complete physical episode per rollout; "
            "use --max-actions-per-attempt for bounded in-episode recovery"
        )
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    urdf = args.urdf.resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf}")

    suite_manifest: dict[str, Any] = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "status": "running",
        "suite_name": suite["name"],
        "suite_path": str(suite_path),
        "suite_sha256": _sha256_file(suite_path),
        "code": {
            "fingerprint_sha256": _code_fingerprint(),
            "source_roots": ["src", "scripts"],
        },
        "seed": suite["seed"],
        "acceptance_unit": suite["acceptance_unit"],
        "provider": "deepseek",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "urdf": str(urdf),
        "min_rollouts_per_family": suite["min_rollouts_per_family"],
        "min_success_rate": suite["min_success_rate"],
        "command": list(sys.argv),
        "families": [],
        "holdout_families": [],
        "holdout_status": "pending_primary_gate",
    }
    _write_json(args.output_dir / "manifest.json", suite_manifest)

    family_reports: list[dict[str, Any]] = []
    holdout_reports: list[dict[str, Any]] = []
    any_preparation_failure = False
    for family_index, family in enumerate(suite["families"]):
        family_report, preparation_failed = _run_family(
            family,
            family_index=family_index,
            suite=suite,
            suite_path=suite_path,
            output_dir=args.output_dir,
            urdf=urdf,
            args=args,
            holdout=False,
        )
        any_preparation_failure = any_preparation_failure or preparation_failed
        family_reports.append(family_report)
        suite_manifest["families"] = family_reports
        _write_json(args.output_dir / "manifest.json", suite_manifest)

    primary_passed = bool(family_reports) and all(item["passed"] for item in family_reports)
    holdouts = suite["holdout_families"]
    if args.prepare_only:
        suite_manifest["holdout_status"] = "prepared"
        for family_index, family in enumerate(holdouts):
            family_report, preparation_failed = _run_family(
                family,
                family_index=family_index,
                suite=suite,
                suite_path=suite_path,
                output_dir=args.output_dir,
                urdf=urdf,
                args=args,
                holdout=True,
            )
            any_preparation_failure = any_preparation_failure or preparation_failed
            holdout_reports.append(family_report)
            suite_manifest["holdout_families"] = holdout_reports
            _write_json(args.output_dir / "manifest.json", suite_manifest)
    elif holdouts and primary_passed:
        suite_manifest["holdout_status"] = "running"
        for family_index, family in enumerate(holdouts):
            family_report, preparation_failed = _run_family(
                family,
                family_index=family_index,
                suite=suite,
                suite_path=suite_path,
                output_dir=args.output_dir,
                urdf=urdf,
                args=args,
                holdout=True,
            )
            any_preparation_failure = any_preparation_failure or preparation_failed
            holdout_reports.append(family_report)
            suite_manifest["holdout_families"] = holdout_reports
            _write_json(args.output_dir / "manifest.json", suite_manifest)
        suite_manifest["holdout_status"] = "passed" if all(
            item["passed"] for item in holdout_reports
        ) else "failed"
    elif holdouts:
        suite_manifest["holdout_status"] = "skipped_primary_gate"
        holdout_reports = [
            _skipped_family_report(family["name"], "primary_families_failed")
            for family in holdouts
        ]
        suite_manifest["holdout_families"] = holdout_reports
        _write_json(args.output_dir / "manifest.json", suite_manifest)

    suite_passed = primary_passed and (
        not holdouts or all(item["passed"] for item in holdout_reports)
    )
    if args.prepare_only:
        status = "prepared_with_errors" if any_preparation_failure else "prepared"
    else:
        status = "passed" if suite_passed else "failed"
    report = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_name": suite["name"],
        "status": status,
        "passed": suite_passed,
        "acceptance_unit": suite["acceptance_unit"],
        "min_rollouts_per_family": suite["min_rollouts_per_family"],
        "min_success_rate": suite["min_success_rate"],
        "families": family_reports,
        "holdout_families": holdout_reports,
        "holdout_status": suite_manifest["holdout_status"],
        "manifest_path": str((args.output_dir / "manifest.json").resolve()),
    }
    suite_manifest["status"] = status
    suite_manifest["report_path"] = str((args.output_dir / "report.json").resolve())
    _write_json(args.output_dir / "manifest.json", suite_manifest)
    _write_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "suite": suite["name"],
                "status": status,
                "families": {
                    item["name"]: {
                        "success_rate": item["success_rate"],
                        "passed": item["passed"],
                    }
                    for item in family_reports
                },
            },
            ensure_ascii=False,
        )
    )
    if args.prepare_only:
        return 1 if any_preparation_failure else 0
    return 0 if suite_passed else 1


def _normalize_families(
    raw_families: Any,
    *,
    field_name: str,
    existing_names: set[str],
    required: bool,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Validate primary or holdout families with one shared schema path."""
    if raw_families is None:
        raw_families = []
    if not isinstance(raw_families, list) or (required and not raw_families):
        qualifier = "non-empty " if required else ""
        raise ValueError(f"suite {field_name} must be a {qualifier}array")
    normalized: list[dict[str, Any]] = []
    family_names = set(existing_names)
    for family_index, raw_family in enumerate(raw_families):
        if not isinstance(raw_family, dict):
            raise TypeError(f"{field_name}[{family_index}] must be a mapping")
        family_allowed = {"name", "cases"}
        family_unknown = set(raw_family) - family_allowed
        if family_unknown:
            raise ValueError(
                f"{field_name}[{family_index}] contains unknown fields: {sorted(family_unknown)}"
            )
        family_name = raw_family.get("name")
        if not isinstance(family_name, str) or not family_name.strip():
            raise ValueError(f"{field_name}[{family_index}].name must be non-empty")
        if family_name in family_names:
            raise ValueError(f"duplicate benchmark family: {family_name!r}")
        family_names.add(family_name)
        cases = raw_family.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"family {family_name!r} cases must be a non-empty array")
        normalized_cases: list[dict[str, Any]] = []
        case_names: set[str] = set()
        for case_index, raw_case in enumerate(cases):
            normalized_case = _validate_case(raw_case, family_name, case_index)
            case_name = normalized_case["id"]
            if case_name in case_names:
                raise ValueError(
                    f"family {family_name!r} has duplicate case id {case_name!r}"
                )
            case_names.add(case_name)
            normalized_cases.append(normalized_case)
        normalized.append({"name": family_name, "cases": normalized_cases})
    return normalized, family_names


def _run_family(
    family: Mapping[str, Any],
    *,
    family_index: int,
    suite: Mapping[str, Any],
    suite_path: Path,
    output_dir: Path,
    urdf: Path,
    args: argparse.Namespace,
    holdout: bool,
) -> tuple[dict[str, Any], bool]:
    """Run one complete family and return its report plus prep-failure flag."""
    case_reports: list[dict[str, Any]] = []
    any_preparation_failure = False
    family_dir = output_dir / _safe_name(str(family["name"]))
    family_dir.mkdir()
    for case_index, case in enumerate(family["cases"]):
        case_dir = family_dir / _safe_name(case["id"])
        case_dir.mkdir()
        task_spec = _load_task_spec_reference(case["task"], suite_dir=suite_path.parent)
        scene_path = write_scene_yaml(task_spec.scene, case_dir / "scene.yaml")
        spec_path = _materialize_randomization_spec(case, case_dir, suite_path.parent)
        count = case.get("count", suite["min_rollouts_per_family"])
        seed = case.get(
            "seed",
            int(suite["seed"])
            + (1_000_000 if holdout else 0)
            + family_index * 100_000
            + case_index * 1_000,
        )
        rollout_dir = case_dir / "rollouts"
        command = [
            sys.executable,
            str(RUNNER),
            "--task",
            str(task_spec.source_path),
            "--randomization-spec",
            str(spec_path),
            "--output-dir",
            str(rollout_dir),
            "--count",
            str(count),
            "--seed",
            str(seed),
            "--urdf",
            str(urdf),
            "--max-attempts",
            str(args.max_attempts),
            "--max-actions-per-attempt",
            str(args.max_actions_per_attempt),
            "--max-action-physics-steps",
            str(args.max_action_physics_steps),
            "--max-action-seconds",
            str(args.max_action_seconds),
            "--feedback-window",
            str(args.feedback_window),
            "--fps",
            str(args.fps),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--physical-gpu-id",
            str(args.physical_gpu_id),
            "--device",
            args.device,
            "--timeout-s",
            str(args.rollout_timeout_s),
        ]
        if args.stream_logs:
            command.append("--stream-logs")
        if args.prepare_only:
            command.append("--prepare-only")
        _write_json(case_dir / "command.json", command)
        print(
            f"[{'holdout' if holdout else 'family'}={family['name']} case={case['id']}] "
            f"{'prepare' if args.prepare_only else 'run'} count={count}",
            flush=True,
        )
        child_env = dict(os.environ)
        source_root = str(PROJECT_ROOT / "src")
        child_env["PYTHONPATH"] = (
            source_root
            if not child_env.get("PYTHONPATH")
            else source_root + os.pathsep + child_env["PYTHONPATH"]
        )
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=child_env,
            capture_output=not args.stream_logs,
            text=True,
        )
        if args.prepare_only and completed.returncode != 0:
            any_preparation_failure = True
        (case_dir / "runner.stdout.log").write_text(
            completed.stdout or "", encoding="utf-8"
        )
        (case_dir / "runner.stderr.log").write_text(
            completed.stderr or "", encoding="utf-8"
        )
        summary = _load_json(rollout_dir / "summary.json") or {}
        rollout_records = summary.get("rollouts", [])
        if not isinstance(rollout_records, list):
            rollout_records = []
        accepted_records = [
            record
            for record in rollout_records
            if isinstance(record, dict)
            and record.get("status") == "succeeded"
            and _formal_rollout_manifest_valid(record)
        ]
        invalid_records = [
            record
            for record in rollout_records
            if isinstance(record, dict)
            and record.get("status") == "invalid_randomization"
        ]
        any_preparation_failure = any_preparation_failure or bool(invalid_records)
        case_reports.append(
            {
                "id": case["id"],
                "task": task_spec.id,
                "scene_source": "embedded_task_spec",
                "scene_human_verified": task_spec.scene_human_verified,
                "scene": str(scene_path),
                "scene_sha256": _sha256_file(scene_path),
                "robot_asset": _scene_robot_asset(scene_path),
                "instruction": task_spec.instruction,
                "randomization_spec": str(spec_path),
                "randomization_spec_sha256": _sha256_file(spec_path),
                "seed": seed,
                "count": count,
                "succeeded": len(accepted_records),
                "failed": max(0, count - len(accepted_records)),
                "success_rate": len(accepted_records) / count,
                "invalid_randomizations": len(invalid_records),
                "returncode": completed.returncode,
                "command": command,
                "rollout_manifest_paths": [
                    str(
                        (Path(record["output_dir"]) / "loop" / "manifest.json").resolve()
                    )
                    for record in accepted_records
                    if isinstance(record.get("output_dir"), str)
                ],
                "rollout_summary_path": str((rollout_dir / "summary.json").resolve()),
            }
        )
    family_report = evaluate_family(
        family["name"],
        case_reports,
        min_rollouts=suite["min_rollouts_per_family"],
        min_success_rate=suite["min_success_rate"],
        prepare_only=args.prepare_only,
    )
    family_report["holdout"] = bool(holdout)
    return family_report, any_preparation_failure


def _skipped_family_report(name: str, reason: str) -> dict[str, Any]:
    """Represent a deliberately unrun holdout without counting it as success."""
    return {
        "name": name,
        "holdout": True,
        "count": 0,
        "succeeded": 0,
        "failed": 0,
        "success_rate": 0.0,
        "threshold": None,
        "passed": False,
        "status": reason,
        "cases": [],
    }


def _validate_case(raw_case: Any, family: str, index: int) -> dict[str, Any]:
    if not isinstance(raw_case, dict):
        raise TypeError(f"family {family!r} case[{index}] must be a mapping")
    allowed = {
        "id",
        "task",
        "count",
        "seed",
        "randomization",
        "randomization_spec",
    }
    unknown = set(raw_case) - allowed
    if unknown:
        raise ValueError(
            f"family {family!r} case[{index}] contains unknown fields: {sorted(unknown)}"
        )
    case_id = raw_case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"family {family!r} case[{index}] id must be non-empty")
    task = raw_case.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"family {family!r} case {case_id!r} task must be non-empty")
    randomization_keys = [key for key in ("randomization", "randomization_spec") if raw_case.get(key) is not None]
    if len(randomization_keys) != 1:
        raise ValueError(
            f"case {case_id!r} needs exactly one randomization or randomization_spec"
        )
    if "randomization_spec" in randomization_keys and not isinstance(raw_case["randomization_spec"], str):
        raise TypeError(f"case {case_id!r} randomization_spec must be a path string")
    if "randomization" in randomization_keys and not isinstance(raw_case["randomization"], dict):
        raise TypeError(f"case {case_id!r} randomization must be a mapping")
    count = raw_case.get("count", 10)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"case {case_id!r} count must be a positive integer")
    seed = raw_case.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"case {case_id!r} seed must be an integer")
    return dict(raw_case)


def _load_task_spec_reference(reference: str, *, suite_dir: Path) -> TaskSpec:
    """Resolve a benchmark task id or path relative to the suite first."""
    raw = Path(reference)
    if not raw.is_absolute():
        for candidate in (suite_dir / raw, PROJECT_ROOT / raw, Path.cwd() / raw):
            if candidate.is_file():
                return load_task_spec(candidate)
    return load_task_spec(reference)


def _materialize_randomization_spec(
    case: dict[str, Any],
    case_dir: Path,
    suite_dir: Path,
) -> Path:
    if "randomization" in case:
        spec = case["randomization"]
        if not isinstance(spec, dict):
            raise TypeError(f"case {case['id']!r} randomization must be a mapping")
        path = case_dir / "randomization.yaml"
        path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path
    source = _resolve_input_path(case["randomization_spec"], suite_dir)
    target = case_dir / "randomization.yaml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _resolve_input_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [base_dir / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"input path does not exist: {value}")


def _formal_rollout_manifest_valid(record: Mapping[str, Any]) -> bool:
    if not isinstance(record.get("output_dir"), str):
        return False
    loop_dir = Path(record["output_dir"]) / "loop"
    manifest = _load_json(loop_dir / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        return False
    if manifest.get("acceptance_status") != "accepted":
        return False
    artifacts = manifest.get("artifact_paths")
    if not isinstance(artifacts, dict):
        return False
    required = (
        "goal_spec",
        "goal_spec_provenance",
        "goal_contract",
        "evidence",
        "action_trace",
        "plan",
        "result",
        "loop_result",
        "video",
    )
    return all(_existing_file(artifacts.get(name)) for name in required)


def _case_artifacts_are_formal(case: Mapping[str, Any]) -> bool:
    return len(case.get("rollout_manifest_paths", ())) == int(case.get("succeeded", 0))


def evaluate_family(
    name: str,
    case_reports: list[dict[str, Any]],
    *,
    min_rollouts: int,
    min_success_rate: float,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Apply the task-family gate over complete episode outcomes."""
    total = sum(int(case["count"]) for case in case_reports)
    succeeded = sum(int(case["succeeded"]) for case in case_reports)
    success_rate = succeeded / total if total else 0.0
    passed = (
        not prepare_only
        and total >= min_rollouts
        and success_rate >= min_success_rate
        and all(case["invalid_randomizations"] == 0 for case in case_reports)
        and all(
            _case_artifacts_are_formal(case)
            for case in case_reports
            if case["succeeded"] > 0
        )
    )
    return {
        "name": name,
        "count": total,
        "succeeded": succeeded,
        "failed": total - succeeded,
        "success_rate": success_rate,
        "threshold": min_success_rate,
        "passed": passed,
        "cases": case_reports,
    }


def _scene_robot_asset(path: Path) -> str | None:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    robot = data.get("robot") if isinstance(data, dict) else None
    asset = robot.get("asset") if isinstance(robot, dict) else None
    return asset if isinstance(asset, str) else None


def _existing_file(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return path.is_file() and path.stat().st_size > 0


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _code_fingerprint() -> str:
    digest = hashlib.sha256()
    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix not in {".py", ".json", ".yaml", ".yml"}:
                continue
            digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= result <= 1.0 and name == "min_success_rate":
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _safe_name(value: str) -> str:
    result = "".join(char if char.isalnum() or char in "-_" else "_" for char in value.strip())
    return result or "unnamed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
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
        "--rollout-timeout-s",
        type=int,
        default=2700,
        help="Hard wall-clock timeout for one rollout (default: 45 minutes)",
    )
    parser.add_argument("--stream-logs", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_suite(build_parser().parse_args()))
