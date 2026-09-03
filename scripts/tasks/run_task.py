"""Unified TaskSpec entrypoint for task-agnostic data generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from r1pro_data_gen.data.scenes import load_scene_data, write_scene_yaml
from r1pro_data_gen.tasks import load_task_spec
from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "asset").is_dir() else Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        help="TaskSpec id or YAML path; the spec supplies the scene and instruction",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        choices=(DEFAULT_VIDEO_FPS,),
        help=f"Video fps (fixed: {DEFAULT_VIDEO_FPS})",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--physical-gpu-id",
        type=int,
        default=6,
        help=(
            "Physical Vulkan/RTX GPU index. Use this with CUDA_VISIBLE_DEVICES "
            "when CUDA logical numbering differs from Omniverse's physical numbering."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Complete physical episodes per product invocation; must remain 1",
    )
    parser.add_argument(
        "--max-actions-per-attempt",
        type=int,
        default=24,
        help="Maximum semantic actions in each persistent simulation attempt",
    )
    parser.add_argument(
        "--max-action-physics-steps",
        type=int,
        default=60000,
        help="Maximum simulator physics steps allowed inside one semantic action",
    )
    parser.add_argument(
        "--max-action-seconds",
        type=float,
        default=600.0,
        help="Maximum wall-clock seconds allowed inside one semantic action",
    )
    parser.add_argument("--feedback-window", type=int, default=3)
    parser.add_argument(
        "--goal-spec",
        type=Path,
        default=None,
        help="Reuse a frozen GoalSpec JSON instead of planning a new one",
    )
    parser.add_argument("--stream-replay-logs", action="store_true")
    # Keep the pure orchestrator import-safe; Isaac Lab is loaded only while
    # constructing the CLI that is about to launch a physical replay.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _load_frozen_goal_spec_hash(path: Path) -> str | None:
    """Read the hash persisted when the GoalSpec was frozen."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("goal_spec_hash") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


def _load_frozen_goal_contract_hash(path: Path) -> str | None:
    """Read the deterministic contract hash persisted with a GoalSpec."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("goal_contract_hash") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


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
    """Hash the generic source/entrypoint files used by a product run."""
    digest = hashlib.sha256()
    for root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix not in {".py", ".json", ".yaml", ".yml"}:
                continue
            digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
            digest.update(b"\0")
            try:
                digest.update(path.read_bytes())
            except OSError:
                continue
            digest.update(b"\0")
    return digest.hexdigest()


def _build_manifest(
    *,
    args: argparse.Namespace,
    task_spec,
    scene,
    scene_path: Path,
    goal_spec_path: Path,
    output_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    video_value = payload.get("video")
    video_path = None
    if isinstance(video_value, str) and video_value:
        video_path = str(
            (Path(video_value) if Path(video_value).is_absolute() else output_dir / video_value).resolve()
        )
    provenance_path = goal_spec_path.with_name(goal_spec_path.name + ".provenance.json")
    artifact_paths = {
        "input": str((output_dir / "input.json").resolve()),
        "goal_spec": str(goal_spec_path.resolve()),
        "goal_spec_provenance": str(provenance_path.resolve()),
        "goal_contract": str((output_dir / "goal_contract.json").resolve()),
        "plan_skeleton": str((output_dir / "plan_skeleton.json").resolve()),
        "evidence": str((output_dir / "evidence.json").resolve()),
        "action_trace": str((output_dir / "action_trace.json").resolve()),
        "plan": str((output_dir / "plan.json").resolve()),
        "result": str((output_dir / "result.json").resolve()),
        "entrypoint_result": str((output_dir / "entrypoint_result.json").resolve()),
        "loop_result": str((output_dir / "loop_result.json").resolve()),
        "video": video_path,
    }
    acceptance = payload.get("acceptance")
    return {
        "schema_version": 2,
        "entrypoint": "scripts/tasks/run_task.py",
        "task": {
            "id": task_spec.id,
            "family": task_spec.family,
            "path": str(task_spec.source_path.resolve()) if task_spec.source_path else None,
            "sha256": _sha256_file(task_spec.source_path.resolve()) if task_spec.source_path else None,
        },
        "code": {
            "fingerprint_sha256": _code_fingerprint(),
            "source_roots": ["src", "scripts"],
        },
        "command": list(sys.argv),
        "scene": {
            "name": scene.name,
            "source": "embedded_task_spec",
            "human_verified": task_spec.scene_human_verified,
            "path": str(scene_path.resolve()),
            "sha256": _sha256_file(scene_path.resolve()),
        },
        "goal_spec": {
            "path": str(goal_spec_path.resolve()),
            "sha256": payload.get("goal_spec_hash"),
            "contract_path": str((output_dir / "goal_contract.json").resolve()),
            "contract_sha256": payload.get("goal_contract_hash"),
        },
        "robot_asset": scene.robot.asset,
        "fps": payload.get("video_fps", args.fps),
        "seed": payload.get("seed", getattr(args, "seed", None)),
        "provider": payload.get("provider"),
        "model": payload.get("model"),
        "evaluation_mode": payload.get("evaluation_mode"),
        "artifact_paths": artifact_paths,
        "acceptance_status": (
            acceptance.get("status") if isinstance(acceptance, dict) else None
        ),
        "attempts": payload.get("attempts"),
        "success_attempt": payload.get("success_attempt"),
    }


def _exception_details(exc: BaseException) -> tuple[str, object, str]:
    """Normalize Python and Kit boundary exceptions for JSON artifacts."""
    exception_type = type(exc).__name__
    exception_code = getattr(exc, "code", None)
    reason = str(exc)
    if not reason:
        reason = f"{exception_type} code={exception_code!r}"
    return exception_type, exception_code, reason


def _write_failure_artifacts(
    *,
    args: argparse.Namespace,
    task_spec,
    scene,
    scene_path: Path,
    goal_spec_path: Path,
    output_dir: Path,
    gpu_health: dict[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    """Persist a truthful product failure before Isaac shutdown can run.

    Some Kit/PhysX failure paths terminate from ``SimulationApp.close`` while
    unwinding the episode.  Therefore the physical host must write its
    structured failure *before* entering the shutdown ``finally`` block; the
    outer ``main`` handler remains a second line of defense for failures that
    occur outside the app-owned region.
    """
    from r1pro_data_gen.evaluation import finalize_result_payload

    exception_type, exception_code, reason = _exception_details(exc)
    frozen_hash = _load_frozen_goal_spec_hash(
        goal_spec_path.with_name(goal_spec_path.name + ".provenance.json")
    )
    frozen_contract_hash = _load_frozen_goal_contract_hash(
        goal_spec_path.with_name(goal_spec_path.name + ".provenance.json")
    )
    failure_result = {
        "result": "failed",
        "status": "failed",
        "evaluation_mode": "goal_spec",
        "goal_spec_hash": frozen_hash,
        "goal_contract_hash": frozen_contract_hash,
        "evaluation": {
            "status": "failed",
            "failure_reason": reason,
            "predicates": [],
            "evidence_complete": False,
            "evidence_coverage_complete": False,
            "stage_success_complete": False,
            "collision_observation_complete": False,
        },
        "failure": {
            "category": "runtime_boundary",
            "reason": reason,
            "exception_type": exception_type,
            "exception_code": exception_code,
        },
        "gpu_health": gpu_health,
    }
    failure_result = finalize_result_payload(
        failure_result,
        expected_goal_spec_hash=frozen_hash,
        expected_contract_hash=frozen_contract_hash,
        artifact_valid=False,
    )
    write_json(output_dir / "result.json", failure_result)
    write_json(output_dir / "entrypoint_result.json", failure_result)
    write_json(
        output_dir / "loop_result.json",
        {
            **failure_result,
            "status": "failed",
            "attempts": 0,
            "success_attempt": None,
            "reason": reason,
            "last_failure": None,
        },
    )
    write_json(
        output_dir / "manifest.json",
        _build_manifest(
            args=args,
            task_spec=task_spec,
            scene=scene,
            scene_path=scene_path,
            goal_spec_path=goal_spec_path,
            output_dir=output_dir,
            payload=failure_result,
        ),
    )
    return failure_result


def _reuse_frozen_goal_spec(*, scene, source: Path, output_path: Path) -> None:
    """Copy a previously frozen GoalSpec and rewrite contract/provenance hashes."""
    from r1pro_data_gen.domain import goal_spec_sha256, parse_goal_spec
    from r1pro_data_gen.planning.goals.compiler import GoalCompiler

    payload = json.loads(source.read_text(encoding="utf-8"))
    spec = parse_goal_spec(payload, scene)
    write_json(output_path, payload)
    compiled = GoalCompiler().compile(spec, scene)
    write_json(output_path.with_name("goal_contract.json"), compiled.to_dict())
    write_json(
        output_path.with_name(output_path.name + ".provenance.json"),
        {
            "goal_spec_hash": goal_spec_sha256(spec),
            "goal_contract_hash": compiled.contract_hash,
            "provider": "reused",
            "model": None,
            "source": str(source),
        },
    )


def freeze_goal_spec(*, scene, instruction: str, output_path: Path, provider=None):
    """Generate, validate, hash, and persist one immutable GoalSpec."""
    from r1pro_data_gen.domain import goal_spec_sha256, goal_spec_to_dict
    from r1pro_data_gen.planning.goals.compiler import GoalCompiler
    from r1pro_data_gen.planning.goals.planner import GoalPlanner, GoalPlanningRequest
    from r1pro_data_gen.planning.llm.providers import DeepSeekClient
    from r1pro_data_gen.planning.context.facts import scene_to_facts

    planner = GoalPlanner(provider or DeepSeekClient.from_env())
    result = planner.plan(
        GoalPlanningRequest(
            task_description=instruction,
            scene_facts=scene_to_facts(scene),
            scene=scene,
        )
    )
    if result.status != "planned" or result.goal_spec is None:
        raise RuntimeError(f"GoalSpec planning failed: {result.reason}")
    payload = goal_spec_to_dict(result.goal_spec)
    write_json(output_path, payload)
    compiled = GoalCompiler().compile(result.goal_spec, scene)
    write_json(output_path.with_name("goal_contract.json"), compiled.to_dict())
    write_json(
        output_path.with_name(output_path.name + ".provenance.json"),
        {
            "goal_spec_hash": goal_spec_sha256(result.goal_spec),
            "goal_contract_hash": compiled.contract_hash,
            "provider": result.provider,
            "model": result.model,
            "usage": result.usage,
        },
    )
    return result.goal_spec


def execute_product_episode(
    *,
    args: argparse.Namespace,
    task_spec,
    scene,
    scene_path: Path,
    instruction: str,
    goal_spec_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Host one Isaac episode and run the closed-loop agent inside it."""
    import random

    import numpy as np
    from isaaclab.app import AppLauncher

    from r1pro_data_gen.agent.host import run_hosted_agent
    from r1pro_data_gen.domain import parse_goal_spec
    from r1pro_data_gen.evaluation import finalize_result_payload
    from r1pro_data_gen.simulation.isaac_sim import AdapterCfg, R1ProSimAdapter
    from r1pro_data_gen.simulation.isaac_sim.video import VideoRecorder

    args.headless = True
    args.livestream = getattr(args, "livestream", 0) or 0
    args.enable_cameras = True
    args.renderer = "RayTracedLighting"
    args.multi_gpu = False
    if args.physical_gpu_id is not None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_id)
    kit_args = [
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
        "--/validate/p2p/enabled=false",
        "--/rtx/post/tonemap/op=4",
        "--/rtx/post/tonemap/enabled=true",
    ]
    if args.physical_gpu_id is not None:
        kit_args.append(f"--/renderer/activeGpu={args.physical_gpu_id}")
    args.kit_args = " ".join(kit_args)

    def lifecycle(phase: str, **details: Any) -> None:
        """Persist setup progress before the Isaac app owns the process.

        Native Kit/PhysX shutdowns can surface as ``SystemExit`` (or another
        ``BaseException``) before the hosted agent has a chance to write its
        normal loop artifacts.  A small last-phase checkpoint makes such a
        run diagnosable without changing the physical state or acceptance
        rules.
        """
        write_json(
            output_dir / "lifecycle_checkpoint.json",
            {
                "schema_version": 1,
                "phase": phase,
                "details": {key: str(value) for key, value in details.items()},
            },
        )

    lifecycle("app_starting", physical_gpu_id=args.physical_gpu_id, device=args.device)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    lifecycle("app_started")
    random.seed(args.seed)
    np.random.seed(args.seed)
    adapter = None
    try:
        goal_spec = parse_goal_spec(
            json.loads(goal_spec_path.read_text(encoding="utf-8")),
            scene,
        )
        adapter = R1ProSimAdapter(
            AdapterCfg(
                scene=scene,
                device=args.device,
                width=args.width,
                height=args.height,
                fps=args.fps,
                wheel_control="velocity",
            )
        )
        lifecycle("adapter_constructed")
        adapter.add_distant_light()
        lifecycle("light_added")
        adapter.reset()
        lifecycle("adapter_reset")
        adapter.set_camera_view()
        lifecycle("camera_configured")
        adapter.lock_joint_mask(
            mask_mode="lock",
            joint_groups=("wheel", "torso"),
            lock_root=False,
            gain_overrides={"wheel": (500.0, 100.0)},
        )
        lifecycle("joint_mask_locked")
        recorder = VideoRecorder(adapter, output_dir / "rollout.mp4", fps=args.fps)
        lifecycle("recorder_created")
        for _ in range(60):
            adapter.step()
            recorder.step_hook()
        lifecycle("warmup_finished")
        setup_violation = adapter.physical_safety_violation()
        if setup_violation is not None:
            lifecycle("setup_safety_failed", violation=setup_violation)
            raise RuntimeError(
                f"physical setup safety gate failed before task episode: {setup_violation}"
            )
        # Startup settling may resolve the authored wheel/ground contact. It
        # is checked above, then becomes the physical baseline for the actual
        # manipulation episode without writing any pose to the simulator.
        adapter.rebaseline_physical_metrics()
        lifecycle("physical_baseline_ready")
        adapter.unlock_joint_mask()
        lifecycle("joint_mask_unlocked")
        payload = run_hosted_agent(
            adapter=adapter,
            scene=scene,
            goal_spec=goal_spec,
            output_dir=output_dir,
            instruction=instruction,
            urdf=args.urdf,
            max_attempts=args.max_attempts,
            max_actions_per_attempt=args.max_actions_per_attempt,
            max_action_physics_steps=args.max_action_physics_steps,
            max_action_seconds=args.max_action_seconds,
            feedback_window=args.feedback_window,
            write_json=write_json,
            evidence_hz=10.0,
            seed=args.seed,
            device=args.device,
            physical_gpu_id=args.physical_gpu_id,
            recorder=recorder,
        )
        lifecycle("hosted_agent_finished", status=payload.get("status"))
        video_stats = recorder.write_and_validate()
        payload["video"] = str(recorder.output_path)
        payload.update(video_stats)
        provenance_path = goal_spec_path.with_name(goal_spec_path.name + ".provenance.json")
        payload = finalize_result_payload(
            payload,
            expected_goal_spec_hash=_load_frozen_goal_spec_hash(provenance_path),
            expected_contract_hash=_load_frozen_goal_contract_hash(provenance_path),
        )
        write_json(output_dir / "result.json", payload)
        write_json(output_dir / "loop_result.json", payload)
        # Persist the complete product boundary before SimulationApp.close().
        # Isaac/Omniverse shutdown can terminate the interpreter after the
        # simulation payload is written; the batch gate must still receive a
        # truthful manifest for an otherwise valid physical episode.
        write_json(output_dir / "entrypoint_result.json", payload)
        write_json(
            output_dir / "manifest.json",
            _build_manifest(
                args=args,
                task_spec=task_spec,
                scene=scene,
                scene_path=scene_path,
                goal_spec_path=goal_spec_path,
                output_dir=output_dir,
                payload=payload,
            ),
        )
        lifecycle("product_artifacts_written", status=payload.get("status"))
        return payload
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
            raise
        lifecycle(
            "product_exception",
            exception_type=type(exc).__name__,
            exception=str(exc),
        )
        _write_failure_artifacts(
            args=args,
            task_spec=task_spec,
            scene=scene,
            scene_path=scene_path,
            goal_spec_path=goal_spec_path,
            output_dir=output_dir,
            gpu_health=_load_json(output_dir / "gpu_health.json") or {},
            exc=exc,
        )
        raise
    finally:
        try:
            if adapter is not None:
                adapter.cleanup()
            from isaaclab.sim import SimulationContext

            SimulationContext.clear_instance()
        except Exception:
            pass
        simulation_app.close()


def _probe_gpu_health(physical_gpu_id: int) -> dict[str, Any]:
    """Run the import-light GPU probe without bootstrapping Isaac Sim."""
    from r1pro_data_gen.infrastructure.gpu_health import probe_gpu_health

    return probe_gpu_health(physical_gpu_id)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.task:
        parser.error("--task is required")
    if args.physical_gpu_id not in {1, 6, 7}:
        print(
            "invalid configuration: this project uses physical GPU 6, "
            "GPU 7 when 6 is occupied, or GPU 1 when both are occupied",
            file=sys.stderr,
        )
        return 2
    if not args.urdf.is_file():
        parser.error(f"URDF does not exist: {args.urdf}")

    try:
        task_spec = load_task_spec(args.task)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    try:
        task_spec.require_human_verified()
    except ValueError as exc:
        print(f"scene verification required: {exc}", file=sys.stderr)
        return _USAGE_EXIT

    scene = load_scene_data(
        task_spec.scene,
        source=task_spec.source_path or task_spec.id,
    )
    instruction = task_spec.instruction
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "outputs" / "tasks" / f"{task_spec.id.replace('.', '_')}_generic_loop"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = write_scene_yaml(task_spec.scene, output_dir / "scene.yaml")
    write_json(output_dir / "input.json", {
        "task": task_spec.to_dict(),
        "task_spec_path": str(task_spec.source_path.resolve()) if task_spec.source_path else None,
        "scene_path": str(scene_path),
        "scene_human_verified": task_spec.scene_human_verified,
        "instruction": instruction,
        "scene_name": scene.name,
    })
    gpu_health = _probe_gpu_health(args.physical_gpu_id)
    write_json(output_dir / "gpu_health.json", gpu_health)
    goal_spec_path = output_dir / "goal_spec.json"
    try:
        if not gpu_health.get("healthy", False):
            raise RuntimeError(
                "GPU health gate failed before physical execution: "
                f"{gpu_health.get('reason', 'unknown GPU health failure')}"
            )
        if args.goal_spec is not None:
            if not args.goal_spec.is_file():
                parser.error(f"goal spec does not exist: {args.goal_spec}")
            _reuse_frozen_goal_spec(scene=scene, source=args.goal_spec, output_path=goal_spec_path)
        else:
            freeze_goal_spec(
                scene=scene,
                instruction=instruction,
                output_path=goal_spec_path,
            )
        outcome = execute_product_episode(
            args=args,
            task_spec=task_spec,
            scene=scene,
            scene_path=scene_path,
            instruction=instruction,
            goal_spec_path=goal_spec_path,
            output_dir=output_dir,
        )
    except BaseException as exc:
        # The product boundary must persist a structured failure for every
        # Python exception, including adapter/observation/schema regressions
        # and native Kit paths that surface as SystemExit.  Narrow exception
        # matching used to let those paths close Isaac Sim without result.json;
        # that makes a complete-episode benchmark unauditable.  Explicit
        # operator interruption remains outside this product failure record.
        if isinstance(exc, (KeyboardInterrupt, GeneratorExit)):
            raise
        failure_result = _write_failure_artifacts(
            args=args,
            task_spec=task_spec,
            scene=scene,
            scene_path=scene_path,
            goal_spec_path=goal_spec_path,
            output_dir=output_dir,
            gpu_health=gpu_health,
            exc=exc,
        )
        print(json.dumps(failure_result, ensure_ascii=False))
        return 1

    payload = outcome if isinstance(outcome, dict) else outcome.to_json()
    write_json(output_dir / "entrypoint_result.json", payload)
    write_json(
        output_dir / "manifest.json",
        _build_manifest(
            args=args,
            task_spec=task_spec,
            scene=scene,
            scene_path=scene_path,
            goal_spec_path=goal_spec_path,
            output_dir=output_dir,
            payload=payload,
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload.get("status") == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
