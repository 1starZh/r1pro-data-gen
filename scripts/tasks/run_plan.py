"""Generic plan runner: TaskSpec + plan data -> skills -> evidence -> result.

This is the replay entrypoint used by the data-generation loop. It loads one
data-only TaskSpec, a plan from a JSON file, validates every skill call against
the skill library, executes the plan through the orchestrator, records video,
and evaluates the task result from execution evidence.

Usage:
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src <isaaclab python> \\
        scripts/tasks/run_plan.py --task pickplace.tabletop_complete \\
        --goal-spec <goal_spec.json> \\
        --plan <plan.json> \\
        [--stages s1,s2] [--output-dir ...]

``--stages`` runs only the given plan stages (e.g. the Phase 4 approach
regression: navigate,lock,open,rear,pregrasp).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from isaaclab.app import AppLauncher

from r1pro_data_gen.domain import (
    GoalSpec,
    Plan,
    SceneModel,
    evidence_to_dict,
    goal_spec_sha256,
    parse_goal_spec,
)
from r1pro_data_gen.data.plan_io import load_plan
from r1pro_data_gen.data.scenes import load_scene_data, write_scene_yaml
from r1pro_data_gen.tasks import TaskSpec, load_task_spec
from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "asset").is_dir() else Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "run_plan"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        help="TaskSpec id or YAML path; the spec supplies the scene and instruction",
    )
    parser.add_argument(
        "--goal-spec",
        type=Path,
        help="Frozen GoalSpec JSON path",
    )
    parser.add_argument("--plan", type=Path, help="Path to the plan.json")
    parser.add_argument(
        "--external-llm-plan",
        action="store_true",
        help="Apply the strict external-LLM skill allowlist and plan limits",
    )
    parser.add_argument("--stages", help="Comma-separated subset of plan stages to run (default: all)")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        choices=(DEFAULT_VIDEO_FPS,),
        help=f"Video fps (fixed: {DEFAULT_VIDEO_FPS})",
    )
    parser.add_argument(
        "--evidence-hz",
        type=float,
        default=10.0,
        help=(
            "Continuous evidence sampling rate; stage boundaries are always "
            "captured regardless of this cadence"
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-action-seconds",
        type=float,
        default=600.0,
        help=(
            "Wall-clock budget for each semantic skill call; increase this "
            "for slow physics/rendering workloads without changing safety limits."
        ),
    )
    parser.add_argument(
        "--physical-gpu-id",
        type=int,
        default=6,
        help=(
            "Physical Vulkan/RTX GPU index. Use this with CUDA_VISIBLE_DEVICES "
            "when CUDA logical numbering differs from Omniverse's physical numbering."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _json_safe(value: object) -> object:
    """Convert runtime evidence objects into the JSON result contract."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _contains_runtime_reference(value: object) -> bool:
    """Return whether a plan parameter tree contains a typed runtime reference."""
    if isinstance(value, dict):
        return "ref" in value or any(
            _contains_runtime_reference(item) for item in value.values()
        )
    if isinstance(value, (tuple, list)):
        return any(_contains_runtime_reference(item) for item in value)
    return False


def _requires_collision_observation(goal_spec: GoalSpec | None) -> bool:
    """Return whether frozen goals require collision telemetry coverage."""
    if goal_spec is None:
        return False
    return any(
        predicate.predicate == "collision_free"
        for predicate in (*goal_spec.required, *goal_spec.invariants)
    )


def load_goal_spec(path: Path, scene: SceneModel) -> GoalSpec:
    """Load and ground a frozen GoalSpec without importing task packages."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"goal spec is unreadable: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("goal spec JSON must contain an object")
    return parse_goal_spec(payload, scene)


def write_marker(output_dir: Path, name: str, **details: object) -> None:
    write_json(output_dir / f".run_plan_{name}.json", {"marker": name, **details})


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
        if not root.is_dir():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path.suffix not in {".py", ".json", ".yaml", ".yml"}:
                continue
            digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _replay_manifest(
    *,
    args: argparse.Namespace,
    task_spec: TaskSpec,
    scene: SceneModel,
    scene_path: Path,
    plan_path: Path,
    output_dir: Path,
    result: Mapping[str, object],
) -> dict[str, object]:
    """Build the same auditable identity contract used by run_task.py."""
    video = result.get("video")
    video_path = None
    if isinstance(video, str) and video:
        video_path = str(
            (Path(video) if Path(video).is_absolute() else output_dir / video).resolve()
        )
    goal_spec_path = args.goal_spec.resolve()
    return {
        "schema_version": 2,
        "entrypoint": "scripts/tasks/run_plan.py",
        "code": {
            "fingerprint_sha256": _code_fingerprint(),
            "source_roots": ["src", "scripts"],
        },
        "command": list(sys.argv),
        "task": {
            "id": task_spec.id,
            "family": task_spec.family,
            "path": str(task_spec.source_path.resolve()) if task_spec.source_path else None,
            "sha256": _sha256_file(task_spec.source_path.resolve())
            if task_spec.source_path
            else None,
        },
        "scene": {
            "name": scene.name,
            "source": "embedded_task_spec",
            "human_verified": task_spec.scene_human_verified,
            "path": str(scene_path.resolve()),
            "sha256": _sha256_file(scene_path.resolve()),
        },
        "goal_spec": {
            "path": str(goal_spec_path),
            "sha256": result.get("goal_spec_hash"),
            "contract_path": str((output_dir / "goal_contract.json").resolve()),
            "contract_sha256": result.get("goal_contract_hash"),
        },
        "robot_asset": scene.robot.asset,
        "fps": result.get("video_fps", args.fps),
        "seed": result.get("seed", args.seed),
        "provider": "frozen_plan",
        "model": None,
        "evaluation_mode": result.get("evaluation_mode"),
        "artifact_paths": {
            "scene": str(scene_path.resolve()),
            "goal_spec": str(goal_spec_path),
            "plan": str(plan_path.resolve()),
            "evidence": str((output_dir / "evidence.json").resolve()),
            "result": str((output_dir / "result.json").resolve()),
            "video": video_path,
        },
        "acceptance_status": (
            result.get("acceptance", {}).get("status")
            if isinstance(result.get("acceptance"), Mapping)
            else None
        ),
    }


def _marker_component(value: str) -> str:
    """Return a filesystem-safe component for per-stage lifecycle markers."""
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.task:
        parser.error("--task is required")
    if args.goal_spec is None:
        parser.error("--goal-spec is required")
    if args.plan is None:
        parser.error("--plan is required")
    if args.physical_gpu_id != 6:
        parser.error("this project is pinned to physical GPU 6")
    if not math.isfinite(args.evidence_hz) or args.evidence_hz <= 0.0:
        parser.error("--evidence-hz must be finite and greater than zero")
    if args.task is not None and not args.task.strip():
        parser.error("--task must not be empty")
    try:
        task_spec = load_task_spec(args.task)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_marker(output_dir, "main_start", argv=sys.argv)
    try:
        task_spec.require_human_verified()
        scene = load_scene_data(
            task_spec.scene,
            source=task_spec.source_path or task_spec.id,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    scene_path = write_scene_yaml(task_spec.scene, output_dir / "scene.yaml")
    if not args.urdf.is_file():
        parser.error(f"URDF does not exist: {args.urdf}")

    args.headless = True
    args.livestream = 0
    args.enable_cameras = True
    args.renderer = "RayTracedLighting"
    args.multi_gpu = False
    # Keep the logical CUDA device (normally cuda:0) pinned to the requested
    # physical card.  An explicitly supplied mask wins, while the product
    # default prevents a replay from creating CUDA contexts on every visible
    # GPU.  Vulkan may still enumerate adapters for its startup probe, but the
    # simulation tensors and renderer remain bound to the selected device.
    if args.physical_gpu_id is not None and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_id)
    kit_args = [
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        "--/renderer/multiGpu/maxGpuCount=1",
        "--/validate/p2p/enabled=false",
        "--/validate/p2p/memoryCheck/enabled=false",
        "--/rtx/post/tonemap/op=4",
        "--/rtx/post/tonemap/filmIso=200",
        "--/rtx/post/tonemap/enabled=true",
    ]
    if args.physical_gpu_id is not None:
        kit_args.append(f"--/renderer/activeGpu={args.physical_gpu_id}")
    args.kit_args = " ".join(kit_args)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    # Keep product-level randomization reproducible without importing torch
    # before Isaac's timeline is running.  The seed is also persisted in the
    # result and supplied by run_llm_loop for every retry.
    random.seed(args.seed)
    np.random.seed(args.seed)
    write_marker(
        output_dir,
        "app_started",
        device=args.device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        renderer_multi_gpu=False,
        physical_gpu_id=args.physical_gpu_id,
    )
    print(
        f"[run_plan] Isaac Sim started headless on physical GPU {args.physical_gpu_id}",
        flush=True,
    )

    try:
        from r1pro_data_gen.evaluation import (
            PredicateVerifier,
            VerificationPolicy,
            finalize_result_payload,
        )
        from r1pro_data_gen.robot import R1PRO_ARM_VELOCITY_LIMITS
        from r1pro_data_gen.robot.kinematics import R1ProKinematics
        from r1pro_data_gen.simulation import EvidenceRecorder
        from r1pro_data_gen.simulation.isaac_sim import AdapterCfg, R1ProSimAdapter
        from r1pro_data_gen.simulation.isaac_sim.video import VideoRecorder
        from r1pro_data_gen.execution import Orchestrator
        from r1pro_data_gen.planning.llm.contracts import (
            LLM_PUBLIC_SKILLS,
            validate_plan,
        )
        from r1pro_data_gen.planning.context.facts import object_names, scene_to_facts
        from r1pro_data_gen.planning.goals.compiler import GoalCompiler
        from r1pro_data_gen.skills import build_default_registry

        # --- TaskSpec + frozen GoalSpec/plan (data-driven) ---
        goal_spec = load_goal_spec(args.goal_spec, scene)
        goal_hash = goal_spec_sha256(goal_spec) if goal_spec is not None else None
        compiled_goal = GoalCompiler().compile(goal_spec, scene) if goal_spec is not None else None
        goal_contract_hash = compiled_goal.contract_hash if compiled_goal is not None else None
        if compiled_goal is not None:
            write_json(output_dir / "goal_contract.json", compiled_goal.to_dict())
            write_marker(
                output_dir,
                "goal_contract_compiled",
                contract_hash=compiled_goal.contract_hash,
                required_observations=list(compiled_goal.required_observations),
            )
        plan_path = args.plan
        plan = load_plan(plan_path)
        if args.stages:
            allowed = set(args.stages.split(","))
            plan = Plan(
                task_name=plan.task_name,
                stages=tuple(s for s in plan.stages if s.name in allowed),
                metadata={**plan.metadata, "stages_subset": list(allowed)},
            )
        if not plan.stages:
            parser.error("plan has no stages to run (check --stages)")
        write_marker(
            output_dir,
            "plan_loaded",
            task=plan.task_name,
            stages=list(plan.stage_names),
            evaluation_mode="goal_spec",
            goal_spec_hash=goal_hash,
            goal_contract_hash=goal_contract_hash,
        )

        # --- Skills: validate the plan against the skill library ---
        kin = R1ProKinematics(str(args.urdf))
        registry = build_default_registry(kin, np.asarray(R1PRO_ARM_VELOCITY_LIMITS))
        if args.external_llm_plan:
            plan = validate_plan(
                plan,
                skill_catalog=registry.llm_descriptions(),
                registry=registry,
                scene_object_names=object_names(scene_to_facts(scene)),
            )
            write_marker(output_dir, "external_plan_policy_validated", skills=sorted(LLM_PUBLIC_SKILLS))
        for stage in plan.stages:
            skill_name = stage.parameters.get("skill")
            if skill_name not in registry:
                raise KeyError(f"stage {stage.name!r}: unknown skill {skill_name!r} (available: {registry.names})")
            if not (args.external_llm_plan and _contains_runtime_reference(stage.parameters)):
                registry.validate_plan_params(str(skill_name), stage.parameters)
        write_marker(output_dir, "plan_validated", skills=registry.names)

        # --- Simulation ---
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
        write_marker(output_dir, "adapter_initialized", device=args.device)
        adapter.add_distant_light()
        adapter.reset()
        write_marker(output_dir, "adapter_reset")
        adapter.set_camera_view()
        write_marker(output_dir, "camera_initialized")
        evidence_recorder = EvidenceRecorder(adapter, scene)
        evidence_interval_s = 1.0 / float(args.evidence_hz)
        current_stage: str | None = None
        continuous_step_count = 0
        write_marker(
            output_dir,
            "initial_capture_start",
            sim_time=float(adapter.sim.current_time),
        )
        evidence_recorder.capture(float(adapter.sim.current_time), stage="__initial__")
        write_marker(
            output_dir,
            "initial_capture_complete",
            sim_time=float(adapter.sim.current_time),
        )
        # Start recording before warmup. If the first plan stage fails (for
        # example navigation rejects an unsafe target), the rollout must still
        # contain a useful startup/failure window instead of a one-frame MP4
        # that appears as a 0-second video in common players.
        recorder = VideoRecorder(adapter, output_dir / "rollout.mp4", fps=args.fps)
        # Hold chassis + torso + root during the render warmup so the floating
        # base does not drift between reset and the first skill call.  Even a
        # few millimetres of unbraked drift shifts the occupancy-grid origin,
        # which flips the free/blocked verdict of a candidate stance sitting on
        # a cell boundary.  The mask is released before the orchestrator runs.
        adapter.lock_joint_mask(
            mask_mode="lock",
            joint_groups=("wheel", "torso"),
            # The floating base remains a physical DOF.  Joint masks hold
            # only the measured wheel/torso targets; they never apply a root
            # pose, force, or torque assist.
            lock_root=False,
            gain_overrides={"wheel": (500.0, 100.0)},
        )
        write_marker(output_dir, "warmup_start", sim_time=float(adapter.sim.current_time))
        for _ in range(60):  # render warmup and record a one-second startup window
            adapter.step()
            recorder.step_hook()
            evidence_recorder.capture_if_due(
                float(adapter.sim.current_time),
                stage=current_stage,
                min_interval_s=evidence_interval_s,
            )
        setup_violation = adapter.physical_safety_violation()
        if setup_violation is not None:
            raise RuntimeError(
                f"physical setup safety gate failed before plan execution: {setup_violation}"
            )
        adapter.rebaseline_physical_metrics()
        adapter.unlock_joint_mask()
        write_marker(output_dir, "warmup_complete", sim_time=float(adapter.sim.current_time))
        def step_hook() -> None:
            nonlocal continuous_step_count
            recorder.step_hook()
            evidence_recorder.capture_if_due(
                float(adapter.sim.current_time),
                stage=current_stage,
                min_interval_s=evidence_interval_s,
            )
            continuous_step_count += 1
            if continuous_step_count % 300 == 0:
                write_marker(
                    output_dir,
                    "heartbeat",
                    stage=current_stage,
                    physics_steps=continuous_step_count,
                    sim_time=float(adapter.sim.current_time),
                )

        def stage_hook(stage_name: str) -> None:
            nonlocal current_stage
            current_stage = stage_name
            write_marker(
                output_dir,
                f"stage_start_{_marker_component(stage_name)}",
                stage=stage_name,
                sim_time=float(adapter.sim.current_time),
            )
            write_marker(
                output_dir,
                f"stage_capture_start_{_marker_component(stage_name)}",
                stage=stage_name,
                sim_time=float(adapter.sim.current_time),
            )
            evidence_recorder.capture(float(adapter.sim.current_time), stage=stage_name)
            write_marker(
                output_dir,
                f"stage_capture_complete_{_marker_component(stage_name)}",
                stage=stage_name,
                sim_time=float(adapter.sim.current_time),
            )

        def frame_converter(value, source: str, target: str):
            """Calibrated world<->base conversion for typed references.

            The URDF model frame used by planning and the USD articulation root
            frame differ by an offset that grows once the arm leaves home (the
            link origins are not the same model).  Registering live link
            positions online recovers ``p_world = R @ p_model + t``; without it
            every ``frame=base`` scene.object reference lands ~10 cm off the
            object.

            In practice the online registration is only trustworthy near the
            neutral home: once the arm reaches a large posture the fitted
            transform drifts by several cm (URDF/USD link mismatch is
            posture-dependent), which makes an absolute scene.object goal worse
            than the plain ``base_pose`` geometry.  Scene objects are static, so
            ``base_pose`` is exact for them; the posture-dependent execution
            error is corrected later by the measurement-driven arm_align_gripper.
            We therefore always return ``None`` to keep the base_pose fallback.
            """
            return None

        def stage_end_hook(stage_name: str, success: bool) -> None:
            write_marker(
                output_dir,
                f"stage_end_{_marker_component(stage_name)}",
                stage=stage_name,
                success=bool(success),
                sim_time=float(adapter.sim.current_time),
            )
            write_marker(
                output_dir,
                f"stage_end_capture_start_{_marker_component(stage_name)}",
                stage=stage_name,
                sim_time=float(adapter.sim.current_time),
            )
            evidence_recorder.capture(
                float(adapter.sim.current_time),
                stage=stage_name,
            )
            write_marker(
                output_dir,
                f"stage_end_capture_complete_{_marker_component(stage_name)}",
                stage=stage_name,
                sim_time=float(adapter.sim.current_time),
            )
            evidence_recorder.finish_stage(
                float(adapter.sim.current_time),
                stage_name,
                success=success,
            )

        orchestrator = Orchestrator(
            adapter,
            registry,
            scene=scene,
            step_hook=step_hook,
            stage_hook=stage_hook,
            stage_end_hook=stage_end_hook,
            frame_converter=frame_converter,
            max_action_seconds=args.max_action_seconds,
        )
        write_marker(output_dir, "orchestrator_start", stages=list(plan.stage_names))
        execution = orchestrator.run_plan(plan)
        write_marker(
            output_dir,
            "orchestrator_complete",
            success=execution.success,
            completed=list(execution.completed),
            failed=execution.failed,
        )

        # --- Evaluate from execution evidence ---
        video_stats = recorder.write_and_validate()
        # Evidence coverage is about the states and boundaries that were
        # actually observed. An unsuccessful stage is still an observed stage;
        # its outcome is carried separately by the execution record.
        observed_stages = list(execution.stage_calls)
        if execution.failed is not None and execution.failed not in observed_stages:
            observed_stages.append(execution.failed)
        evidence = evidence_recorder.finish(
            complete=True,
            expected_stages=tuple(observed_stages),
            require_collision_observation=_requires_collision_observation(goal_spec),
        )
        evidence_path = output_dir / "evidence.json"
        write_json(evidence_path, evidence_to_dict(evidence))
        verification = PredicateVerifier().verify(
            goal_spec,
            evidence,
            VerificationPolicy(),
        )
        result = _json_safe({
            "result": "failed",
            "task": task_spec.id,
            "task_family": task_spec.family,
            "scene": scene.name,
            "plan_stages": list(plan.stage_names),
            "evaluation_mode": "goal_spec",
            "goal_spec_hash": goal_hash,
            "goal_contract_hash": goal_contract_hash,
            "execution": {
                "completed": list(execution.completed),
                "failed": execution.failed,
                "failure_reason": execution.failure_reason,
                "stage_results": {
                    name: {
                        "skill": r.skill,
                        "success": r.success,
                        **r.metrics,
                        "details": r.details,
                        "call": _json_safe(execution.stage_calls.get(name)),
                    }
                    for name, r in execution.stage_results.items()
                },
            },
            "evaluation": {
                "status": verification.status.value,
                "failure_reason": verification.failure_reason,
                "predicates": verification.predicates,
                "evidence_complete": verification.evidence_complete,
                "evidence_coverage_complete": evidence.complete,
                "stage_success_complete": evidence.stage_success_complete,
                "collision_observation_complete": evidence.collision_observation_complete,
            },
            "evidence_path": str(evidence_path),
            "evidence_hz": float(args.evidence_hz),
            "evidence_frame_count": len(evidence.frames),
            "video": str(recorder.output_path),
            **video_stats,
            "gpu_logical_device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu_id": args.physical_gpu_id,
            "seed": args.seed,
        })
        result = finalize_result_payload(
            result,
            expected_goal_spec_hash=goal_hash,
            expected_contract_hash=goal_contract_hash,
        )
        all_gates = result["acceptance"]["status"] == "accepted"
        write_json(output_dir / "result.json", result)
        manifest = _replay_manifest(
            args=args,
            task_spec=task_spec,
            scene=scene,
            scene_path=scene_path,
            plan_path=plan_path,
            output_dir=output_dir,
            result=result,
        )
        write_json(output_dir / "manifest.json", manifest)
        (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        write_marker(output_dir, "result_written", result=result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if all_gates else 1
    except BaseException as exc:
        import traceback

        write_marker(
            output_dir,
            "error",
            exception_type=type(exc).__name__,
            exception=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        write_marker(output_dir, "shutdown_started")
        try:
            if "adapter" in locals():
                adapter.cleanup()
            from isaaclab.sim import SimulationContext

            SimulationContext.clear_instance()
        except Exception:  # pragma: no cover - best-effort resource release
            pass
        write_marker(output_dir, "shutdown_cleanup_done")
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
