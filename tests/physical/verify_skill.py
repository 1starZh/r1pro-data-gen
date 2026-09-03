"""Verify one reusable skill physically in Isaac Sim and record one MP4.

The scenario supplies replaceable test parameters; it does not change the
skill implementation. The only persistent artifact is
``outputs/skills/<skill>.mp4``. Pass/fail and diagnostics are printed to the
console so the output directory remains suitable for video review and dataset
collection.

Usage:
    CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src <isaaclab python> \\
        tests/physical/verify_skill.py --skill arm_move_to --scene tabletop_basic \
            --headless --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS

PROJECT_ROOT = Path.cwd() if (Path.cwd() / "asset").is_dir() else Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "asset" / "r1pro" / "r1_pro_with_gripper.urdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "skills"


def _restore_backup(backup_path: Path, output_path: Path) -> None:
    """Restore a backup even when /tmp and the workspace use different filesystems."""
    if not backup_path.is_file():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, output_path)
    backup_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", required=True, help="Skill name to verify")
    parser.add_argument(
        "--scene",
        default="tabletop_basic",
        help="Fixture name/path or verified TaskSpec id (for example pickplace.tabletop)",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory containing only <skill>.mp4 files (default: outputs/skills)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        choices=(DEFAULT_VIDEO_FPS,),
        help=f"Video fps (fixed: {DEFAULT_VIDEO_FPS})",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--side", choices=("left", "right"), default="left", help="Arm/gripper side used by side-aware showcase scenarios")
    parser.add_argument("--physical-gpu-id", type=int, default=6, help="Physical Vulkan/RTX GPU index (project-pinned to 6)")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.physical_gpu_id != 6:
        parser.error("this project is pinned to physical GPU 6")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.skill}.mp4"
    # Preserve a previously validated video while a new run is in progress.
    # Isaac Sim can terminate its Python host during shutdown, so the recorder
    # writes directly to the final MP4 and the exception path restores this
    # explicit backup when possible.
    backup_path = Path("/tmp") / f"r1pro_skill_backup_{os.getpid()}_{args.skill}.mp4"
    backup_path.unlink(missing_ok=True)
    if output_path.is_file():
        shutil.copy2(output_path, backup_path)
    output_path.unlink(missing_ok=True)
    if not args.urdf.is_file():
        parser.error(f"URDF does not exist: {args.urdf}")

    from skill_scenarios import get_scenario, prepare_scenario

    try:
        scenario = get_scenario(args.skill)
    except KeyError as exc:
        parser.error(str(exc))

    args.headless = True
    args.livestream = 0
    args.enable_cameras = True
    args.renderer = "RayTracedLighting"
    args.multi_gpu = False
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
    exit_code = 1
    result_line: str | None = None

    try:
        from r1pro_data_gen.robot import R1PRO_ARM_VELOCITY_LIMITS
        from r1pro_data_gen.robot.kinematics import R1ProKinematics
        from r1pro_data_gen.data.scenes import load_scene_data, load_scene_yaml
        from r1pro_data_gen.simulation.isaac_sim import AdapterCfg, R1ProSimAdapter
        from r1pro_data_gen.simulation.isaac_sim.video import VideoRecorder
        from r1pro_data_gen.skills import build_default_registry
        from r1pro_data_gen.tasks import load_task_spec

        if "." in args.scene and "/" not in args.scene and "\\" not in args.scene:
            task_spec = load_task_spec(args.scene)
            task_spec.require_human_verified()
            scene = load_scene_data(task_spec.scene, source=task_spec.source_path or task_spec.id)
        else:
            scene_path = Path(args.scene)
            if not scene_path.is_absolute():
                scene_path = PROJECT_ROOT / scene_path
            if not scene_path.is_file():
                scene_path = PROJECT_ROOT / "tests" / "fixtures" / "scenes" / f"{args.scene}.yaml"
            if not scene_path.is_file():
                parser.error(f"scene fixture does not exist: {scene_path}")
            scene = load_scene_yaml(scene_path)
        kin = R1ProKinematics(str(args.urdf), side=args.side)
        registry = build_default_registry(kin, np.asarray(R1PRO_ARM_VELOCITY_LIMITS))
        if args.skill not in registry:
            raise KeyError(f"skill {args.skill!r} not in registry (available: {registry.names})")

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
        adapter.add_distant_light()
        adapter.reset()
        adapter.set_camera_view()
        adapter._verification_side = args.side

        # Some physical fixtures depend on live post-reset kinematics. Set
        # them up before the recorder starts so the video never contains a
        # falling placeholder followed by a visible state teleport.
        prepared, preparation_metrics = prepare_scenario(
            args.skill, adapter, kin, scene, registry
        )
        if not prepared:
            raise RuntimeError(f"scenario preparation failed: {preparation_metrics}")

        recorder = VideoRecorder(adapter, output_path, fps=args.fps)

        def step_hook() -> None:
            recorder.step_hook()

        # Render warmup: every step is also recorded so short/read-only scenarios
        # (e.g. the query skills, which never advance the simulation) still
        # produce a valid video.
        for _ in range(30):
            adapter.step()
            recorder.step_hook()

        ok, metrics = scenario(adapter, kin, scene, registry, step_hook)

        # Hold tail so even zero-step scenarios end with recorded frames.
        for _ in range(20):
            adapter.step()
            recorder.step_hook()

        video_stats = recorder.write_and_validate()

        video_ok = bool(video_stats.get("video_rgb_valid", 0.0))
        final_ok = bool(ok and video_ok)
        if not final_ok:
            output_path.unlink(missing_ok=True)
            _restore_backup(backup_path, output_path)
        result_line = (
            "SKILL_RESULT "
            f"skill={args.skill} scene={args.scene} side={args.side} "
            f"result={'passed' if final_ok else 'failed'} "
            f"frames={int(video_stats['video_frame_count'])} "
            f"bytes={int(video_stats['video_bytes'])} "
            f"mean_rgb={video_stats['video_mean_rgb']:.1f} "
            f"metrics={_compact_metrics(metrics)} "
            f"video={output_path}"
        )
        exit_code = 0 if final_ok else 1
        print(result_line, flush=True)
    except BaseException as exc:
        import traceback

        result_line = (
            f"SKILL_RESULT skill={args.skill} scene={args.scene} result=error "
            f"exception={type(exc).__name__}: {exc}"
        )
        traceback.print_exc(file=sys.stderr)
        output_path.unlink(missing_ok=True)
        _restore_backup(backup_path, output_path)
        exit_code = 1
        print(result_line, flush=True)
    finally:
        try:
            if "adapter" in locals():
                adapter.cleanup()
            from isaaclab.sim import SimulationContext

            SimulationContext.clear_instance()
        except Exception:  # pragma: no cover - best-effort resource release
            pass
        # Kit may terminate its Python host with status 0 from close(), which
        # used to hide a failed physical gate from run_all.sh. On failure the
        # output video has already been removed/restored, so exit before Kit's
        # shutdown handler can overwrite the verifier status.
        if exit_code != 0:
            os._exit(int(exit_code))
        simulation_app.close()
    backup_path.unlink(missing_ok=True)
    return exit_code


def _compact_metrics(metrics: object) -> str:
    """Render scalar showcase metrics without creating a sidecar artifact."""
    if not isinstance(metrics, dict):
        return "{}"
    keys = (
        "reason", "waypoints", "planned_path_length_m", "path_length_m",
        "planned_detour_m", "actual_detour_m", "min_crate_clearance_m",
        "required_footprint_radius_m", "arrival_error_m", "replay_arrival_error_m",
        "replay_reason", "replay_failed_waypoint", "cylinder_drift_m", "forces",
        "moved_m", "final_error_rad", "max_joint_step_rad",
    )
    values = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, (list, tuple)) and len(value) > 8:
            value = list(value[:8]) + ["..."]
        values.append(f"{key}={value}")
    return "{" + ",".join(values) + "}"


if __name__ == "__main__":
    # Isaac Sim shutdown can install an exit handler of its own.  main() has
    # already released the simulation before returning, so force the verifier
    # status through to run_all.sh instead of allowing Kit to turn a failed
    # skill gate into a shell-level success.
    _exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(_exit_code))
