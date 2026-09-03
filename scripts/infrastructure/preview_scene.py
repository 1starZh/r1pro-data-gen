"""Render an unverified TaskSpec scene for a few seconds so a human can inspect it.

This is not a product rollout. It does not require scene_human_verified and
does not run the agent.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from r1pro_data_gen.video_config import DEFAULT_VIDEO_FPS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="TaskSpec id or YAML path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--physical-gpu-id", type=int, default=7)
    return parser


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.fps != DEFAULT_VIDEO_FPS:
        parser.error(f"video fps is fixed at {DEFAULT_VIDEO_FPS}")
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")

    from r1pro_data_gen.data.scenes import load_scene_data
    from r1pro_data_gen.tasks import load_task_spec

    task_spec = load_task_spec(args.task)
    scene = load_scene_data(task_spec.scene, source=task_spec.source_path or task_spec.id)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    adapter = None
    try:
        from r1pro_data_gen.simulation.isaac_sim import AdapterCfg, R1ProSimAdapter
        from r1pro_data_gen.simulation.isaac_sim.video import VideoRecorder

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
        adapter.lock_joint_mask(
            mask_mode="lock",
            joint_groups=("wheel", "torso"),
            lock_root=False,
            gain_overrides={"wheel": (500.0, 100.0)},
        )
        recorder = VideoRecorder(adapter, output_dir / "rollout.mp4", fps=args.fps)
        steps = max(1, int(round(float(args.duration_s) * 60.0)))
        for _ in range(steps):
            adapter.step()
            recorder.step_hook()
        stats = recorder.write_and_validate()
        print(
            {
                "task": task_spec.id,
                "scene": scene.name,
                "scene_human_verified": task_spec.scene_human_verified,
                "video": str(recorder.output_path),
                **stats,
            }
        )
        return 0
    finally:
        try:
            if adapter is not None:
                adapter.cleanup()
            from isaaclab.sim import SimulationContext

            SimulationContext.clear_instance()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
