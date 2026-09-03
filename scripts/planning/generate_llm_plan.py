"""Generate a validated semantic Plan without starting Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from r1pro_data_gen.data.plan_io import save_plan
from r1pro_data_gen.planning.llm.providers import DeepSeekClient
from r1pro_data_gen.planning.context.facts import scene_to_facts
from r1pro_data_gen.planning.task.interfaces import TaskPlanningRequest
from r1pro_data_gen.planning.task.planner import LLMTaskPlanner
from r1pro_data_gen.robot import R1PRO_ARM_VELOCITY_LIMITS
from r1pro_data_gen.robot.kinematics import R1ProKinematics
from r1pro_data_gen.data.scenes import load_scene_data
from r1pro_data_gen.skills import build_default_registry
from r1pro_data_gen.tasks import load_task_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        help="TaskSpec id or YAML path; its scene and instruction define the plan",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output plan JSON path")
    parser.add_argument("--urdf", type=Path, required=True, help="R1Pro URDF used to build the skill catalogue")
    parser.add_argument("--constraints", type=Path, help="Optional JSON constraints object")
    parser.add_argument(
        "--skill-names",
        help=(
            "Optional comma-separated public skill subset to expose to the LLM. "
            "The selected names are still checked against the registry catalogue."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    task_spec = load_task_spec(args.task)
    scene = load_scene_data(
        task_spec.scene,
        source=task_spec.source_path or task_spec.id,
    )
    constraints = {}
    if args.constraints:
        constraints = json.loads(args.constraints.read_text(encoding="utf-8"))
        if not isinstance(constraints, dict):
            raise ValueError("constraints JSON must be an object")

    # The generator only exports the registry's semantic/public surface.
    if not args.urdf.is_file():
        raise FileNotFoundError(f"URDF does not exist: {args.urdf}")
    kin = R1ProKinematics(str(args.urdf))
    registry = build_default_registry(kin, np.asarray(R1PRO_ARM_VELOCITY_LIMITS))
    catalog = registry.llm_descriptions()
    if args.skill_names:
        requested_names = tuple(
            name.strip() for name in args.skill_names.split(",") if name.strip()
        )
        if not requested_names:
            raise ValueError("--skill-names must contain at least one skill name")
        available = {str(item["name"]): item for item in catalog}
        unknown = sorted(set(requested_names) - set(available))
        if unknown:
            raise ValueError(f"--skill-names contains non-public skills: {unknown}")
        catalog = [available[name] for name in requested_names]

    goal_spec_path = args.output.with_name("goal_spec.json")
    if not goal_spec_path.is_file():
        raise FileNotFoundError(
            "generic Plan generation requires goal_spec.json next to --output"
        )
    from r1pro_data_gen.domain import goal_spec_sha256, goal_spec_to_dict, parse_goal_spec

    goal_payload = json.loads(goal_spec_path.read_text(encoding="utf-8"))
    goal_spec = parse_goal_spec(goal_payload, scene)
    frozen_goal_hash = goal_spec_sha256(goal_spec)
    provenance_path = goal_spec_path.with_name(goal_spec_path.name + ".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    if provenance.get("goal_spec_hash") != frozen_goal_hash:
        raise ValueError("GoalSpec provenance hash does not match frozen GoalSpec")

    request = TaskPlanningRequest(
        task_description=task_spec.instruction,
        scene_facts=scene_to_facts(scene, kinematics=kin),
        skill_catalog=tuple(catalog),
        constraints=constraints,
        metadata={"scene_name": scene.name, "task_id": task_spec.id},
        goal_spec=goal_spec_to_dict(goal_spec),
        goal_spec_hash=frozen_goal_hash,
    )
    result = LLMTaskPlanner(DeepSeekClient.from_env()).plan(request)
    if result.status != "planned" or result.plan is None:
        print(json.dumps({"status": result.status, "reason": result.reason}, ensure_ascii=False))
        return 2
    save_plan(result.plan, args.output)
    # Persist the parsed raw provider response and usage as provenance, next to
    # the plan. ``raw_response`` is the validated JSON envelope parsed from the
    # provider text; it never contains the API key (the key is only sent in the
    # HTTP Authorization header and is never part of a prompt or response body).
    raw_path = args.output.with_name(args.output.name + ".raw.json")
    raw_path.write_text(
        json.dumps(
            {
                "status": result.status,
                "provider": result.provider,
                "model": result.model,
                "usage": result.usage,
                "raw_response": result.raw_response,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "planned", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
