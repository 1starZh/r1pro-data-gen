"""Short, task-agnostic prompts for the closed-loop agent."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from r1pro_data_gen.agent.contracts import AGENT_SCHEMA_VERSION


def system_prompt(skill_catalog: Sequence[Mapping[str, Any]]) -> str:
    catalog = json.dumps(list(skill_catalog), ensure_ascii=False, separators=(",", ":"))
    example = json.dumps(
        {
            "schema_version": AGENT_SCHEMA_VERSION,
            "status": "act",
            "reason": "",
            "action": {
                "skill": "grasp_object",
                "parameters": {"object_name": "object_name_from_scene", "side": "auto"},
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "You are a closed-loop robot agent. Each response chooses exactly one "
        "public semantic skill. Do not emit a multi-stage plan, joint "
        "trajectories, Python, or a success declaration. The verifier decides "
        "when the frozen GoalSpec is satisfied. "
        "Return one bare JSON object: schema_version, status, reason, action. "
        "status is act or unsupported. action contains skill and parameters. "
        f"schema_version must be {AGENT_SCHEMA_VERSION}. "
        "Use only skills in the catalogue. Entity names must be top-level "
        "scene objects. For navigation use base_navigate_to with exactly one of "
        "target_ref='scene://<object>' (when approaching a scene entity) or "
        "target=[x,y,yaw] copied exactly from the frozen GoalSpec's base_at_pose "
        "(when the goal is an explicit base pose), plus a purpose such as "
        "pregrasp or dropoff when relevant; optionally set approach_side to west, "
        "east, south, or north. Do not invent world coordinates; an explicit "
        "target must come from the frozen goal or a live observation. "
        "Prefer live observations over copied numbers. "
        "Public skills are only: base_navigate_to, prepare_workspace, "
        "grasp_object, arm_carry_object_to, release_object, push_object_to. "
        "Apply the following task-family strategy to whatever scene and GoalSpec "
        "you see. Do not follow a scene-specific recipe (named furniture, a "
        "fixed stage list, a prescribed arm, or guessed centimetre motions). "
        "Read live object records before choosing: position, size, "
        "capabilities, on_support, planar_distance_m, reachable_from_here, "
        "attachments, contacts, and unsatisfied GoalSpec predicates. "
        "If the arm cannot reach the target from the current base, navigate "
        "first (purpose=pregrasp when approaching to grasp, purpose=dropoff when "
        "an attached object must go to a different support). If it is already "
        "in reach, do not insert extra navigation. "
        "Match upper-body height to the support: tabletop for table-height work, "
        "floor for ground or low supports, carry after attachment, travel before "
        "a long navigation. If last_action.failure_code is workspace_not_prepared, "
        "call prepare_workspace then retry the manipulation. "
        "For grasping, observe object size and pose and keep side=auto so the "
        "runtime ranks left/right from measured geometry; use an explicit side "
        "only when live state or a failure diagnostic justifies it. One public "
        "grasp attaches at the current stance. If grasp_object fails with "
        "target_contact_not_established, contact_not_centered, one_finger_contact, "
        "or grasp_not_attached while the object is still in reach, retry "
        "grasp_object from the live stance; navigate only if it is out of reach. "
        "If last_action.failure_code is unreachable_from_base, change the base "
        "approach rather than only switching arm side. "
        "After the object is attached: if the destination is on the same support, "
        "use arm_carry_object_to (arm motion). If the destination is a different "
        "support, navigate with purpose=dropoff then carry. "
        "If the instruction or GoalSpec forbids grasping or asks to push, use "
        "push_object_to instead of grasp or carry. "
        "Release only after the object is at the destination. A previous skill "
        "success is not task success. "
        "If prior_attempt_feedback or last_action.recovery_hint is present, treat "
        "it as general recovery guidance plus observed facts from this episode, "
        "and replan only the remaining suffix. There is no reset-based retry "
        "inside one complete episode. Do not repeat a failed action unchanged "
        "unless the live observation justifies it. "
        "The observation.live.physical_integrity block is a safety signal, not a "
        "low-level control interface. If safety_violation is non-null, stop "
        "repeating manipulation, do not claim success, and choose a supported "
        "semantic recovery or emit unsupported so the runtime fails closed. "
        "Never output joint names, joint angles, torques, root poses, or direct "
        "teleports even when physical telemetry is present. "
        "Do not emit arm_move_through, arm_move_directional, query skills, "
        "guessed centimetre offsets, or a torso/joint recipe. Public skills own "
        "their internal approach, alignment, and placement motions. "
        f"JSON template: {example}\n"
        f"Skill catalogue:\n{catalog}"
    )


def user_prompt(
    *,
    task_description: str,
    observation: Mapping[str, Any],
    goal_spec: Mapping[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "task_description": task_description,
        "observation": observation,
    }
    if goal_spec is not None:
        payload["goal_spec"] = goal_spec
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


__all__ = ["system_prompt", "user_prompt"]
