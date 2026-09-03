#!/usr/bin/env bash
# Record one clean showcase video per registered skill.
#
# The scenario parameters live in skill_scenarios.py and can be changed without
# changing the reusable skill implementations. The runner deliberately writes
# no manifest, JSON, summary, or per-run directory: outputs/skills contains
# only <skill>.mp4 files.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${ISAACLAB_PYTHON:-${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}}"
SIDE="${SIDE:-left}"
PHYSICAL_GPU_ID="${PHYSICAL_GPU_ID:-6}"
if [[ -z "$PY" || ! -x "$PY" ]]; then
  echo "Set ISAACLAB_PYTHON to the Isaac Lab 2.3 Python executable or activate its conda environment." >&2
  exit 2
fi
cd "$ROOT" || exit 1
failed=0

# Skill -> fixture. Default tabletop_basic (table + cylinder); pure robot skills use bare.
declare -A SCENE=(
  [arm_joint_to]=bare
  [arm_trajectory_follow]=bare
  [arm_move_to]=tabletop_basic
  [arm_move_directional]=bare
  [arm_rotate_ee]=bare
  [torso_move_to]=bare
  [gripper_set]=bare
  [base_move_to]=bare
  [base_rotate_to]=bare
  [base_follow_path]=bare
  [base_velocity_set]=bare
  [base_lock_wheels]=bare
  [base_unlock_wheels]=bare
  [joint_mask_lock]=bare
  [joint_mask_unlock]=bare
  [base_navigate_to]=pickplace.tabletop
  [gripper_grasp]=gripper_fixture
  [query_object_pose]=tabletop_basic
  [query_contacts]=tabletop_basic
  [query_ee_pose]=bare
  [query_joint_pos]=bare
  [query_ik_solution]=bare
  [query_arm_path]=tabletop_basic
  [query_base_path]=pickplace.tabletop
)

SKILLS=(
  # Execution layer: smooth joint-space / trajectory execution.
  arm_joint_to
  arm_trajectory_follow
  arm_move_to
  base_move_to
  base_rotate_to
  base_follow_path
  base_navigate_to
  base_velocity_set
  base_lock_wheels
  base_unlock_wheels
  joint_mask_lock
  joint_mask_unlock
  torso_move_to
  gripper_set
  # Feedback layer.
  arm_move_directional
  arm_rotate_ee
  gripper_grasp
  # Query/solve/plan skills (read-only / planning).
  query_object_pose
  query_contacts
  query_ee_pose
  query_joint_pos
  query_ik_solution
  query_arm_path
  query_base_path
)

for skill in "${SKILLS[@]}"; do
  scene="${SCENE[$skill]:-tabletop_basic}"
  echo "=== $skill ($scene) $(date +%H:%M:%S) ==="
  timeout 900 env CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/r1pro-matplotlib "$PY" \
    tests/physical/verify_skill.py --skill "$skill" --scene "$scene" --output-dir outputs/skills \
    --side "$SIDE" --headless --device cuda:0 --physical-gpu-id "$PHYSICAL_GPU_ID" --livestream 0
  rc=$?
  video="outputs/skills/$skill.mp4"
  if [[ $rc -eq 0 && ! -s "$video" ]]; then
    rc=1
  fi
  echo ">>> $skill EXIT=$rc"
  [[ $rc -eq 0 ]] || failed=$((failed + 1))
done

echo "=== ALL DONE $(date +%H:%M:%S), failed=$failed ==="
exit "$failed"
