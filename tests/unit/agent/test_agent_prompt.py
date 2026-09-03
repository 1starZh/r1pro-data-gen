from r1pro_data_gen.agent.prompt import system_prompt


def test_prompt_is_family_strategy_not_a_scene_recipe() -> None:
    text = system_prompt([])
    assert "Do not follow a scene-specific recipe" in text
    assert "ordinary tabletop pick-and-place" not in text
    assert "grasp_object, arm_carry_object_to, then release_object" not in text
    assert "call prepare_workspace then retry" in text
    assert "retry grasp_object from the live stance" in text
    assert "if the destination is on the same support" in text
    assert "observe object size and pose" in text
    assert "reachable_from_here" in text
    assert "keep side=auto" in text
    assert "prepare_workspace" in text
    assert "Do not emit arm_move_through" in text
    assert "prefer the public whole_body_transfer_object_between_supports skill" not in text
    assert "Use support_aware_grasp_object" not in text
    assert "arm_align_gripper" not in text
    assert "gripper_set" not in text
