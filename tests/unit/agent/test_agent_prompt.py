from r1pro_data_gen.agent.prompt import system_prompt


def test_tabletop_prompt_prefers_grasp_carry_release_not_micro_skills() -> None:
    text = system_prompt([])
    assert "grasp_object, arm_carry_object_to, then release_object" in text
    assert "ordinary tabletop pick-and-place" in text
    assert "Do not replace those stages with arm_move_through" in text
    assert "do not call base_navigate_to again" in text
    assert "retry grasp_object from the live stance" in text
    assert "prepare_workspace" in text
    assert "prefer the public whole_body_transfer_object_between_supports skill" not in text
    assert "Use support_aware_grasp_object" not in text
