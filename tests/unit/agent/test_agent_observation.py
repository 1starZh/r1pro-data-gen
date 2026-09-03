from types import SimpleNamespace

from r1pro_data_gen.agent.observation import build_agent_observation


class _Adapter:
    def read_observation(self, timestamp):
        del timestamp
        return SimpleNamespace(base_pose=(0.0, 0.0, 0.0))

    def finger_contact_forces(self, side="left"):
        if side == "left":
            return (0.1, 0.2)
        if side == "right":
            return (0.3, 0.4)
        raise KeyError(side)


def test_agent_observation_reports_both_gripper_contacts() -> None:
    payload = build_agent_observation(
        adapter=_Adapter(),
        remaining_actions=4,
    )
    live = payload["live"]
    assert live["contacts_left"] == [0.1, 0.2]
    assert live["contacts_right"] == [0.3, 0.4]
    assert live["contacts"] == {"left": [0.1, 0.2], "right": [0.3, 0.4]}
