"""Query skill behavior against the smallest adapter contract."""

from __future__ import annotations

from r1pro_data_gen.skills import QueryContacts, QueryEEPose, QueryJointPos, QueryObjectPose

from tests.support import FakeAdapter, FakeKinematics


def test_query_object_pose():
    adapter = FakeAdapter(object_positions={"cylinder": (0.5, 0.15, 1.11)})
    result = QueryObjectPose().execute(adapter, None, object_name="cylinder")
    assert result.success
    assert result.details["position"] == [0.5, 0.15, 1.11]


def test_query_object_pose_missing_object_fails():
    result = QueryObjectPose().execute(FakeAdapter(), None, object_name="nope")
    assert not result.success
    assert "not" in result.details["reason"]


def test_query_contacts_returns_sensor_forces():
    result = QueryContacts().execute(FakeAdapter(contacts=(2.5, 0.0)), None)
    assert result.success
    assert result.details["contact_forces"] == [2.5, 0.0]


def test_query_ee_pose_returns_position_and_quat():
    arm = {f"left_arm_joint{i}": 0.0 for i in range(1, 8)}
    result = QueryEEPose(FakeKinematics()).execute(FakeAdapter(joint_positions=arm), None, side="left")
    assert result.success
    assert result.details["position"] == [0.4, 0.0, 1.2]
    assert result.details["quaternion"] == [1.0, 0.0, 0.0, 0.0]


def test_query_joint_pos_returns_selected_joints():
    adapter = FakeAdapter(joint_positions={"left_arm_joint1": 0.1, "torso_joint1": 0.2})
    result = QueryJointPos().execute(adapter, None, joints=["left_arm_joint1"])
    assert result.success
    assert result.details["joint_positions"] == {"left_arm_joint1": 0.1}
