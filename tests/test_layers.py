"""The three layers: adapters (anything in -> robot_msgs), dashboards, viewers (Rerun now, the rest on the roadmap)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from robot_msg_viz.ruler import read
from robot_msgs.sensor import JointState
from robot_msgs.std import Header

import robot_dashboards as dash
from robot_dashboards import adapters, viewers
from robot_dashboards.adapters import Adapter, ros2


def _states(n: int = 6) -> list[JointState]:
    names = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
             "wrist_3_joint")  # fmt: skip
    return [
        JointState(name=names, position=[0.1 * i, -1.2, 1.4, -1.6, -1.57, 0.0], header=Header(stamp=100.0 + 0.04 * i))
        for i in range(n)
    ]


# ── adapters ────────────────────────────────────────────────────────────────────────────────────────


def test_robot_msgs_messages_pass_through_unchanged():
    state = _states(1)[0]
    assert adapters.to_robot_msgs(state) is state
    assert adapters.adapter_for(state).name == "robot_msgs"


def test_ros2_messages_convert_through_robot_msgs_ros():
    from robot_msgs.ros import to_ros

    state = _states(1)[0]
    ros_msg = to_ros(state)
    assert adapters.adapter_for(ros_msg).name == "ros2"
    assert adapters.to_robot_msgs(ros_msg) == state


def test_a_ros2_bag_feeds_a_dashboard(tmp_path):
    from robot_msgs.ros import write_bag

    bag = tmp_path / "run"
    write_bag(bag, [("/joint_states", s) for s in _states()])
    ticks = ros2.bag_ticks(bag, joint_state="/joint_states")
    assert [round(t, 6) for t, _ in ticks] == [round(0.04 * i, 6) for i in range(6)]
    assert all(isinstance(inputs["joint_state"], JointState) for _, inputs in ticks)
    path = dash.save(tmp_path / "state.rrd", "robot_state", ticks)
    assert read(path).rows >= 6


def test_a_bag_without_the_topic_says_which_topics_it_has(tmp_path):
    from robot_msgs.ros import write_bag

    bag = tmp_path / "run"
    write_bag(bag, [("/joint_states", s) for s in _states(2)])
    with pytest.raises(KeyError, match="/joint_states"):
        ros2.bag_ticks(bag, joint_state="/nope")


def test_your_own_format_through_a_registered_adapter(tmp_path):
    @dataclass
    class MyArmLog:  # someone's own record, not a robot_msgs message
        stamp: float
        angles: list[float]

    adapters.register(Adapter(
        "my_arm_log", "test: a team's own arm log",
        convert=lambda log: _states(1)[0].replace(position=log.angles, header=Header(stamp=log.stamp)),
        accepts=lambda obj: isinstance(obj, MyArmLog),
    ))  # fmt: skip
    try:
        samples = [(0.04 * i, {"joint_state": MyArmLog(100.0 + 0.04 * i, [0.1 * i, -1.2, 1.4, -1.6, -1.57, 0.0])})
                   for i in range(4)]  # fmt: skip
        path = dash.save(tmp_path / "mine.rrd", "robot_state", samples)
        assert read(path).rows >= 4
    finally:
        adapters.ADAPTERS.pop("my_arm_log")


def test_an_unknown_input_type_names_the_two_ways_out():
    with pytest.raises(TypeError, match="convert it to a robot_msgs message, or register an Adapter"):
        adapters.to_robot_msgs(object())


def test_a_live_stream_adapts_each_tick(tmp_path):
    from robot_msgs.ros import to_ros

    path = tmp_path / "live.rrd"
    with dash.stream("robot_state", spawn=False, save=path) as live:
        for i, state in enumerate(_states(3)):
            live.log(t=0.04 * i, joint_state=to_ros(state))
    assert read(path).rows >= 3


# ── viewers ─────────────────────────────────────────────────────────────────────────────────────────


def test_rerun_is_supported_and_every_dashboard_has_a_rerun_renderer():
    assert viewers.get("rerun").status == "supported"
    for dashboard in dash.all_dashboards():
        assert "rerun" in dashboard.viewers


@pytest.mark.parametrize("name", [n for n, v in viewers.VIEWERS.items() if v.status == "roadmap"])
def test_a_roadmap_viewer_says_what_it_still_needs(name, tmp_path):
    with pytest.raises(NotImplementedError, match="roadmap"):
        dash.save(tmp_path / "x", "object_detection", viewer=name)


def test_an_unknown_viewer_lists_the_known_ones():
    with pytest.raises(KeyError, match="rerun"):
        dash.show("object_detection", viewer="paraview")
