"""ROS 2 in: rosbags message objects, or a whole MCAP / rosbag2 recording, as dashboard ticks.

    from robot_dashboards.adapters import ros2
    run = ros2.bag_ticks("run_0412/", image="/camera/color/image_raw", detections="/detector/detections")
    dash.show("object_detection", ticks=run)

No ROS install: robot_msgs.ros (the rosbags typestore) does the decoding, so this adapter is the conversion
robot_msgs already owns, never a second copy. Needs ``pip install 'robot-msgs[ros]'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_dashboards.adapters import Adapter, Tick


def _from_ros(obj: Any) -> Any:
    from robot_msgs.ros import from_ros  # lazy: rosbags is an extra

    return from_ros(obj)


ADAPTER = Adapter(
    "ros2",
    "ROS 2 messages (rosbags dataclasses from a bag, a live rclpy bridge or robot_msgs.ros.to_ros), via robot_msgs.ros",
    convert=_from_ros,
    accepts=lambda obj: hasattr(obj, "__msgtype__"),
)


def _stamp(msg: Any, fallback_ns: int) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    return float(stamp) if stamp else fallback_ns * 1e-9


def bag_ticks(path: str | Path, **topics: str) -> list[Tick]:
    """Read a rosbag2 directory (mcap or sqlite3) into ticks: ``input=topic`` names which topic feeds which input.

    One tick per message, at its header stamp (else its log time), in seconds from the first; only the named topics
    are read, so a bag full of other traffic is fine. A named topic whose type robot_msgs cannot read raises.
    """
    from rosbags.rosbag2 import Reader
    from robot_msgs.ros import typestore

    store = typestore()
    by_topic = {topic: name for name, topic in topics.items()}
    rows: list[tuple[float, str, Any]] = []
    with Reader(path) as reader:
        connections = [c for c in reader.connections if c.topic in by_topic]
        missing = set(by_topic) - {c.topic for c in connections}
        if missing:
            have = sorted({c.topic for c in reader.connections})
            raise KeyError(f"topics not in {path}: {sorted(missing)}; it has {have}")
        for connection, log_ns, raw in reader.messages(connections=connections):
            msg = _from_ros(store.deserialize_cdr(raw, connection.msgtype))
            rows.append((_stamp(msg, log_ns), by_topic[connection.topic], msg))
    rows.sort(key=lambda row: row[0])
    t0 = rows[0][0] if rows else 0.0
    return [(t - t0, {name: msg}) for t, name, msg in rows]
