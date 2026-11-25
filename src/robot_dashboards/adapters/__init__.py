"""Adapters: anything in, robot_msgs out. A dashboard only ever sees robot_msgs messages.

    data you have ──adapter──▶ robot_msgs ──dashboard──▶ viewer (Rerun today; Foxglove, Viser… on the roadmap)

Two ways to bring your own data:

1. **Convert to robot_msgs yourself** and pass the messages (the `robot_msgs` adapter is the identity):
   ``dash.show("object_detection", image=Image(...), detections=Detection2DArray(...))``.
2. **Register an adapter** for a format you have many of, once:

       from robot_dashboards.adapters import Adapter, register
       register(Adapter("my_detector", "My detector's output",
                        convert=lambda out: Detection2DArray(...),         # one source message -> one robot_msgs message
                        accepts=lambda obj: isinstance(obj, MyResult)))

   then ``ticks(my_samples)`` (or ``ticks(..., adapter="my_detector")``) turns ``[(t, {input: source}), ...]`` into the
   dashboard's ticks.

Built in: ``robot_msgs`` (the identity) and ``ros2`` (rosbags/ROS 2 messages through robot_msgs.ros, plus
`ros2.bag_ticks` to feed a dashboard straight from an MCAP / rosbag2 recording).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from robot_msgs import Msg

Tick = tuple[float | None, dict[str, Any]]


@dataclass(frozen=True)
class Adapter:
    """One input format: how to turn one of its messages into one robot_msgs message.

    - ``convert(obj) -> Msg``: one source message to its robot_msgs value (raise if it has none).
    - ``accepts(obj) -> bool``: whether this adapter recognises ``obj`` (used when no adapter is named).
    """

    name: str
    summary: str
    convert: Callable[[Any], Msg]
    accepts: Callable[[Any], bool]


ADAPTERS: dict[str, Adapter] = {}


def register(adapter: Adapter) -> Adapter:
    """Add (or replace) an adapter; returns it."""
    ADAPTERS[adapter.name] = adapter
    return adapter


def get(name: str) -> Adapter:
    _load()
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(f"no adapter {name!r}; adapters: {sorted(ADAPTERS)}") from None


def adapter_for(obj: Any) -> Adapter:
    """The first registered adapter that accepts ``obj`` (robot_msgs first, so a message is never converted)."""
    _load()
    for adapter in ADAPTERS.values():
        if adapter.accepts(obj):
            return adapter
    raise TypeError(
        f"no adapter accepts a {type(obj).__name__}: convert it to a robot_msgs message, or register an Adapter "
        f"(robot_dashboards.adapters.register); adapters: {sorted(ADAPTERS)}"
    )


def to_robot_msgs(value: Any, adapter: str | None = None) -> Any:
    """One input value as robot_msgs: a message, or a list/tuple of them (kept a tuple)."""
    if isinstance(value, (list, tuple)):
        return tuple(to_robot_msgs(item, adapter) for item in value)
    chosen = get(adapter) if adapter is not None else adapter_for(value)
    return chosen.convert(value)


def ticks(samples: Iterable[Tick], adapter: str | None = None) -> list[Tick]:
    """``[(t, {input: source}), ...]`` -> the same ticks with every input converted to robot_msgs."""
    return [(t, {name: to_robot_msgs(value, adapter) for name, value in inputs.items()}) for t, inputs in samples]


_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    register(Adapter(
        "robot_msgs", "robot_msgs messages, passed through unchanged (convert your data to robot_msgs and use this)",
        convert=lambda obj: obj, accepts=lambda obj: isinstance(obj, Msg),
    ))  # fmt: skip
    from robot_dashboards.adapters import ros2

    register(ros2.ADAPTER)


__all__ = ["ADAPTERS", "Adapter", "adapter_for", "get", "register", "ticks", "to_robot_msgs"]
