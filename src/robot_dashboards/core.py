"""What a dashboard is, and show / save / stream for any of them.

A dashboard is a fixed Rerun layout for one robotics use case, fed robot_msgs messages:

    import robot_dashboards as dash

    dash.show("object_detection", image=frame, detections=dets)          # one moment, desktop viewer
    dash.save("det.rrd", "object_detection")                             # its sample data, blueprint inside
    with dash.stream("object_detection") as live:                        # live
        for t, frame, dets in camera:
            live.log(t=t, image=frame, detections=dets)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.blueprint as rrb
from robot_msg_viz import TIMELINE, Stream
from robot_msg_viz._engine import Timed

Tick = tuple[float | None, dict[str, Any]]  # (seconds on "t", or None for static data; the inputs at that time)


@dataclass(frozen=True)
class Dashboard:
    """One use case.

    - ``key``: ``"object_detection"``; also the file stem of its media and its gallery key.
    - ``category``: ``"perception"``, ``"manipulation"``, ``"motion"`` or ``"robot state"``.
    - ``consumes``: the robot_msgs types its inputs are.
    - ``inputs``: input name -> one-line meaning (``{"image": "the camera frame", ...}``).
    - ``blueprint``: the fixed layout.
    - ``log(rec, **inputs)``: draws one tick's inputs into ``rec`` (the caller has set the time).
      Inputs may be partial (a tick may carry only ``image``); draw what is given.
    - ``sample``: deterministic ticks that show the dashboard working (static ones have time None).
    """

    key: str
    title: str
    category: str
    summary: str
    consumes: tuple[type, ...]
    inputs: dict[str, str]
    blueprint: Callable[[], rrb.Blueprint]
    log: Callable[..., None]
    sample: Callable[[], list[Tick]]
    code: str = ""


DASHBOARDS: dict[str, Dashboard] = {}


def register(dashboard: Dashboard) -> Dashboard:
    DASHBOARDS[dashboard.key] = dashboard
    return dashboard


def get(key: str) -> Dashboard:
    _load()
    try:
        return DASHBOARDS[key]
    except KeyError:
        raise KeyError(f"no dashboard {key!r}; dashboards: {sorted(DASHBOARDS)}") from None


def all_dashboards() -> list[Dashboard]:
    _load()
    return list(DASHBOARDS.values())


def application_id(key: str) -> str:
    return f"robot_dashboards_{key}"


def _write(rec: rr.RecordingStream, dashboard: Dashboard, ticks: list[Tick]) -> None:
    timed = Timed(rec)
    for t, inputs in ticks:
        if t is None:
            dashboard.log(rec, **inputs)
        else:
            rec.set_time(TIMELINE, duration=float(t))
            dashboard.log(timed, **inputs)


def save(path: str | Path, key: str, ticks: list[Tick] | None = None) -> Path:
    """Write ``ticks`` (default: the dashboard's sample) to an .rrd with the dashboard's blueprint inside."""
    dashboard = get(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(application_id(key))
    rec.save(str(path), default_blueprint=dashboard.blueprint())
    _write(rec, dashboard, dashboard.sample() if ticks is None else ticks)
    rec.flush()
    rec.disconnect()
    return path


def show(key: str, ticks: list[Tick] | None = None, *, web: bool = False, **inputs: Any) -> rr.RecordingStream:
    """Open the dashboard: on ``inputs`` (one static moment), on ``ticks``, or on its sample data."""
    dashboard = get(key)
    rec = rr.RecordingStream(application_id(key))
    if inputs:
        ticks = [(None, inputs)]
    _write(rec, dashboard, dashboard.sample() if ticks is None else ticks)
    if web:
        url = rec.serve_grpc(default_blueprint=dashboard.blueprint())
        rr.serve_web_viewer(connect_to=url)
    else:
        rec.spawn(default_blueprint=dashboard.blueprint())
    return rec


class DashboardStream:
    """A live dashboard: ``log(t=..., **inputs)`` per tick; time defaults like robot_msg_viz.Stream."""

    def __init__(self, dashboard: Dashboard, **options: Any) -> None:
        self.dashboard = dashboard
        self._stream = Stream(application_id(dashboard.key), blueprint=dashboard.blueprint(), **options)

    @property
    def rec(self) -> rr.RecordingStream:
        return self._stream.rec

    def log(self, t: float | None = None, **inputs: Any) -> None:
        first = next(iter(inputs.values()), None)
        self._stream.rec.set_time(TIMELINE, duration=self._stream._time_of(first, t))
        self.dashboard.log(Timed(self._stream.rec), **inputs)

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> DashboardStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def stream(key: str, **options: Any) -> DashboardStream:
    """A live dashboard (``spawn=True`` desktop viewer, ``web=True`` web viewer, ``save=`` an .rrd too)."""
    return DashboardStream(get(key), **options)


_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from robot_dashboards import dashboards  # noqa: F401  (imports every dashboard module)
