"""The Rerun viewer: a dashboard's Rerun renderer, and show / save / stream for it.

A dashboard supports Rerun by carrying ``viewers={"rerun": RerunView(blueprint=..., log=...)}``:

- ``blueprint()``: the fixed layout (a ``rerun.blueprint.Blueprint``);
- ``log(rec, **inputs)``: draws one tick's robot_msgs inputs into ``rec`` (the caller has set the time; inputs may be
  partial). Draw the types through robot_msg_viz (``robot_msg_viz.log``) so a type looks the same everywhere.
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

Tick = tuple[float | None, dict[str, Any]]


@dataclass(frozen=True)
class RerunView:
    blueprint: Callable[[], rrb.Blueprint]
    log: Callable[..., None]


def application_id(key: str) -> str:
    return f"robot_dashboards_{key}"


def _write(rec: rr.RecordingStream, view: RerunView, ticks: list[Tick]) -> None:
    # WHY one Timed for the whole recording: views keep per-recording state keyed on the stream object (class colours,
    # the origin triad); a fresh wrapper per tick reset it, so class colours could change between frames.
    timed = Timed(rec)
    for t, inputs in ticks:
        if t is None:
            view.log(rec, **inputs)
        else:
            rec.set_time(TIMELINE, duration=float(t))
            view.log(timed, **inputs)


def save(path: Path, key: str, view: RerunView, ticks: list[Tick]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(application_id(key))
    rec.save(str(path))
    # WHY send_blueprint and not save(default_blueprint=) as well: either alone embeds it, but default_blueprint= is
    # make_active=false (a remembered layout wins) and both together store it twice (+0.5 MB for the stock robot page).
    rec.send_blueprint(view.blueprint(), make_active=True, make_default=True)
    _write(rec, view, ticks)
    rec.flush()
    rec.disconnect()
    return path


def show(key: str, view: RerunView, ticks: list[Tick], *, web: bool = False) -> rr.RecordingStream:
    rec = rr.RecordingStream(application_id(key))
    _write(rec, view, ticks)
    if web:
        url = rec.serve_grpc(default_blueprint=view.blueprint())
        rr.serve_web_viewer(connect_to=url)
    else:
        rec.spawn(default_blueprint=view.blueprint())
    return rec


class LiveRerun:
    """A live dashboard in Rerun: ``log(t=..., **inputs)`` per tick; time defaults like robot_msg_viz.Stream."""

    def __init__(self, key: str, view: RerunView, **options: Any) -> None:
        self.view = view
        self._stream = Stream(application_id(key), blueprint=view.blueprint(), **options)
        self._timed = Timed(self._stream.rec)  # WHY one: see _write

    @property
    def rec(self) -> rr.RecordingStream:
        return self._stream.rec

    def log(self, t: float | None = None, **inputs: Any) -> None:
        first = next(iter(inputs.values()), None)
        self._stream.rec.set_time(TIMELINE, duration=self._stream._time_of(first, t))
        self.view.log(self._timed, **inputs)

    def close(self) -> None:
        self._stream.close()
