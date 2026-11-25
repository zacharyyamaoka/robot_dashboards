"""What a dashboard is, and show / save / stream for any of them, in any supported viewer.

    your data ──adapter──▶ robot_msgs ──dashboard──▶ viewer (Rerun today; Foxglove, Viser… on the roadmap)

    import robot_dashboards as dash

    dash.show("object_detection", image=frame, detections=dets)          # one moment, desktop viewer
    dash.save("det.rrd", "object_detection")                             # its sample data, layout inside
    with dash.stream("object_detection") as live:                        # live
        for t, frame, dets in camera:
            live.log(t=t, image=frame, detections=dets)

Inputs that are not robot_msgs messages go through an adapter first (robot_dashboards.adapters): a ROS 2 message is
converted by the built-in ``ros2`` adapter, and your own format by an adapter you register.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_dashboards import adapters
from robot_dashboards import viewers as viewer_registry
from robot_dashboards.viewers import rerun as rerun_viewer
from robot_dashboards.viewers.rerun import RerunView

Tick = tuple[float | None, dict[str, Any]]  # (seconds on "t", or None for static data; the inputs at that time)


@dataclass(frozen=True)
class Dashboard:
    """One use case, independent of the viewer that draws it.

    - ``key``: ``"object_detection"``; also the file stem of its media and its gallery key.
    - ``category``: ``"perception"``, ``"manipulation"``, ``"motion"`` or ``"robot state"``.
    - ``summary``: the one question the dashboard answers.
    - ``consumes``: the robot_msgs types its inputs are.
    - ``inputs``: input name -> one-line meaning (``{"image": "the camera frame", ...}``).
    - ``sample``: deterministic ticks that show it working (static ones have time None).
    - ``viewers``: one renderer per viewer it supports, ``{"rerun": RerunView(blueprint=..., log=...)}``.
    """

    key: str
    title: str
    category: str
    summary: str
    consumes: tuple[type, ...]
    inputs: dict[str, str]
    sample: Callable[[], list[Tick]]
    viewers: dict[str, Any] = field(default_factory=dict)
    code: str = ""

    def renderer(self, viewer: str) -> Any:
        """This dashboard's renderer for ``viewer``, or why there is none."""
        viewer_registry.require(viewer)
        try:
            return self.viewers[viewer]
        except KeyError:
            raise NotImplementedError(f"{self.key} has no {viewer} renderer yet; it has {sorted(self.viewers)}") from None


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


def _ticks(dashboard: Dashboard, ticks: list[Tick] | None, inputs: dict[str, Any], adapter: str | None) -> list[Tick]:
    if inputs:
        ticks = [(None, inputs)]
    if ticks is None:
        return dashboard.sample()
    return adapters.ticks(ticks, adapter)


def save(
    path: str | Path, key: str, ticks: list[Tick] | None = None, *, viewer: str = "rerun", adapter: str | None = None
) -> Path:
    """Write ``ticks`` (default: the dashboard's sample) for ``viewer``, layout inside (Rerun: an .rrd)."""
    dashboard = get(key)
    renderer = dashboard.renderer(viewer)
    ticks = _ticks(dashboard, ticks, {}, adapter)
    if viewer == "rerun":
        return rerun_viewer.save(Path(path), key, renderer, ticks)
    raise NotImplementedError(viewer)  # pragma: no cover - require() stops roadmap viewers first


def show(
    key: str,
    ticks: list[Tick] | None = None,
    *,
    viewer: str = "rerun",
    web: bool = False,
    adapter: str | None = None,
    **inputs: Any,
) -> Any:
    """Open the dashboard: on ``inputs`` (one static moment), on ``ticks``, or on its sample data."""
    dashboard = get(key)
    renderer = dashboard.renderer(viewer)
    ticks = _ticks(dashboard, ticks, inputs, adapter)
    if viewer == "rerun":
        return rerun_viewer.show(key, renderer, ticks, web=web)
    raise NotImplementedError(viewer)  # pragma: no cover


class DashboardStream:
    """A live dashboard: ``log(t=..., **inputs)`` per tick; inputs go through the adapters like show()'s."""

    def __init__(self, dashboard: Dashboard, *, viewer: str = "rerun", adapter: str | None = None, **options: Any):
        self.dashboard = dashboard
        self._adapter = adapter
        self._live = rerun_viewer.LiveRerun(dashboard.key, dashboard.renderer(viewer), **options)

    @property
    def rec(self) -> Any:
        return self._live.rec

    def log(self, t: float | None = None, **inputs: Any) -> None:
        self._live.log(t=t, **{k: adapters.to_robot_msgs(v, self._adapter) for k, v in inputs.items()})

    def close(self) -> None:
        self._live.close()

    def __enter__(self) -> DashboardStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def stream(key: str, *, viewer: str = "rerun", adapter: str | None = None, **options: Any) -> DashboardStream:
    """A live dashboard (Rerun: ``spawn=True`` desktop viewer, ``web=True`` web viewer, ``save=`` an .rrd too)."""
    return DashboardStream(get(key), viewer=viewer, adapter=adapter, **options)


_LOADED = False


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from robot_dashboards import dashboards  # noqa: F401  (imports every dashboard module)


__all__ = ["DASHBOARDS", "Dashboard", "DashboardStream", "RerunView", "all_dashboards", "get", "register", "save",
           "show", "stream"]  # fmt: skip
