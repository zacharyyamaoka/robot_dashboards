"""The use cases as robot_msg_viz gallery entries (registered by the `robot_msg_viz.gallery` entry point)."""

from __future__ import annotations

from pathlib import Path

from robot_msg_viz.gallery import GalleryEntry
from robot_msg_viz.registry import key_of

from robot_dashboards import adapters, viewers
from robot_dashboards.core import all_dashboards, save


def entries() -> list[GalleryEntry]:
    adapters.get("robot_msgs")  # loads the built-in adapters
    out = []
    for dashboard in all_dashboards():
        code = dashboard.code or (
            "import robot_dashboards as dash\n\n"
            f'dash.show("{dashboard.key}", ' + ", ".join(f"{k}=..." for k in dashboard.inputs) + ")\n"
            f'with dash.stream("{dashboard.key}") as live:   # live\n'
            "    live.log(t=t, " + ", ".join(f"{k}={k}" for k in dashboard.inputs) + ")"
        )

        def write(path: Path, key: str = dashboard.key) -> Path:
            return save(path, key)

        timed = any(t is not None for t, _ in dashboard.sample())
        out.append(GalleryEntry(
            kind="use_case", key=dashboard.key, title=dashboard.title, group=dashboard.category,
            summary=dashboard.summary, supported=True, code=code, write=write,
            consumes=tuple(key_of(t) for t in dashboard.consumes), source="robot_dashboards", timed=timed,
            viewers=tuple((v.title, "supported" if v.status == "supported" and v.name in dashboard.viewers else "roadmap")
                          for v in viewers.VIEWERS.values()),
            adapters=tuple(sorted(adapters.ADAPTERS, key=lambda n: (n != "robot_msgs", n))),
        ))  # fmt: skip
    return out
