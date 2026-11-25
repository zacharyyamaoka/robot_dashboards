"""Viewers: where a dashboard is drawn. Rerun today; the others are on the roadmap (docs/ROADMAP.md).

    adapters ──▶ robot_msgs ──▶ dashboard (the question, its inputs, sample data) ──▶ viewer

A dashboard carries one renderer per viewer it supports (``Dashboard.viewers["rerun"]``); `show`, `save` and
`stream` take ``viewer=`` (default ``"rerun"``). Asking for a viewer on the roadmap says what it still needs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Viewer:
    name: str
    title: str
    status: str  # "supported" | "roadmap"
    summary: str
    plan: str = ""  # for a roadmap viewer: what it takes, from the 2026-11-24 prior-art sweep


VIEWERS: dict[str, Viewer] = {
    v.name: v
    for v in (
        Viewer(
            "rerun", "Rerun", "supported",
            "Open source; the layout (a blueprint) travels inside the .rrd; desktop, web and notebook viewers; no account.",
        ),
        Viewer(
            "foxglove", "Foxglove", "roadmap",
            "Write the ticks as an MCAP (robot_msgs.ros.write_bag already writes one Foxglove opens) plus a layout JSON.",
            "Per dashboard: a Foxglove layout (foxglove-sdk's foxglove.layouts, MIT) and a message mapping onto "
            "Foxglove schemas (ImageAnnotations for 2D boxes, SceneUpdate for 3D, JointStates for the URDF). "
            "Caveats found: a Basic seat cannot open local files, comparison mode and embedding are paid, and the 3D "
            "panel has no native vision_msgs; Lichtblick (the open-source fork) takes ?layoutUrl= and lichtblick://.",
        ),
        Viewer(
            "viser", "Viser", "roadmap",
            "Interactive Python 3D in the browser (the house 3D viewer; Isaac Lab moved to it from Rerun in 3.2).",
            "Per dashboard: a scene builder on viser's API; a static .viser recording embeds in the landing page.",
        ),
        Viewer(
            "html", "Static HTML report", "roadmap",
            "One self-contained page per run (plots + stills), for reports and PR comments.",
            "Per dashboard: matplotlib figures from robot_msgs' .plot() plus the dashboard's derived numbers.",
        ),
    )
}


def get(name: str) -> Viewer:
    try:
        return VIEWERS[name]
    except KeyError:
        raise KeyError(f"no viewer {name!r}; viewers: {sorted(VIEWERS)}") from None


def require(name: str) -> Viewer:
    """The viewer, or an error saying it is on the roadmap and what it needs."""
    viewer = get(name)
    if viewer.status != "supported":
        raise NotImplementedError(f"the {viewer.title} viewer is on the roadmap (docs/ROADMAP.md): {viewer.plan}")
    return viewer


__all__ = ["VIEWERS", "Viewer", "get", "require"]
