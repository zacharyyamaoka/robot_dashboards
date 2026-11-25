"""The contract every dashboard meets (docs/CONTRACT.md). Frozen: lanes never edit this file.

1. its sample inputs are robot_msgs messages of the types it declares, deterministic;
2. its gallery .rrd loads (`rerun rrd verify`), carries a default blueprint, and every view in that
   blueprint points at data actually in the file (read back by `rerun rrd print`);
3. streaming its sample through `stream()` lands each timed tick on the timeline;
4. robot_msg_viz's gallery lists it (the entry point is installed).
"""

from __future__ import annotations

import subprocess

import pytest
from robot_msg_viz.gallery import use_case_entries
from robot_msg_viz.ruler import covers, read, rerun_cli, views_in

import robot_dashboards as dash

DASHBOARDS = dash.all_dashboards()
IDS = [d.key for d in DASHBOARDS]


def test_there_are_dashboards():
    assert DASHBOARDS, "no dashboard registered"


def _messages(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _messages(item)
    else:
        yield value


@pytest.mark.parametrize("dashboard", DASHBOARDS, ids=IDS)
def test_sample_inputs_are_the_declared_robot_msgs_types(dashboard):
    assert dashboard.summary.strip() and dashboard.title.strip()
    assert dashboard.category in {"perception", "manipulation", "motion", "robot state"}
    ticks = dashboard.sample()
    assert ticks, "sample data"
    assert ticks == dashboard.sample(), "deterministic"
    times = [t for t, _ in ticks if t is not None]
    assert times == sorted(times)
    for _, inputs in ticks:
        assert set(inputs) <= set(dashboard.inputs), f"undeclared inputs {set(inputs) - set(dashboard.inputs)}"
        for value in inputs.values():
            for msg in _messages(value):
                assert isinstance(msg, dashboard.consumes), f"{type(msg).__name__} is not a declared robot_msgs type"


@pytest.mark.parametrize("dashboard", DASHBOARDS, ids=IDS)
def test_gallery_rrd_loads_with_a_blueprint_whose_views_have_data(dashboard, tmp_path):
    path = dash.save(tmp_path / f"{dashboard.key}.rrd", dashboard.key)
    assert path.stat().st_size > 0
    assert path.stat().st_size < 8_000_000, "the sample ships on the landing page"
    subprocess.run([*rerun_cli(), "rrd", "verify", str(path)], check=True, capture_output=True, timeout=120)
    contents = read(path)
    assert contents.data_paths
    assert contents.has_viewport and contents.default_blueprint
    views = views_in(dashboard.blueprint())
    assert len(views) >= 2, "a dashboard composes several views"
    for view in views:
        assert covers(contents, str(view.origin)), f"{type(view).__name__} at {view.origin} shows nothing"


@pytest.mark.parametrize("dashboard", DASHBOARDS, ids=IDS)
def test_streaming_lands_ticks_on_the_timeline(dashboard, tmp_path):
    timed = [(t, inputs) for t, inputs in dashboard.sample() if t is not None]
    if len(timed) < 2:
        pytest.skip("a static dashboard")
    path = tmp_path / "live.rrd"
    with dash.stream(dashboard.key, spawn=False, save=path) as live:
        for t, inputs in timed:
            live.log(t=t, **inputs)
    contents = read(path)
    assert contents.rows >= 2


def test_robot_msg_viz_gallery_lists_every_dashboard():
    keys = {e.key for e in use_case_entries() if e.source == "robot_dashboards"}
    assert keys == {d.key for d in DASHBOARDS}
