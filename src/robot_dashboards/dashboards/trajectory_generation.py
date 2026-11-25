"""Trajectory generation: did the generator turn these waypoints into a smooth trajectory inside the limits?

    import robot_dashboards as dash

    dash.show("trajectory_generation", waypoints=waypoints, trajectory=traj, tool=tcp)   # one plan
    dash.save("plan.rrd", "trajectory_generation", ticks=[(None, {"waypoints": waypoints, "trajectory": traj})])

- ``waypoints``: the tool poses the generator was asked to pass through (a tuple of `Waypoint`, or the
  `WaypointAction` it was given), in the tool path's frame (the URDF's root link unless ``tool`` says otherwise).
- ``trajectory``: what it made, a `JointTrajectory` whose ``limits`` (`JointLimits`) are the bounds it had to keep.
- ``tool`` (optional): the tool's `PoseTrajectory`; without it the tool is the URDF's last link, by forward kinematics.

What you see: robot_msgs' stock trajectory viewer (the URDF arm with start/end ghosts and the tool path on the left;
tabs q, qd, qdd, qddd, q / qd / qdd and TCP on the right), and what generation adds on top: the waypoints in the 3D
scene, numbered in order; a verdict (inside every limit? jerk bounded? through every waypoint?) with the worst case
of each check; lanes saying WHEN each check holds (green) or fails (red), and which waypoint the motion is heading
for; and each joint's peak |q|, |qd|, |qdd|, |qddd| as % of its own limit.

Entity layout: robot_msgs' (``/world/trajectory/...``, ``/positions/<joint>/trajectory``, ...), the waypoints at
``/world/waypoint_<k>/...`` (robot_msg_viz's Waypoint view), and ``/trajectory_generation/{verdict,checks,peak_pct}``.

The sample is a real plan: bam_ruckig (jerk-limited, time-optimal) on the UR5e through four tool poses, the
trajectory_tracking_agent dataset's task workspace-dev-029; the waypoints are that task's own keyposes
(``assets/trajectory_generation/workspace-dev-029-waypoints.json``), not reconstructed from the motion.

WHY the blueprint is sent again by `log`: the stock viewer's strips carry the trajectory's own numbers (travel, peak,
% of limit, the y range, the camera), so the fixed blueprint (built on the sample) would print the sample's numbers
over your data; `log` sends the same layout built on the trajectory it was given, as robot_msgs' own ``recording``
does.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz as viz
from robot_msgs.geometry import Point, Pose, Quaternion
from robot_msgs.geometry.rotations import quat_angle, quat_from_matrix, quat_slerp
from robot_msgs.manipulation import Waypoint, WaypointAction
from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory
from robot_msgs.trajectory.joint_limit_violation import joint_limit_violations
from robot_msgs.viz.rerun import TIMELINE, robot_blueprint

from robot_dashboards.assets import ASSETS, UR5E_TRAJECTORY, UR5E_URDF
from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._motion_state_helpers import forward_kinematics, inner, joint_label, tip_link

KEY = "trajectory_generation"
NAME = "trajectory"  # the entity name robot_msgs' viewer draws the trajectory under
URDF = UR5E_URDF  # the robot the trajectory is drawn on; set it to your URDF for another arm
WAYPOINTS = ASSETS / "trajectory_generation" / "workspace-dev-029-waypoints.json"
# WHY 1 mm and 1°: a waypoint is "passed" when the tool path comes this close to it (between samples, on the
# segment); the dashboard's tolerance, printed in the verdict, not a robot rating. The sample's misses are < 0.1 mm.
WAYPOINT_MM, WAYPOINT_DEG = 1.0, 1.0
DEG = 180.0 / np.pi

# pass / fail are the only green and red on the page (prior-art sweep: green/red ONLY for verdicts)
PASS, FAIL, UNCHECKED = (60, 160, 90), (214, 60, 60), (150, 155, 165)
# the bars: the stock viewer's card colours (q blue, qd teal, qdd grey) and a fourth, quiet violet for qddd
BARS = (("q", "position", (38, 110, 214)), ("qd", "velocity", (20, 150, 140)),
        ("qdd", "acceleration", (140, 140, 150)), ("qddd", "jerk", (150, 110, 190)))  # fmt: skip
TRACK = (215, 218, 224)  # the 100 % track behind each joint's bars


# ── the verdict: worked out once from the inputs, then drawn ─────────────────────────────────────────────


@dataclass
class _Check:
    name: str
    state: str  # "pass", "fail" or "unchecked"
    worst: str


@dataclass
class _Judgement:
    checks: list[_Check] = field(default_factory=list)
    peaks: dict[str, np.ndarray] = field(default_factory=dict)  # short -> (J,) peak % of limit (NaN = no limit)
    lanes: dict[str, tuple[np.ndarray, list[str], dict[str, tuple[int, int, int]]]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _percent(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """|value| on [low, high] -> [0, 100] % from the middle (robot_msgs' cards: 100 % is at a limit)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs((values - 0.5 * (low + high)) / (0.5 * (high - low))) * 100.0


def _bounds(limits: JointLimits | None, short: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    nan = np.full(count, np.nan)
    if limits is None:
        return nan, nan
    if short == "q":
        low = nan if limits.min_position is None else np.asarray(limits.min_position, dtype=np.float64)
        high = nan if limits.max_position is None else np.asarray(limits.max_position, dtype=np.float64)
        return low, high
    bound = {"qd": limits.max_velocity, "qdd": limits.max_acceleration, "qddd": limits.max_jerk}[short]
    if bound is None:
        return nan, nan
    bound = np.asarray(bound, dtype=np.float64)
    bound = np.where(np.isfinite(bound) & (bound > 0), bound, np.nan)
    return -bound, bound


def _tool_path(traj: JointTrajectory, tool: PoseTrajectory | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t from the trajectory's start, xyz, xyzw) of the tool: the given path, else the URDF tip by FK."""
    t0 = float(traj.t[0]) if len(traj) else 0.0
    if tool is not None:
        return (np.asarray(tool.t, dtype=np.float64) - t0, np.asarray(tool.positions, dtype=np.float64),
                np.asarray(tool.orientations, dtype=np.float64))  # fmt: skip
    pose = forward_kinematics(URDF, traj.joint_names, np.asarray(traj.positions))[tip_link(URDF)]
    return np.asarray(traj.t, dtype=np.float64) - t0, pose[:, :3, 3], quat_from_matrix(pose[:, :3, :3])


def _closest(
    t: np.ndarray, xyz: np.ndarray, quat: np.ndarray, target: Pose, start: int
) -> tuple[int, float, float, float]:
    """The point of the tool path (on its segments, from sample `start` on) nearest `target`: (segment, t, mm, deg)."""
    goal = target.position.as_array()
    goal_q = target.orientation.as_array()
    if len(xyz) == 1:
        return 0, float(t[0]), float(np.linalg.norm(xyz[0] - goal)) * 1e3, float(quat_angle(quat[0], goal_q)) * DEG
    a, b = xyz[start:-1], xyz[start + 1:]
    ab = b - a
    length = np.einsum("ij,ij->i", ab, ab)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.clip(np.where(length > 1e-18, np.einsum("ij,ij->i", goal - a, ab) / length, 0.0), 0.0, 1.0)
    miss = np.linalg.norm(a + s[:, None] * ab - goal, axis=1)
    k = int(np.argmin(miss))
    i = start + k
    q = quat_slerp(quat[i], quat[i + 1], s[k])
    when = float(t[i] + s[k] * (t[i + 1] - t[i]))
    return i, when, float(miss[k]) * 1e3, float(quat_angle(q, goal_q)) * DEG


def _quantity_text(short: str, joint_index: int, joint: str, percent: float, bound: float, when: float) -> str:
    unit = {"q": "°", "qd": "°/s", "qdd": "°/s²", "qddd": "°/s³"}[short]
    limit = "of the way to its limit" if short == "q" else f"of {bound * DEG:.0f} {unit}"
    return f"{percent:.1f} % {limit} · {joint_label(joint_index, joint)} at {when:.2f} s"


def _judge(waypoints: tuple[Waypoint, ...], traj: JointTrajectory, tool: PoseTrajectory | None,
           on_urdf: bool = True) -> _Judgement:
    out = _Judgement()
    seconds = np.asarray(traj.t, dtype=np.float64) - (float(traj.t[0]) if len(traj) else 0.0)
    full = traj
    if len(traj) >= 2 and (traj.velocities is None or traj.accelerations is None):
        full = traj.with_derivatives()
        out.notes.append("velocities / accelerations derived from the positions (the trajectory does not carry them)")
    limits = traj.limits
    count = traj.n_joints
    signals = {"q": np.asarray(full.positions, dtype=np.float64)}
    if full.velocities is not None:
        signals["qd"] = np.asarray(full.velocities, dtype=np.float64)
    if full.accelerations is not None:
        signals["qdd"] = np.asarray(full.accelerations, dtype=np.float64)
        # WHY np.gradient (first order at the ends): the rule `joint_limit_violations` judges jerk by, so the bar
        # and the verdict are one number
        two = len(full) >= 2
        signals["qddd"] = np.gradient(signals["qdd"], full.t, axis=0) if two else np.zeros_like(signals["qdd"])

    # 1. each joint's peak as % of its own limit, and when; the sample-by-sample "past a bound" for the lanes
    past = {short: np.zeros(len(seconds), dtype=bool) for short in signals}
    worst: dict[str, tuple[float, int, int]] = {}
    for short, values in signals.items():
        low, high = _bounds(limits, short, count)
        percent = _percent(values, low[None, :], high[None, :])
        out.peaks[short] = np.array([np.nan if np.all(np.isnan(column)) else np.nanmax(column) for column in percent.T])
        if np.any(np.isfinite(percent)):
            row, column = np.unravel_index(int(np.nanargmax(percent)), percent.shape)
            worst[short] = (float(percent[row, column]), int(column), int(row))
        with np.errstate(invalid="ignore"):
            past[short] = np.any(percent > 100.0 + 1e-9, axis=1)

    # 2. the checks: robot_msgs' own rule decides pass / fail; the worst case says by how much
    def check(name: str, shorts: tuple[str, ...], quantities: tuple[str, ...]) -> None:
        judged = [s for s in shorts if s in worst]
        if limits is None or not judged:
            missing = "no JointLimits on the trajectory" if limits is None else "no limit given"
            out.checks.append(_Check(name, "unchecked", missing))
            return
        found = joint_limit_violations(full, quantities=quantities)
        for short in judged:
            percent, column, row = worst[short]
            bound = _bounds(limits, short, count)[1][column]
            text = _quantity_text(short, column, full.joint_names[column], percent, bound, seconds[row])
            label = {"q": "position", "qd": "velocity", "qdd": "acceleration", "qddd": "jerk"}[short]
            state = "fail" if any(v.quantity == label for v in found) else "pass"
            out.checks.append(_Check(f"{name}: {label}" if len(shorts) > 1 else name, state, text))

    check("limits", ("q", "qd", "qdd"), ("position", "velocity", "acceleration"))
    check("smooth: jerk", ("qddd",), ("jerk",))
    if "qddd" not in signals:
        out.checks[-1] = _Check("smooth: jerk", "unchecked", "no accelerations to differentiate")

    # 3. the waypoints, in order: where the tool path comes nearest each one after the one before
    lane_t, lane_values, lane_colours = [0.0], [], {}
    if waypoints and tool is None and not on_urdf:
        out.checks.append(_Check("waypoints", "unchecked", "no tool path: pass tool=, or a URDF that has these joints"))
    elif waypoints:
        t, xyz, quat = _tool_path(traj, tool)
        if tool is None:
            out.notes.append(f"the tool path is the URDF's last link {tip_link(URDF)} by forward kinematics (no tool= "
                             "given); waypoints for a TCP beyond it need the tool's PoseTrajectory")
        start, misses = 0, []
        for k, waypoint in enumerate(waypoints):
            segment, when, mm, deg = _closest(t, xyz, quat, waypoint.pose, start)
            start = segment
            misses.append((mm, deg, k, when))
            ok = mm <= WAYPOINT_MM and deg <= WAYPOINT_DEG
            if k == 0:
                continue  # the lane names where the motion is HEADING; the start is where it begins
            value = f"to {k + 1} {waypoint.name} · {mm:.2f} mm, {deg:.2f}°"
            lane_values.append(value)
            lane_colours[value] = PASS if ok else FAIL
            lane_t.append(when)
        mm, deg, k, when = max(misses, key=lambda m: (m[0] / WAYPOINT_MM, m[1] / WAYPOINT_DEG))
        hit = sum(m[0] <= WAYPOINT_MM and m[1] <= WAYPOINT_DEG for m in misses)
        state = "pass" if hit == len(misses) else "fail"
        out.checks.append(_Check("waypoints", state, f"{hit} of {len(misses)} within {WAYPOINT_MM:g} mm and "
                                 f"{WAYPOINT_DEG:g}° · worst {mm:.2f} mm, {deg:.2f}° ({k + 1} {waypoints[k].name}, "
                                 f"{when:.2f} s)"))  # fmt: skip
        if len(seconds) and seconds[-1] > lane_t[-1] + 1e-6:
            lane_values.append("after the last waypoint")
            lane_colours["after the last waypoint"] = UNCHECKED
        else:
            lane_t.pop()
        if lane_values:
            out.lanes["waypoints"] = (np.array(lane_t[: len(lane_values)]), lane_values, lane_colours)
    else:
        out.checks.append(_Check("waypoints", "unchecked", "no waypoints given"))

    # 4. a lane per bound check: inside (green) or past (red), sample by sample
    for lane, shorts, inside, outside in (
        ("limits", ("q", "qd", "qdd"), "inside q, qd, qdd limits", "past a limit"),
        ("jerk", ("qddd",), "jerk inside its limit", "jerk past its limit"),
    ):
        judged = [s for s in shorts if s in worst]
        if limits is None or not judged or not len(seconds):
            continue
        bad = np.any([past[s] for s in judged], axis=0)
        changes = np.flatnonzero(np.r_[True, bad[1:] != bad[:-1]])
        values = [outside if bad[i] else inside for i in changes]
        out.lanes[lane] = (seconds[changes], values, {inside: PASS, outside: FAIL})
    return out


def _verdict_text(judgement: _Judgement, traj: JointTrajectory, waypoints: tuple[Waypoint, ...]) -> str:
    states = [c.state for c in judgement.checks]
    head = "FAIL" if "fail" in states else ("PASS" if all(s == "pass" for s in states) else "PASS (some unchecked)")
    seconds = float(traj.t[-1] - traj.t[0]) if len(traj) else 0.0
    steps = np.abs(np.diff(np.asarray(traj.positions), axis=0)) if len(traj) > 1 else np.zeros((0, traj.n_joints))
    moving = np.flatnonzero(np.any(steps > 1e-9, axis=1))
    still = f" (moves for {traj.t[moving[-1] + 1] - traj.t[0]:.2f} s, then holds)" if len(moving) and \
        traj.t[moving[-1] + 1] < traj.t[-1] - 1e-9 else ""  # fmt: skip
    lines = [
        f"## {head} · {traj.n_joints} joints, {len(traj)} samples, {seconds:.2f} s{still}, {len(waypoints)} waypoints",
        "",
        "| check | verdict | worst case |",
        "|---|---|---|",
    ]
    lines += [f"| {c.name} | **{c.state}** | {c.worst} |" for c in judgement.checks]
    lines += ["", "Limits are the trajectory's own JointLimits; % of limit is each joint's own (100 % = at it). "
              "The lanes (top right) say when: green = the check holds, red = it fails, sample by sample."]  # fmt: skip
    lines += [f"Note: {note}." for note in judgement.notes]
    return "\n".join(lines)


# ── drawing ──────────────────────────────────────────────────────────────────────────────────────────────


def _as_waypoints(waypoints: Any) -> tuple[Waypoint, ...]:
    if waypoints is None:
        return ()
    if isinstance(waypoints, WaypointAction):
        return tuple(waypoints.waypoints)
    if isinstance(waypoints, Waypoint):
        return (waypoints,)
    return tuple(waypoints)


# the bar grid, in its own 2D scene units: a row per joint, a column per quantity, each cell a 0-100 % track
_LEFT, _CELL, _BAR, _TOP, _ROW = 128.0, 160.0, 100.0, 52.0, 30.0
_INK = (200, 205, 215)  # labels: light on Rerun's dark canvas


def _grid_size(count: int, columns: int) -> tuple[float, float]:
    return _LEFT + columns * _CELL, _TOP + count * _ROW


def _log_bars(recording: Any, judgement: _Judgement, joints: tuple[str, ...]) -> None:
    """Each joint's peak |q|, |qd|, |qdd|, |qddd| as % of its own limit: a row per joint (J1 at the top), a column
    per quantity, the bar on a grey 0-100 % track with a tick at 100 % and its value beside it.

    WHY a 2D scene and not Rerun's BarChart: a BarChart has no per-bar labels (its x axis reads 0..n, not joint
    names), no reference line for the limit, and one colour per chart; here each bar says its joint, quantity and
    number. A bar past 100 % is drawn red to the tick (red = a failed check), its number in red beside it."""
    shown = [(short, word, colour) for short, word, colour in BARS if short in judgement.peaks]
    base = f"{KEY}/peak_pct"
    tracks, ticks, bars, bar_colours, anchors, labels, label_colours = [], [], [], [], [], [], []
    for column, (short, word, colour) in enumerate(shown):
        x0 = _LEFT + column * _CELL
        anchors.append([x0 + 0.5 * _BAR, 4.0])
        labels.append(f"{short} · {word}")
        label_colours.append(colour)
        for row, value in enumerate(judgement.peaks[short]):
            y = _TOP + row * _ROW
            tracks.append([[x0, y], [x0 + _BAR, y]])
            ticks.append([[x0 + _BAR, y - 8.0], [x0 + _BAR, y + 8.0]])
            if not np.isfinite(value):
                anchors.append([x0 + 0.5 * _BAR, y - 9.0])
                labels.append("no limit")
                label_colours.append(UNCHECKED)
                continue
            past = float(value) > _BAR
            # WHY the whole bar red and capped at the tick, not a red overrun: an overrun ran under its own number
            bars.append([[x0, y], [x0 + min(float(value), _BAR), y]])
            bar_colours.append(FAIL if past else colour)
            anchors.append([x0 + _BAR + 30.0, y - 9.0])
            labels.append(f"{value:.0f} %" if value < 99.5 or value >= 100.5 else f"{value:.1f} %")
            label_colours.append(FAIL if value > 100.0 else _INK)
    for row, joint in enumerate(joints):
        anchors.append([0.5 * _LEFT - 6.0, _TOP + row * _ROW - 9.0])
        labels.append(joint_label(row, joint))
        label_colours.append(_INK)
    thin = rr.Radius.ui_points(1.0)
    recording.log(f"{base}/tracks", rr.LineStrips2D(tracks, colors=[(110, 115, 125)], radii=thin, draw_order=1.0),
                  static=True)  # fmt: skip
    recording.log(f"{base}/limit_100", rr.LineStrips2D(ticks, colors=[(200, 205, 215)], radii=rr.Radius.ui_points(1.5),
                                                       draw_order=3.0), static=True)  # fmt: skip
    recording.log(f"{base}/bars", rr.LineStrips2D(bars, colors=bar_colours, radii=7.0, draw_order=2.0), static=True)
    recording.log(f"{base}/labels", rr.Points2D(anchors, labels=labels, colors=label_colours, radii=0.01,
                                                show_labels=True, draw_order=4.0), static=True)  # fmt: skip


def _log_lanes(recording: Any, judgement: _Judgement) -> None:
    for lane, (times, values, colours) in judgement.lanes.items():
        names = list(dict.fromkeys(values))
        packed = [((c[0] << 24) | (c[1] << 16) | (c[2] << 8) | 0xFF) for c in (colours[v] for v in names)]
        path = f"{KEY}/checks/{lane}"
        recording.log(path, rr.StateConfiguration(values=names, colors=np.array(packed, dtype=np.uint32)), static=True)
        recording.send_columns(path, indexes=[rr.TimeColumn(TIMELINE, duration=np.asarray(times, dtype=np.float64))],
                               columns=rr.StateChange.columns(state=values))  # fmt: skip


def log(recording: Any, *, waypoints: Any = None, trajectory: JointTrajectory | None = None,
        tool: PoseTrajectory | None = None) -> None:  # fmt: skip
    # WHY the inner stream: a trajectory carries its own clock (robot_msgs puts it on t from 0); lifting its styles
    # and meshes to a stream's "now" would hide them before that time
    recording = inner(recording)
    points = _as_waypoints(waypoints)
    for k, waypoint in enumerate(points):
        # WHY the number in the name: the scene labels each waypoint with its name; the number says the order
        viz.log(waypoint.replace(name=f"{k + 1} {waypoint.name}"), f"waypoint_{k + 1}", recording=recording)
    if trajectory is None:
        return
    robot = _robot_for(trajectory)
    if robot is not None:
        sample_points, sample_traj, sample_tool = _sample_inputs()
        # WHY not for the sample: the fixed blueprint IS this layout built on the sample, and every copy of the
        # stock page is 0.5 MB in the file (about 40 views)
        if not (trajectory == sample_traj and tool == sample_tool and len(points) == len(sample_points)):
            recording.send_blueprint(_layout(trajectory, tool, len(points)))  # see the module's WHY
        viz.log(trajectory, NAME, recording=recording, robot=robot, tool=tool, ghosts=True)
    else:
        recording.send_blueprint(_layout_without_robot(trajectory))
        viz.log(trajectory, NAME, recording=recording)
        if tool is not None:
            viz.log(tool, "tool", recording=recording)
    judgement = _judge(points, trajectory, tool, robot is not None)
    recording.log(f"{KEY}/verdict", rr.TextDocument(_verdict_text(judgement, trajectory, points),
                                                    media_type="text/markdown"), static=True)  # fmt: skip
    _log_bars(recording, judgement, trajectory.joint_names)
    _log_lanes(recording, judgement)


def _robot_for(trajectory: JointTrajectory) -> str | None:
    """The URDF when it knows every joint of the trajectory, else None (then the plots are drawn without an arm)."""
    from robot_dashboards.dashboards._motion_state_helpers import urdf_tree

    tree = urdf_tree(str(URDF))
    return str(URDF) if all(tree.get_joint_by_name(str(j)) is not None for j in trajectory.joint_names) else None


def _layout(traj: JointTrajectory, tool: PoseTrajectory | None, waypoints: int) -> rrb.Blueprint:
    """The stock viewer's page (the arm left, its tabs right) with the verdict and the % bars added.

    Left: the arm with the waypoints (stock 3D view, its eye), the verdict under it. Right: the check lanes and
    the bars on top, always visible (they answer the question); the stock tabs under them for the detail."""
    stock = robot_blueprint(traj, NAME, robot=URDF, tool=tool, ghosts=True)
    world, tabs = stock.root_container.contents
    world.contents = ["/world/**"]  # the arm, its tool path and the waypoints (robot_msg_viz logs them beside it)
    world.name = f"{world.name} · numbered: the {waypoints} waypoints" if waypoints else world.name
    verdict = rrb.TextDocumentView(origin=f"/{KEY}/verdict", name="Did it turn the waypoints into a smooth trajectory "
                                   "inside the limits?")  # fmt: skip
    lanes = rrb.StateTimelineView(origin=f"/{KEY}/checks", name="when each check holds (green) or fails (red)")
    # four columns whenever there are two samples: a missing qd / qdd is derived for the verdict (see _judge)
    width, height = _grid_size(traj.n_joints, 4 if len(traj) >= 2 else 1)
    bars = rrb.Spatial2DView(
        origin=f"/{KEY}/peak_pct", name="peak as % of each joint's own limit (100 % = at it)",
        visual_bounds=rrb.VisualBounds2D(x_range=[-4.0, width + 4.0], y_range=[-8.0, height + 4.0]),
        background=rrb.Background(kind="SolidColor", color=(22, 24, 28)),
    )  # fmt: skip
    # WHY closer than the stock eye: the stock sets it for half of a 1280 px page; here the arm has 3/5 of that half
    eye = world.properties["EyeControls3D"]
    look = np.asarray(eye.look_target.as_arrow_array().to_pylist()[0], dtype=np.float64)
    position = np.asarray(eye.position.as_arrow_array().to_pylist()[0], dtype=np.float64)
    world.properties["EyeControls3D"] = rrb.EyeControls3D(
        kind="Orbital", look_target=look.tolist(), eye_up=[0.0, 0.0, 1.0],
        position=(look + 0.8 * (position - look)).tolist(),
    )  # fmt: skip
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(world, verdict, row_shares=[3, 2]),
            rrb.Vertical(lanes, bars, tabs, row_shares=[1.35, 1.65, 2.7]),
            column_shares=[1, 1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        # WHY paused, as the stock viewer: it opens on the start pose and every still lands where it is asked
        rrb.TimePanel(state="collapsed", timeline=TIMELINE, play_state="paused", loop_mode="all"),
        collapse_panels=False,
    )


def _layout_without_robot(traj: JointTrajectory) -> rrb.Blueprint:
    """For a trajectory the URDF cannot draw: the same page with robot_msg_viz's joint plots in place of the arm's
    tabs, and the 3D scene holding only the waypoints and the tool path (when given)."""
    width, height = _grid_size(traj.n_joints, 4 if len(traj) >= 2 else 1)
    plots = [rrb.TimeSeriesView(origin=f"/{field}", name=f"{short} ({unit}) · {NAME}") for field, short, unit in
             (("positions", "q", "°"), ("velocities", "qd", "°/s"), ("accelerations", "qdd", "°/s²"))
             if getattr(traj, field) is not None]  # fmt: skip
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(origin="/world", name=f"waypoints and tool path · {URDF.name} has no joints "
                                  f"{list(traj.joint_names)[:2]}…, so no arm"),  # fmt: skip
                rrb.TextDocumentView(origin=f"/{KEY}/verdict", name="Did it turn the waypoints into a smooth "
                                     "trajectory inside the limits?"),  # fmt: skip
                row_shares=[3, 2],
            ),
            rrb.Vertical(
                rrb.StateTimelineView(origin=f"/{KEY}/checks", name="when each check holds (green) or fails (red)"),
                rrb.Spatial2DView(origin=f"/{KEY}/peak_pct", name="peak as % of each joint's own limit (100 % = at it)",
                                  visual_bounds=rrb.VisualBounds2D(x_range=[-4.0, width + 4.0],
                                                                   y_range=[-8.0, height + 4.0])),  # fmt: skip
                rrb.Tabs(*plots, active_tab=0),
                row_shares=[1.35, 1.65, 2.7],
            ),
            column_shares=[1, 1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline=TIMELINE, play_state="paused", loop_mode="all"),
        collapse_panels=False,
    )


# ── the sample: a real UR5e plan through four tool poses ─────────────────────────────────────────────────


@functools.cache
def _sample_inputs() -> tuple[tuple[Waypoint, ...], JointTrajectory, PoseTrajectory]:
    data = np.load(UR5E_TRAJECTORY)
    meta = json.loads(WAYPOINTS.read_text())
    names = tuple(str(n) for n in data["names"])
    limits = JointLimits(
        joint_names=names, min_position=data["lo"], max_position=data["hi"], max_velocity=data["vmax"],
        max_acceleration=data["amax"], max_jerk=np.full(len(names), float(meta["limits"]["max_jerk_rad_s3"])),
        position_kind=("software",) * len(names),
    )  # fmt: skip
    header = Header(meta={"source": "trajectory_tracking_agent workspace/v2 workspace-dev-029",
                          "time_law": "bam_ruckig"})
    traj = JointTrajectory(t=data["t"], joint_names=names, positions=data["q"], velocities=data["qd"],
                           accelerations=data["qdd"], limits=limits, header=header)  # fmt: skip
    matrices = np.tile(np.eye(4), (len(data["t"]), 1, 1))
    matrices[:, :3, :3], matrices[:, :3, 3] = data["tool_rot"], data["tool_xyz"]
    tool = PoseTrajectory.from_matrices(data["t"], matrices, child_frame_id="tcp")
    waypoints = tuple(
        Waypoint(name=w["name"],
                 pose=Pose(Point.from_array(w["position"]), Quaternion.from_array(w["orientation_xyzw"])))
        for w in meta["waypoints"]
    )
    return waypoints, traj, tool


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    waypoints, traj, tool = _sample_inputs()
    return [(None, {"waypoints": waypoints, "trajectory": traj, "tool": tool})]


@functools.cache
def _sample_layout() -> rrb.Blueprint:
    waypoints, traj, tool = _sample_inputs()
    return _layout(traj, tool, len(waypoints))


register(Dashboard(
    key=KEY,
    title="Trajectory generation",
    category="motion",
    summary="Did the generator turn these waypoints into a smooth trajectory inside the limits? The arm on its "
            "path through the numbered waypoints, a pass/fail verdict per check (limits, jerk, waypoints) with its "
            "worst case and when it holds, and each joint's peak as % of its limit.",
    consumes=(Waypoint, WaypointAction, JointTrajectory, PoseTrajectory),
    inputs={
        "waypoints": "the tool poses the generator was asked to pass through (Waypoints, or its WaypointAction)",
        "trajectory": "the JointTrajectory it made, with the JointLimits it had to keep",
        "tool": "optional: the tool's PoseTrajectory (default: the URDF's last link by forward kinematics)",
    },
    viewers={"rerun": RerunView(blueprint=lambda: _sample_layout(), log=log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        "# traj: a JointTrajectory with limits=JointLimits(...); waypoints: the Waypoints it was asked to pass\n"
        'dash.show("trajectory_generation", waypoints=waypoints, trajectory=traj, tool=tcp_path)'
    ),
))  # fmt: skip
