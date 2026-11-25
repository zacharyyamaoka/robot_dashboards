"""Pick and place: where is the arm in the pick-and-place cycle, and did each phase go as planned?

Inputs: the arm's JointTrajectory (a UR5e), its tool's PoseTrajectory (optional; without it the flange is the
tool, by FK), the waypoints (a WaypointAction, or a tuple of Waypoint) and the planner's GraspPose (optional).

- Left, the 3D cell: robot_msgs' stock URDF arm on the trajectory (`robot_msgs.viz.rerun.log(traj, robot=...)`,
  the tool path solid up to the playhead), robot_msg_viz's waypoints numbered in order and its grasp glyph, plus
  the jaws riding on the tool at the commanded width, labelled with the phase.
- Right, three state lanes over time: the phase, the commanded gripper and "on plan?" (green / red: pass / fail);
  under them tabs: each phase against its plan (planned and actual time, how close the tool came to the
  waypoint, and which joint was nearest its limit when a phase ran long), the joints as % of their limits (the
  stock cards) and the TCP speed and position.

A phase ends when the tool leaves its waypoint (see `_manipulation_helpers.phases`), so a waypoint's dwell and a
close's hold count in its phase; it is on plan when the tool came within 2 mm and 2° of the waypoint (a blended
one within its blend radius) and took its waypoint's `duration` (+ `dwell`) within 10 % or 50 ms.

The sample is a cycle planned here on the shipped UR5e URDF (IK at each waypoint, quintic rest-to-rest legs,
straight tool lines for the short moves), every leg stretched until the joints keep the URDF's 180°/s and the
shipped trajectory's 6 rad/s² (`assets/data/workspace-dev-029.npz` `amax`). Its carry budget (1.0 s) is tighter
than the shoulder can swing 112° at that acceleration, so the carry runs long and the dashboard says so.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz as viz
from robot_msg_viz import TIMELINE
from robot_msg_viz.views.manipulation import END, FINGER_DEPTH, INK, START, ramp
from robot_msgs.geometry import Pose
from robot_msgs.geometry.rotations import quat_from_matrix, quat_to_matrix
from robot_msgs.manipulation import GraspPose, Waypoint, WaypointAction
from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory
from robot_msgs.viz import rerun as stock

from robot_dashboards.assets import UR5E_TRAJECTORY, UR5E_URDF
from robot_dashboards.core import Dashboard, register
from robot_dashboards.dashboards._manipulation_helpers import DEG, Arm, Phase, phases, plan_cycle

KEY = "pick_and_place"
ARM = "arm"  # the trajectory's entity name: /world/arm/robot/..., /limit_pct/<joint>/arm, /tcp/.../arm
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
          "wrist_3_joint")  # fmt: skip
TOOL_LENGTH = 0.145  # m, flange to TCP: the shipped trajectory's tool (its tool_xyz sits 145 mm along the flange z)
PASS, FAIL, NEUTRAL = (60, 170, 90), (220, 60, 60), (150, 155, 165)
OPEN_JAWS, MOVING_JAWS, CLOSED_JAWS = (175, 180, 190), (120, 160, 210), INK
_ROWS = 400  # moving-gripper samples at most (the stock tool path's own cap), so a long cycle stays small


def short(joint: str) -> str:
    """`shoulder_pan_joint` → `J1 shoulder_pan` for the UR5e's joints (Zach: URDF names plus J1…J6)."""
    name = joint[: -len("_joint")] if joint.endswith("_joint") else joint
    return f"J{JOINTS.index(joint) + 1} {name}" if joint in JOINTS else name


# ── inputs → what the views need ─────────────────────────────────────────────


def as_action(waypoints: WaypointAction | tuple[Waypoint, ...] | None) -> WaypointAction | None:
    if waypoints is None or isinstance(waypoints, WaypointAction):
        return waypoints
    return WaypointAction(waypoints=tuple(waypoints))


@functools.cache
def _arm(joints: tuple[str, ...]) -> Arm:
    return Arm(UR5E_URDF, joints)


def tool_path(trajectory: JointTrajectory, tool: PoseTrajectory | None) -> tuple[np.ndarray, np.ndarray]:
    """(times from the trajectory's start, (N, 4, 4) tool poses in the arm's root frame): the given tool path, else
    the flange by FK (then the waypoints must be flange poses)."""
    t0 = float(trajectory.t[0])
    if tool is not None:
        poses = np.broadcast_to(np.eye(4), (len(tool), 4, 4)).copy()
        poses[:, :3, :3] = quat_to_matrix(np.asarray(tool.orientations, dtype=np.float64))
        poses[:, :3, 3] = np.asarray(tool.positions, dtype=np.float64)
        return np.asarray(tool.t, dtype=np.float64) - t0, poses
    arm = _arm(tuple(str(j) for j in trajectory.joint_names))
    return np.asarray(trajectory.t, dtype=np.float64) - t0, arm.fk(np.asarray(trajectory.positions))


def limits(trajectory: JointTrajectory) -> tuple[np.ndarray, np.ndarray | None]:
    """Each joint's speed limit (the message's, else the URDF's) and acceleration limit (the message's, or none)."""
    arm = _arm(tuple(str(j) for j in trajectory.joint_names))
    given = trajectory.limits
    velocity = arm.max_velocity if given is None or given.max_velocity is None else np.asarray(given.max_velocity)
    acceleration = None if given is None or given.max_acceleration is None else np.asarray(given.max_acceleration)
    return velocity, acceleration


def cycle_phases(trajectory: JointTrajectory, tool: PoseTrajectory | None, action: WaypointAction) -> list[Phase]:
    t, poses = tool_path(trajectory, tool)
    velocity, acceleration = limits(trajectory)
    joint_t = np.asarray(trajectory.t, dtype=np.float64) - float(trajectory.t[0])
    return phases(t, poses, action.waypoints, joint_t=joint_t, qd=trajectory.velocities,
                  qdd=trajectory.accelerations, joints=tuple(str(j) for j in trajectory.joint_names),
                  max_velocity=velocity, max_acceleration=acceleration)  # fmt: skip


def colour(index: int, count: int) -> tuple[int, int, int]:
    """Waypoint ``index``'s colour: robot_msg_viz's blue → orange ramp, so a phase's lane matches its 3D label."""
    return ramp((START, END), index / max(count - 1, 1))


def rate_text(phase: Phase) -> str:
    """Why a phase ran long, when a joint was at its limit there: `J1 shoulder_pan at 99 % of its 344°/s² limit`."""
    if phase.peak is None or phase.peak[1] < 0.9:
        return ""
    joint, fraction, what, limit = phase.peak
    unit = "°/s" if what == "speed" else "°/s²"
    return f"{short(joint)} at {fraction * 100:.0f} % of its {limit * DEG:.0f}{unit} {what} limit"


def verdict(phase: Phase) -> str:
    """One cell: ✅, or ❌ and what was off."""
    problems = []
    if not phase.reached:
        problems.append(f"missed the waypoint by {phase.miss_mm:.1f} mm, {phase.miss_deg:.1f}°")
    if phase.on_time is False:
        late = phase.actual - phase.planned
        text = f"{'late' if late > 0 else 'early'} {late:+.2f} s"
        reason = rate_text(phase) if late > 0 else ""
        problems.append(text + (f": {reason}" if reason else ""))
    return "✅" if not problems else "❌ " + "; ".join(problems)


def gripper_states(action: WaypointAction, found: list[Phase]) -> list[tuple[float, str, float | None, float | None]]:
    """The commanded gripper as (from time, lane state, width at that time, width at the phase's end): a phase whose
    waypoint sets a new width moves the jaws over the phase (a waypoint's width is the gap to have reached there)."""
    width: float | None = None
    out: list[tuple[float, str, float | None, float | None]] = [(0.0, "not commanded yet", None, None)]
    for phase in found:
        target = action.waypoints[phase.index].gripper_width
        if target is None or (width is not None and abs(target - width) < 1e-9):
            continue
        closing = width is not None and target < width
        word = "closing to" if closing else "opening to"
        out.append((phase.start, f"{word} {target * 1000:.0f} mm", width, target))
        out.append((phase.end, f"{'closed' if closing else 'open'} {target * 1000:.0f} mm", target, target))
        width = target
    return out


def width_at(states: list[tuple[float, str, float | None, float | None]], times: np.ndarray,
             ends: list[float]) -> np.ndarray:  # fmt: skip
    """The commanded jaw gap at each time (NaN before the first command), linear across each moving phase."""
    out = np.full(len(times), np.nan)
    for k, (start, _, before, after) in enumerate(states):
        if after is None:
            continue
        stop = ends[k]
        inside = times >= start - 1e-9
        if before is None or stop <= start:
            out[inside] = after
        else:
            fraction = np.clip((times[inside] - start) / (stop - start), 0.0, 1.0)
            out[inside] = before + fraction * (after - before)
    return out


def report(action: WaypointAction, found: list[Phase], grasp: GraspPose | None, now: int | None, at: float) -> str:
    """Markdown: the cycle against its plan, one row per phase, ``now`` (a phase's position) marked; ``None`` = done."""
    planned = [p.planned for p in found]
    total = found[-1].end if found else 0.0
    good = sum(p.as_planned for p in found)
    head = ("**Cycle done**" if now is None else f"**Now: {found[now].index} {found[now].name}**") + f" · t = {at:.2f} s"
    plan = f", plan {sum(planned):.2f} s" if all(v is not None for v in planned) else ""
    lines = [head, "", f"**{good} of {len(found)} phases on plan** · cycle {total:.2f} s{plan}", ""]
    late = [p for p in found if not p.as_planned]
    if late:
        lines += ["Not on plan: " + "; ".join(f"**{p.index} {p.name}** {verdict(p)[2:]}" for p in late), ""]
    if grasp is not None:
        lines += [grasp_line(action, grasp), ""]
    lines += ["| # | phase | plan | took | tool at waypoint | on plan? |", "|---:|---|---:|---:|---:|---|"]
    for k, p in enumerate(found):
        mark = "▶ " if k == now else ""
        name = f"**{mark}{p.name}**" if k == now else p.name
        plan_text = "–" if p.planned is None else f"{p.planned:.2f} s"
        lines.append(f"| {p.index} | {name} | {plan_text} | {p.actual:.2f} s | "
                     f"{p.miss_mm:.1f} mm · {p.miss_deg:.1f}° | {verdict(p).split(':')[0]} |")  # fmt: skip
    lines.append("")
    lines.append("*A phase runs until the tool leaves its waypoint; on plan = within 2 mm and 2° of the waypoint, "
                 "and its duration (+ dwell) within 10 % or 50 ms.*")  # fmt: skip
    return "\n".join(lines)


def grasp_line(action: WaypointAction, grasp: GraspPose) -> str:
    """Does the waypoint that closes the jaws sit on the planner's grasp, at its width?"""
    closes = action.labelled("close") or tuple(w for w in action.waypoints if w.gripper_width is not None)
    if not closes:
        return "**Grasp:** no waypoint closes the jaws ❌"
    close = min(closes, key=lambda w: np.linalg.norm(w.pose.position.as_array() - grasp.pose.position.as_array()))
    a, b = close.pose.to_matrix(), grasp.pose.to_matrix()
    offset = float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1000.0
    cos = float(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    turn = math.degrees(math.acos(cos))
    # WHY 180° counts as the same grasp: a parallel gripper's two fingers swap, nothing else changes
    turn = min(turn, abs(180.0 - turn)) if abs(a[:3, 2] @ b[:3, 2] - 1.0) < 1e-6 else turn
    width = close.gripper_width
    same_width = width is not None and abs(width - grasp.width) < 0.0005
    ok = offset <= 2.0 and turn <= 2.0 and same_width
    jaw = "–" if width is None else f"{width * 1000:.0f}"
    return (f"**Grasp:** `{close.name}` is {offset:.1f} mm and {turn:.1f}° from the planner's "
            f"{grasp.object_id or 'grasp'} grasp (score {grasp.score:.2f}), jaws {jaw} mm for its "
            f"{grasp.width * 1000:.0f} mm {'✅' if ok else '❌'}")  # fmt: skip


# ── log ──────────────────────────────────────────────────────────────────────


def log(rec: rr.RecordingStream, trajectory: JointTrajectory | None = None, tool: PoseTrajectory | None = None,
        waypoints: WaypointAction | tuple[Waypoint, ...] | None = None, grasp: GraspPose | None = None) -> None:  # fmt: skip
    action = as_action(waypoints)
    if trajectory is not None:
        stock.log(trajectory, ARM, recording=rec, robot=UR5E_URDF, tool=tool, ghosts=False)
        # WHY: the stock logger remembers every path per stream for its own auto-layout (robot_msg_viz's engine
        # drops them the same way); this dashboard has a fixed layout, so a long stream must not grow that list
        getattr(stock, "_PATHS", {}).pop(id(rec), None)
        getattr(stock, "_PATHS", {}).pop(id(getattr(rec, "_inner", None)), None)
    if action is not None:
        viz.log(action, "waypoints", recording=rec)
    if grasp is not None:
        viz.log(grasp, "grasp", recording=rec)
    if trajectory is not None and action is not None and len(action.waypoints) > 1:
        log_cycle(rec, trajectory, tool, action, grasp)


def log_cycle(rec: rr.RecordingStream, trajectory: JointTrajectory, tool: PoseTrajectory | None,
              action: WaypointAction, grasp: GraspPose | None) -> None:  # fmt: skip
    found = cycle_phases(trajectory, tool, action)
    count = len(action.waypoints)
    starts = [p.start for p in found]
    end = found[-1].end

    # the lanes: phase, commanded gripper, on plan? (a StateChange per change; the configuration names and colours)
    phase_states = [f"{p.index} {p.name}" for p in found] + ["cycle done"]
    rec.log(f"{KEY}/lanes/cycle_phase", rr.StateConfiguration(
        values=phase_states, colors=[colour(p.index, count) for p in found] + [NEUTRAL]), static=True)  # fmt: skip
    rec.send_columns(f"{KEY}/lanes/cycle_phase", indexes=[rr.TimeColumn(TIMELINE, duration=starts + [end])],
                     columns=rr.StateChange.columns(state=phase_states))  # fmt: skip
    checks = [f"{p.index}" for p in found] + ["done"]
    labels = ["on plan" if p.as_planned else verdict(p)[2:].split(":")[0] for p in found] + [""]
    rec.log(f"{KEY}/lanes/on_plan", rr.StateConfiguration(
        values=checks, labels=labels, colors=[PASS if p.as_planned else FAIL for p in found] + [NEUTRAL]),
        static=True)  # fmt: skip
    rec.send_columns(f"{KEY}/lanes/on_plan", indexes=[rr.TimeColumn(TIMELINE, duration=starts + [end])],
                     columns=rr.StateChange.columns(state=checks))  # fmt: skip
    gripper = gripper_states(action, found)
    gripper_colours = {"not": NEUTRAL, "open": OPEN_JAWS, "closed": CLOSED_JAWS}
    values = list(dict.fromkeys(state for _, state, _, _ in gripper))
    rec.log(f"{KEY}/lanes/gripper", rr.StateConfiguration(
        values=values, colors=[gripper_colours.get(v.split()[0], MOVING_JAWS) for v in values]), static=True)  # fmt: skip
    rec.send_columns(f"{KEY}/lanes/gripper", indexes=[rr.TimeColumn(TIMELINE, duration=[s[0] for s in gripper])],
                     columns=rr.StateChange.columns(state=[s[1] for s in gripper]))  # fmt: skip

    # the report: one document per phase start (latest-at holds it), and the finished cycle at the end
    docs = [report(action, found, grasp, k, p.start) for k, p in enumerate(found)] + \
        [report(action, found, grasp, None, end)]  # fmt: skip
    rec.send_columns(f"{KEY}/report", indexes=[rr.TimeColumn(TIMELINE, duration=starts + [end])],
                     columns=rr.TextDocument.columns(text=docs, media_type=["text/markdown"] * len(docs)))  # fmt: skip

    # the jaws on the tool at the commanded width, labelled with the phase (the arm's model ends at the flange)
    t, poses = tool_path(trajectory, tool)
    rows = np.unique(np.linspace(0, len(t) - 1, min(len(t), _ROWS)).round().astype(int))
    ends = [found[-1].end] * len(gripper)
    for k, (start, _, _, _) in enumerate(gripper):
        later = [p.end for p in found if p.start >= start - 1e-9]
        ends[k] = later[0] if later else end
    widths = width_at(gripper, t[rows], ends)
    on_t = [rr.TimeColumn(TIMELINE, duration=t[rows])]
    rec.send_columns("world/gripper", indexes=on_t, columns=rr.Transform3D.columns(
        translation=poses[rows, :3, 3], quaternion=quat_from_matrix(poses[rows, :3, :3])))  # fmt: skip
    known = np.where(np.isnan(widths), 0.0, widths)
    # WHY robot_msg_viz's glyph (fingers along x, tips at the tool point, palm FINGER_DEPTH back): the jaws read the
    # same as its grasp and waypoint views; drawn per row here because their width moves
    d = FINGER_DEPTH
    strips = [np.array([[-w / 2, 0, 0], [-w / 2, 0, -d], [w / 2, 0, -d], [w / 2, 0, 0]]) for w in known]
    rec.log("world/gripper/jaws", rr.LineStrips3D.from_fields(radii=[0.0035], colors=[(235, 238, 245)]), static=True)
    rec.send_columns("world/gripper/jaws", indexes=on_t, columns=rr.LineStrips3D.columns(strips=strips))
    rec.log("world/gripper/phase", rr.Points3D.from_fields(positions=[[0.0, 0.0, -0.075]], radii=[0.004],
                                                           colors=[(235, 238, 245)]), static=True)  # fmt: skip
    rec.send_columns("world/gripper/phase", indexes=[rr.TimeColumn(TIMELINE, duration=starts + [end])],
                     columns=rr.Points3D.columns(labels=[f"▶ {s}" for s in phase_states]))  # fmt: skip


# ── blueprint ────────────────────────────────────────────────────────────────


def blueprint() -> rrb.Blueprint:
    world = rrb.Spatial3DView(
        origin="/world", name="UR5e cycle · waypoints numbered in order · jaws on the tool = the commanded gripper",
        eye_controls=rrb.EyeControls3D(kind="Orbital", position=[1.2, -0.72, 0.8], look_target=[0.18, 0.08, 0.22],
                                       eye_up=[0.0, 0.0, 1.0]),
    )  # fmt: skip
    lanes = rrb.StateTimelineView(origin=f"/{KEY}/lanes", name="phase · gripper (commanded) · on plan?",
                                  time_view=rrb.TimeAxis(link="LinkToGlobal"))  # fmt: skip
    phases_tab = rrb.TextDocumentView(origin=f"/{KEY}/report", name="each phase against its plan")
    cards = rrb.Grid(*(
        rrb.TimeSeriesView(origin=f"/limit_pct/{joint}", name=f"{short(joint)} · % of limit",
                           axis_y=rrb.ScalarAxis(range=(-125.0, 125.0), zoom_lock=False),
                           plot_legend=rrb.PlotLegend(visible=False), axis_x=rrb.TimeAxis(link="LinkToGlobal"))
        for joint in JOINTS
    ), grid_columns=2, name="joints % of limit (q blue · qd teal · qdd grey)")  # fmt: skip
    tcp = rrb.Vertical(
        rrb.TimeSeriesView(origin="/tcp/speed_mm_s", name="TCP speed [mm/s] · zero at every stop",
                           plot_legend=rrb.PlotLegend(visible=False), axis_x=rrb.TimeAxis(link="LinkToGlobal")),
        rrb.TimeSeriesView(origin="/tcp/position_mm", name="TCP position [mm] · x / y / z",
                           plot_legend=rrb.PlotLegend(visible=True, corner="RightBottom"),
                           axis_x=rrb.TimeAxis(link="LinkToGlobal")),
        name="TCP",
    )  # fmt: skip
    right = rrb.Vertical(lanes, rrb.Tabs(phases_tab, cards, tcp, active_tab=0), row_shares=[1.0, 2.6])
    return rrb.Blueprint(
        rrb.Horizontal(world, right, column_shares=[1.1, 1.0]),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        # WHY collapsed: the lanes are the cycle's timeline already; the bar keeps play and the scrubber
        rrb.TimePanel(state="collapsed", timeline=TIMELINE, play_state="paused", loop_mode="all"),
        collapse_panels=False,
    )


# ── sample: a UR5e pick-and-place cycle planned on the shipped URDF ──────────


def _at(x: float, y: float, z: float, yaw_deg: float) -> Pose:
    """The tool pointing straight down at (x, y, z), turned ``yaw_deg`` about z (a parallel gripper's yaw and
    yaw ± 180° are one grasp; these keep the wrist from spinning)."""
    return Pose.from_xyz_rpy(x, y, z, math.pi, 0.0, math.radians(yaw_deg))


@functools.cache
def _sample() -> tuple[JointTrajectory, PoseTrajectory, WaypointAction, GraspPose]:
    arm = Arm(UR5E_URDF, JOINTS, tool=TOOL_LENGTH)
    q_ready = np.radians([10.0, -105.0, 100.0, -85.0, -90.0, 0.0])  # elbow up, wrist down, clear of the table
    ready = Pose.from_matrix(arm.fk(q_ready))
    pick, place = (0.42, -0.26, 0.025, -160.0), (0.08, 0.48, 0.030, -90.0)  # a 50 mm block's centre; the shelf drop
    above_pick, above_place = _at(*pick[:2], 0.20, pick[3]), _at(*place[:2], 0.22, place[3])
    grasp = GraspPose(pose=_at(*pick), width=0.048, pre_grasp_width=0.080, score=0.91, object_id="block",
                      header=Header(frame_id=arm.root))  # fmt: skip
    action = WaypointAction(
        tool_frame="tcp", arm_joints=JOINTS, header=Header(frame_id=arm.root),
        waypoints=(
            Waypoint(name="start", label="start", pose=ready),
            Waypoint(name="approach", label="approach", pose=above_pick, gripper_width=0.080, duration=1.2),
            Waypoint(name="pre_grasp", label="pre_grasp", pose=_at(*pick[:2], 0.08, pick[3]), duration=0.6),
            Waypoint(name="grasp", label="grasp", pose=grasp.pose, duration=0.4),
            Waypoint(name="close", label="close", pose=grasp.pose, gripper_width=0.048, gripper_max_effort=20.0,
                     duration=0.4),
            Waypoint(name="lift", label="lift", pose=above_pick, duration=0.8),
            Waypoint(name="carry", label="carry", pose=above_place, duration=1.0),
            Waypoint(name="place", label="place", pose=_at(*place), duration=0.8),
            Waypoint(name="release", label="release", pose=_at(*place), gripper_width=0.080, duration=0.3),
            Waypoint(name="retreat", label="retreat", pose=above_place, duration=0.8),
            Waypoint(name="ready", label="ready", pose=ready, duration=1.2),
        ),
    )  # fmt: skip
    accel = np.asarray(np.load(UR5E_TRAJECTORY)["amax"], dtype=np.float64)  # 6 rad/s² each
    cycle = plan_cycle(arm, action.waypoints, q_ready, rate=100.0, max_acceleration=accel)
    meta = {"source": "robot_dashboards pick_and_place sample: IK on assets/ur5e (ur_description), quintic legs "
                      "stretched to the URDF speed limits and workspace-dev-029's amax"}  # fmt: skip
    trajectory = JointTrajectory(
        t=cycle.t, joint_names=JOINTS, positions=cycle.q, velocities=cycle.qd, accelerations=cycle.qdd,
        limits=JointLimits(joint_names=JOINTS, min_position=arm.lower, max_position=arm.upper,
                           max_velocity=arm.max_velocity, max_acceleration=accel),
        header=Header(frame_id=arm.root, meta=meta),
    )  # fmt: skip
    tool = PoseTrajectory(t=cycle.t, positions=cycle.tool[:, :3, 3], orientations=quat_from_matrix(cycle.tool[:, :3, :3]),
                          child_frame_id="tcp", header=Header(frame_id="", meta=meta))  # fmt: skip
    return trajectory, tool, action, grasp


def sample() -> list[tuple[float | None, dict]]:
    trajectory, tool, action, grasp = _sample()
    return [(None, {"trajectory": trajectory, "tool": tool, "waypoints": action, "grasp": grasp})]


register(Dashboard(
    key=KEY,
    title="Pick and place",
    category="manipulation",
    summary="Where is the arm in the pick-and-place cycle, and did each phase go as planned?",
    consumes=(JointTrajectory, PoseTrajectory, WaypointAction, Waypoint, GraspPose),
    inputs={
        "trajectory": "the arm's JointTrajectory (a UR5e) over the whole cycle",
        "tool": "its tool point's PoseTrajectory in the arm's base frame (optional; else the flange, by FK)",
        "waypoints": "the cycle's waypoints in order: a WaypointAction, or a tuple of Waypoint",
        "grasp": "the planner's GraspPose the cycle picks with (optional)",
    },
    blueprint=blueprint,
    log=log,
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        "dash.show(\"pick_and_place\", trajectory=plan, tool=tcp, waypoints=action, grasp=grasp)"
    ),
))  # fmt: skip
