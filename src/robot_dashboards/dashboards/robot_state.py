"""Robot state: what is the robot doing right now, and is anything unhealthy?

A live dashboard. Stream whatever your robot publishes, each at its own rate; a tick may carry any subset:

    import robot_dashboards as dash

    with dash.stream("robot_state") as live:                   # opens the viewer; save="run.rrd" keeps a copy too
        for t, state, wrench, imu in robot.read():               # your loop, e.g. 25-500 Hz
            live.log(t=t, joint_state=state, wrench=wrench, imu=imu)
            if t >= next_health:                                 # slower inputs ride on any tick, e.g. 1 Hz
                live.log(t=t, servos=drives.read(), diagnostics=monitor.records())
                next_health += 1.0

- ``joint_state`` (`JointState`): moves the URDF arm AND is plotted (positions in degrees, efforts), as Foxglove's
  3D panel and plots both read one joint-state topic. Joints the URDF does not have are listed, not drawn.
- ``wrench`` (`Wrench`): the wrist force/torque sensor, in its own frame, drawn as arrows at the flange
  (orange force, purple torque) and plotted.
- ``imu`` (`Imu`): an IMU on the base: its angular velocity and acceleration, for vibration.
- ``servos`` (`ServoState`, or a tuple of them, one per bus): the drives' temperatures (and voltage / fault when
  reported). Their position, velocity and effort are the JointState's, so they are not drawn twice; without a
  ``joint_state`` in the tick, the servos' own kinematics (`ServoState.joint_state()`) drive the arm instead.
- ``diagnostics`` (`DiagnosticRecord`, or a tuple): one health lane per component (ok / warn / error over time)
  and the WARN / ERROR records as a log.

The page is split: on the left what the robot is doing (the "now" table and the arm), in the middle its state
signals (joints, efforts, wrist wrench, base IMU), on the right its health (lanes, temperatures, warnings), so a
problem never hides among the motion. Set ``robot_state.URDF`` to your robot's URDF before streaming another arm.

The sample is ~5 s of a UR5e at 25 Hz: a real bam_ruckig trajectory (``assets/data/workspace-dev-029.npz``, held
0.5 s before and ~1.8 s after it), efforts from ur_description's UR5e masses and centres of mass with a 1 kg tool,
a 25 N contact bump on the tool while it holds, a base IMU with a little vibration, drive temperatures rising slowly
with wrist_3 crossing the sample monitor's 60 °C warn threshold, and that monitor's 1 Hz DiagnosticRecords.
"""

from __future__ import annotations

import functools
import math
import weakref
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz as viz
from robot_msgs.control import ServoState
from robot_msgs.geometry import Quaternion, Vector3, Wrench
from robot_msgs.geometry.rotations import quat_from_euler
from robot_msgs.sensor import Imu, JointState
from robot_msgs.std import DiagnosticRecord, Header

from robot_dashboards.assets import UR5E_TRAJECTORY, UR5E_URDF
from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._motion_state_helpers import (
    eye,
    forward_kinematics,
    inner,
    joint_label,
    log_arm,
    pose_arm,
    tip_link,
    urdf_tree,
)

KEY = "robot_state"
ARM = "arm"  # the arm's entity and frame prefix: /world/arm/robot, frames arm/<link>
URDF = UR5E_URDF
DEG = 180.0 / math.pi
# WHY the type names as entity names: robot_msg_viz's legends then read the joint alone ("shoulder_pan_joint")
# rather than "shoulder_pan_joint · joint_state" six times over in a third of the page
NAMES = {"joint_state": "JointState", "wrench": "Wrench", "imu": "Imu", "servos": "ServoState",
         "diagnostics": "DiagnosticRecord"}  # fmt: skip
_SERVO_HEALTH = ("temperature", "voltage", "fault")  # the servo fields the JointState does not already carry
_LATEST: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
_PLACED: weakref.WeakSet[Any] = weakref.WeakSet()


class _Only:
    """A recording that writes only the entity roots in `keep` (robot_msg_viz's ServoState view writes positions,
    velocities and efforts too; here those are the JointState's, so they would draw every joint twice)."""

    def __init__(self, recording: Any, keep: tuple[str, ...]) -> None:
        self._recording, self._keep = recording, keep

    def _kept(self, path: Any) -> bool:
        return str(path).lstrip("/").split("/", 1)[0] in self._keep

    def log(self, path: Any, *args: Any, **options: Any) -> None:
        if self._kept(path):
            self._recording.log(path, *args, **options)

    def send_columns(self, path: Any, *args: Any, **options: Any) -> None:
        if self._kept(path):
            self._recording.send_columns(path, *args, **options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._recording, name)


def _many(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def _place_wrench(recording: Any) -> None:
    """Put the wrench arrows in the flange's frame (a wrist F/T sensor reports in its own frame), once."""
    stream = inner(recording)
    if stream in _PLACED:
        return
    _PLACED.add(stream)
    frame = rr.CoordinateFrame(f"{ARM}/{tip_link(URDF)}")
    for sub in ("force", "torque", "torque/spin", "at_rest"):
        stream.log(f"world/{NAMES['wrench']}/{sub}", frame, static=True)


def _now_text(latest: dict[str, Any]) -> str:
    """The table at the playhead: each joint's value now (J-numbered), the wrist load, and the health headline."""
    state: JointState | None = latest.get("joint_state")
    records: dict[str, DiagnosticRecord] = latest.get("diagnostics", {})
    order = {"error": 3, "warn": 2, "unknown": 1, "ok": 0}
    bad = sorted((r for r in records.values() if r.level in ("warn", "error")), key=lambda r: -order[r.level])
    if bad:
        health = "**health: " + bad[0].level.upper() + "** · " + " · ".join(
            f"{r.path}: {r.code}{' — ' + r.message if r.message else ''}" for r in bad)
    elif records:
        levels = {r.level for r in records.values()}
        health = "**health: ok**" if levels == {"ok"} else "**health: unknown** (" + ", ".join(sorted(levels)) + ")"
    else:
        health = "health: no diagnostics yet"
    lines = [health, ""]
    temps: dict[str, float] = {}
    for servo in latest.get("servos", ()):
        if servo.temperature is not None:
            temps.update({name: float(v) for name, v in zip(servo.name, servo.temperature, strict=True)})
    if state is not None:
        has_v, has_e = state.velocity is not None, state.effort is not None
        header = "| joint | q (°) |" + (" qd (°/s) |" if has_v else "") + (" effort (N·m) |" if has_e else "") \
            + (" drive (°C) |" if temps else "")  # fmt: skip
        lines += [header, "|---|---:|" + "---:|" * (has_v + has_e + bool(temps))]
        for index, joint in enumerate(state.name):
            row = f"| {joint_label(index, joint)} | {state.position[index] * DEG:+.1f} |"
            row += f" {state.velocity[index] * DEG:+.1f} |" if has_v else ""
            row += f" {state.effort[index]:+.1f} |" if has_e else ""
            row += (f" {temps[joint]:.1f} |" if joint in temps else " – |") if temps else ""
            lines.append(row)
        missing = latest.get("missing", [])
        if missing:
            lines += ["", f"not on the arm ({URDF.name if hasattr(URDF, 'name') else URDF} has no such joint): "
                      + ", ".join(missing)]  # fmt: skip
    extra = []
    wrench: Wrench | None = latest.get("wrench")
    if wrench is not None:
        force, torque = wrench.force.as_array(), wrench.torque.as_array()
        extra.append(f"wrist F/T |F| {np.linalg.norm(force):.1f} N, |τ| {np.linalg.norm(torque):.2f} N·m")
    imu: Imu | None = latest.get("imu")
    if imu is not None:
        extra.append(f"base IMU |ω| {np.linalg.norm(imu.angular_velocity.as_array()) * DEG:.2f} °/s")
    if extra:
        lines += ["", " · ".join(extra)]
    return "\n".join(lines)


def log(recording: Any, *, joint_state: JointState | None = None, wrench: Wrench | None = None,
        imu: Imu | None = None, servos: Any = None, diagnostics: Any = None) -> None:  # fmt: skip
    stream = inner(recording)
    latest = _LATEST.setdefault(stream, {})
    servo_states = _many(servos)
    if joint_state is None and servo_states:
        joint_state = servo_states[0].joint_state()  # the drives' own kinematics, when no JointState came with them
    if joint_state is not None:
        tree = log_arm(recording, URDF, ARM)
        latest["missing"] = pose_arm(recording, tree, ARM, joint_state.name, np.asarray(joint_state.position))
        viz.log(joint_state, NAMES["joint_state"], recording=recording)
        latest["joint_state"] = joint_state
    if wrench is not None:
        _place_wrench(recording)
        viz.log(wrench, NAMES["wrench"], recording=recording)
        latest["wrench"] = wrench
    if imu is not None:
        viz.log(imu, NAMES["imu"], recording=recording)
        latest["imu"] = imu
    for servo in servo_states:
        viz.log(servo, NAMES["servos"], recording=_Only(recording, _SERVO_HEALTH))
    if servo_states:
        latest["servos"] = servo_states
    for record in _many(diagnostics):
        viz.log(record, NAMES["diagnostics"], recording=recording)
        latest.setdefault("diagnostics", {})[record.path] = record
    # the table at the playhead, from the latest of every input (each arrives at its own rate)
    recording.log(f"{KEY}/now", rr.TextDocument(_now_text(latest), media_type="text/markdown"))


def _blueprint() -> rrb.Blueprint:
    """Left: what it is doing (the now table, the arm). Middle: state signals. Right: health, apart from state."""
    legend_off = rrb.PlotLegend(visible=False)
    now = rrb.TextDocumentView(origin=f"/{KEY}/now", name="Now: what is the robot doing, is anything unhealthy?")
    arm = rrb.Spatial3DView(
        origin="/", name="STATE · the arm (JointState) · wrist F/T arrows: orange force, purple torque",
        contents=["/world/**", f"- /world/{NAMES['imu']}/**", f"- /world/{NAMES['imu']}_level/**"],
        eye_controls=_sample_eye(),
    )  # fmt: skip
    state = rrb.Vertical(
        # WHY no legends on these plots: in a third of the page a six-joint legend covered half the curves; hover
        # names a line, and the now table names every joint with its value
        rrb.TimeSeriesView(origin="/positions", name="STATE · joint positions (°) · hover a line for its joint",
                           plot_legend=legend_off),
        rrb.TimeSeriesView(origin="/effort", name="STATE · joint efforts (N·m) · colours as positions",
                           plot_legend=legend_off),
        rrb.Tabs(
            rrb.TimeSeriesView(origin="/force", name="wrist force (N) · sensor frame x red, y green, z blue",
                               plot_legend=legend_off),
            rrb.TimeSeriesView(origin="/torque", name="wrist torque (N·m) · sensor frame x red, y green, z blue",
                               plot_legend=legend_off),
            rrb.TimeSeriesView(origin="/angular_velocity", name="base IMU angular velocity (°/s) · x, y, z",
                               plot_legend=legend_off),
            rrb.TimeSeriesView(origin="/linear_acceleration", name="base IMU acceleration (m/s²) · x, y, z",
                               plot_legend=legend_off),
            active_tab=0,
        ),
        row_shares=[1.1, 1, 1],
    )  # fmt: skip
    health = rrb.Vertical(
        rrb.StateTimelineView(origin="/health", name="HEALTH · each component (ok green, warn amber, error red)"),
        rrb.TimeSeriesView(origin="/temperature", plot_legend=legend_off,
                           name="HEALTH · drive temperatures (°C) · hover a line for its joint"),
        rrb.TextLogView(
            origin=f"/notes/{NAMES['diagnostics']}", name="HEALTH · warnings and errors",
            rows=rrb.archetypes.TextLogRows(filter_by_log_level=["WARN", "ERROR", "CRITICAL"]),
            columns=rrb.archetypes.TextLogColumns(text_log_columns=[
                rrb.encodings.TextLogColumn("LogLevel"),
                rrb.encodings.TextLogColumn("Body"),
                rrb.encodings.TextLogColumn("EntityPath", visible=False),
            ]),
        ),
        row_shares=[1, 1.3, 1],
    )  # fmt: skip
    return rrb.Blueprint(
        rrb.Horizontal(rrb.Vertical(now, arm, row_shares=[1.5, 2]), state, health, column_shares=[1.25, 1, 1]),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline="t"),
        collapse_panels=False,
    )


# ── the sample: ~5 s of a UR5e at 25 Hz ───────────────────────────────────────────────────────────────

RATE_HZ, SECONDS = 25.0, 5.0
LEAD_S = 0.5  # held at the start pose before the trajectory begins
STAMP = 1_795_521_600.0  # 2026-11-24 12:00:00 UTC, so the stamps read like a real log
# ur_description config/ur5e/physical_parameters.yaml: link masses (kg) and centres of mass (m, link frames)
_MASS = {"shoulder_link": (3.761, (0.0, -0.00193, -0.02561)), "upper_arm_link": (8.058, (-0.2125, 0.0, 0.11336)),
         "forearm_link": (2.846, (-0.2422, 0.0, 0.0265)), "wrist_1_link": (1.37, (0.0, -0.01634, -0.0018)),
         "wrist_2_link": (1.3, (0.0, 0.01634, -0.0018)), "wrist_3_link": (0.365, (0.0, 0.0, -0.001159))}  # fmt: skip
TOOL_KG, TOOL_M = 1.0, 0.144  # a sample tool: 1 kg at the dataset's TCP, 0.144 m out along the flange's z
GRAVITY = np.array([0.0, 0.0, -9.81])
# rough reflected inertia per joint (kg·m²), so efforts carry the accelerations too: sample numbers, not a UR5e model
_INERTIA = np.array([2.0, 2.0, 0.6, 0.12, 0.12, 0.04])
WARN_C = 60.0  # the sample monitor's drive-temperature warning (moteus boards start their own derate at 60 °C,
#                robot_thermal docs/V1_RULE.md); a sample number, not a UR5e rating
_TEMP0 = np.array([34.2, 41.6, 38.3, 44.1, 43.0, 58.9])  # °C at t = 0
_TEMP_RATE = np.array([0.02, 0.06, 0.04, 0.05, 0.05, 0.30])  # °C/s: wrist_3 runs hot and crosses WARN_C at ~3.7 s


def _bump(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """0 → 1 → 0 between start and end (sin²), 0 outside."""
    phase = np.clip((t - start) / (end - start), 0.0, 1.0)
    return np.sin(np.pi * phase) ** 2


@functools.cache
def _motion() -> dict[str, Any]:
    data = np.load(UR5E_TRAJECTORY)
    names = tuple(str(n) for n in data["names"])
    t = np.round(np.arange(0.0, SECONDS + 1e-9, 1.0 / RATE_HZ), 6)
    local = np.clip(t - LEAD_S, data["t"][0], data["t"][-1])
    q, qd, qdd = (np.stack([np.interp(local, data["t"], data[key][:, j]) for j in range(len(names))], axis=1)
                  for key in ("q", "qd", "qdd"))  # fmt: skip
    rng = np.random.default_rng(20261124)
    links = forward_kinematics(URDF, names, q)
    flange = links[tip_link(URDF)]
    tcp = flange[:, :3, 3] + TOOL_M * flange[:, :3, 2]
    # the contact: the tool pushes into something while it holds; the reaction, 25 N back along the tool, a little
    # sideways
    contact = 25.0 * _bump(t, 3.6, 4.6)[:, None] * -(flange[:, :3, 2] + 0.15 * flange[:, :3, 0])
    # efforts: what each motor holds against gravity (links + tool) and the contact, plus inertia, plus sensor noise
    tree = urdf_tree(str(URDF))
    joint_of = {joint.name: joint for joint in tree.joints()}
    loads: list[tuple[np.ndarray, np.ndarray, str]] = []  # (point (N,3), force (N,3), link it acts on)
    for link, (mass, centre) in _MASS.items():
        point = links[link][:, :3, :3] @ np.asarray(centre) + links[link][:, :3, 3]
        loads.append((point, np.broadcast_to(mass * GRAVITY, point.shape), link))
    loads.append((tcp, np.broadcast_to(TOOL_KG * GRAVITY, tcp.shape), tip_link(URDF)))
    loads.append((tcp, contact, tip_link(URDF)))
    downstream = {}
    for name in names:
        child, below = joint_of[name].child_link, set()
        frontier = [child]
        while frontier:
            link = frontier.pop()
            below.add(link)
            frontier += [j.child_link for j in tree.joints() if j.parent_link == link]
        downstream[name] = below
    effort = np.zeros_like(q)
    for index, name in enumerate(names):
        joint = joint_of[name]
        pose = links[joint.child_link]
        origin, axis = pose[:, :3, 3], pose[:, :3, :3] @ np.asarray(joint.axis, dtype=np.float64)
        external = sum(np.einsum("ij,ij->i", np.cross(point - origin, force), axis)
                       for point, force, link in loads if link in downstream[name])  # fmt: skip
        effort[:, index] = -external + _INERTIA[index] * qdd[:, index]
    effort += rng.normal(0.0, 0.15, effort.shape)
    # the wrist F/T: the tool's weight and the contact, in the flange frame, about the flange origin
    rotation_t = np.transpose(flange[:, :3, :3], (0, 2, 1))
    load = TOOL_KG * GRAVITY + contact
    lever = tcp - flange[:, :3, 3]
    force = np.einsum("nij,nj->ni", rotation_t, load) + rng.normal(0.0, 0.25, load.shape)
    torque = np.einsum("nij,nj->ni", rotation_t, np.cross(lever, load)) + rng.normal(0.0, 0.01, load.shape)
    # the base IMU: gravity (it reads +g at rest), a 0.2° mounting tilt, and the base answering the arm's accelerations
    accel = np.tile([0.0, 0.0, 9.81], (len(t), 1)) + rng.normal(0.0, 0.03, (len(t), 3))
    accel[:, 0] += 0.04 * qdd[:, 1]
    accel[:, 1] += 0.04 * qdd[:, 0]
    gyro = rng.normal(0.0, 0.0008, (len(t), 3))
    gyro[:, 2] += 0.0025 * qdd[:, 0]
    gyro[:, 0] += 0.0015 * qdd[:, 1]
    tilt = Quaternion.from_array(quat_from_euler(math.radians(0.2), math.radians(-0.1), 0.0))
    temperature = np.round(_TEMP0[None, :] + _TEMP_RATE[None, :] * t[:, None] + rng.normal(0.0, 0.03, q.shape), 1)
    return {"names": names, "t": t, "q": q, "qd": qd, "effort": effort, "force": force, "torque": torque,
            "accel": accel, "gyro": gyro, "tilt": tilt, "temperature": temperature}  # fmt: skip


def _diagnostics(second: int, m: dict[str, Any], k: int) -> tuple[DiagnosticRecord, ...]:
    """The sample monitor's records at one whole second: the controller, the F/T sensor and the drives' heat."""
    header = Header(stamp=STAMP + second)
    hottest = int(np.argmax(m["temperature"][k]))
    hot_c = float(m["temperature"][k, hottest])
    hot = m["names"][hottest]
    up = 1830.0 + second
    records = [
        DiagnosticRecord(header=header, path="ur5e.controller", level="ok", message="running", mode="operational",
                         uptime_s=up),
        DiagnosticRecord(header=header, path="ur5e.ft_sensor", level="ok", message="streaming at 25 Hz",
                         mode="operational", uptime_s=up),
    ]  # fmt: skip
    if hot_c >= WARN_C:
        records.append(DiagnosticRecord(header=header, path="ur5e.drives.thermal", level="warn",
                                        code="TemperatureHigh", mode="operational", uptime_s=up,
                                        message=f"{hot} drive {hot_c:.1f} °C ≥ {WARN_C:.0f} °C warn"))  # fmt: skip
    else:
        records.append(DiagnosticRecord(header=header, path="ur5e.drives.thermal", level="ok", mode="operational",
                                        uptime_s=up, message=f"hottest {hot} {hot_c:.1f} °C"))  # fmt: skip
    return tuple(records)


@functools.cache
def _sample() -> tuple[tuple[float, dict[str, Any]], ...]:
    m = _motion()
    names = m["names"]
    servo_every = int(RATE_HZ // 5)  # the drives' temperatures at 5 Hz
    ticks = []
    for k, t in enumerate(m["t"]):
        header = Header(stamp=STAMP + float(t), frame_id="base_link_ur")
        inputs: dict[str, Any] = {
            "joint_state": JointState(name=names, position=m["q"][k], velocity=m["qd"][k], effort=m["effort"][k],
                                      header=header),
            "wrench": Wrench(Vector3.from_array(m["force"][k]), Vector3.from_array(m["torque"][k])),
            "imu": Imu(orientation=m["tilt"], angular_velocity=Vector3.from_array(m["gyro"][k]),
                       linear_acceleration=Vector3.from_array(m["accel"][k]), header=header),
        }  # fmt: skip
        if k % servo_every == 0:
            drives = ServoState(name=names, position=m["q"][k], velocity=m["qd"][k], effort=m["effort"][k],
                                temperature=m["temperature"][k], seq=k // servo_every, header=header)  # fmt: skip
            inputs["servos"] = (drives,)  # one bus
        if abs(t - round(t)) < 1e-9:
            inputs["diagnostics"] = _diagnostics(int(round(t)), m, k)
        ticks.append((float(t), inputs))
    return tuple(ticks)


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    return list(_sample())


@functools.cache
def _sample_eye() -> rrb.EyeControls3D:
    """A fixed camera on everything the arm can sweep in the sample, centred on its base.

    WHY on the base and not on the motion's box: a live view must not move, and the arm turns about its base; the
    box's centre sits off to one side of the base, which cut the arm off at the edge once it had turned away."""
    m = _motion()
    links = forward_kinematics(URDF, m["names"], m["q"])
    points = np.vstack([pose[:, :3, 3] for pose in links.values()])
    reach = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    low, high = float(points[:, 2].min()), float(points[:, 2].max())
    ring = np.array([[reach * math.cos(a), reach * math.sin(a), z] for a in np.linspace(0, 2 * math.pi, 16)
                     for z in (low, high)])  # fmt: skip
    return eye(ring, distance_radii=2.05)


register(Dashboard(
    key=KEY,
    title="Robot state",
    category="robot state",
    summary="What is the robot doing right now, and is anything unhealthy? A live page: the URDF arm moved by the "
            "JointState with its wrist wrench, the joints, efforts and base IMU over time, and apart from them the "
            "health: a lane per component, drive temperatures and the warnings.",
    consumes=(JointState, Wrench, Imu, ServoState, DiagnosticRecord),
    inputs={
        "joint_state": "the measured joints: they move the arm and are plotted",
        "wrench": "optional: the wrist force/torque sensor, in its own frame",
        "imu": "optional: an IMU on the base (vibration)",
        "servos": "optional: the drives' ServoState (one, or a tuple per bus): temperatures, voltage, faults",
        "diagnostics": "optional: DiagnosticRecords (one, or a tuple): a health lane per component",
    },
    viewers={"rerun": RerunView(blueprint=_blueprint, log=log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        'with dash.stream("robot_state") as live:\n'
        "    for t, state, wrench, imu in robot.read():          # e.g. 25-500 Hz\n"
        "        live.log(t=t, joint_state=state, wrench=wrench, imu=imu)\n"
        "        if t >= next_health:                            # e.g. 1 Hz\n"
        "            live.log(t=t, servos=drives.read(), diagnostics=monitor.records())\n"
        "            next_health += 1.0"
    ),
))  # fmt: skip
