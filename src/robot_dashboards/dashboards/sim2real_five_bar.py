"""Sim-to-real · five-bar: does the sim put the five-bar's tool where the real one goes, and how fast does it stop?

    import robot_dashboards as dash

    dash.show("sim2real_five_bar", dataset=runs)   # a robot_msgs Dataset: runs × {real, sim_flex, sim, target}

WHY a five-bar (BAM's pilot cell is one): a parallel arm, two base motors sharing one tool, so a joint error reaches
the tool through a Jacobian that changes across the workspace, and the ruler is the tool itself (a camera on an ArUco
marker), not the encoders.

- ``dataset`` (`robot_msgs.compare.Dataset`): one `Run` per experiment, recordings ``real`` (first: the reference)
  and each model. Streams by name: ``servo_state`` (the two base motors' encoders, velocity, torque) and ``tool``
  (a `PoseTrajectory`: the tool seen by an overhead camera, 60 Hz).

The page (robot_dashboards' sim-to-real layout, `_sim2real_helpers`): the models ranked on the tool position in mm
with a tolerance that grows with time, where each source's tool went, robot_msg_viz's DatasetComparison across runs,
then each run's comparisons and its recorded streams overlaid.

The sample (seeded, deterministic): base motors 0.2 m apart, 0.25 m proximal and 0.35 m distal links, a pick to
place move along a straight line (minimum-jerk, 0.1 s dwells) at three speeds. Each motor is a position servo (a
first-order lag) through a gear with backlash (a play operator on the link side), and the distal links flex under the
tool's acceleration. Four sources per run:

- ``real``: 12 ms servo lag, 0.4° of backlash, 0.9 mm of tool deflection per m/s², camera noise 0.4 mm.
- ``sim_flex``: the identified model: 11 ms, 0.35°, 0.8 mm per m/s² (a flexible-link sim).
- ``sim``: rigid links, ideal gears, 10 ms lag.
- ``target``: the plan itself (the tool does exactly what it was told): the baseline a sim must beat.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msg_viz import TIMELINE
from robot_msgs.compare import Dataset, Recording, Run
from robot_msgs.std import Header
from robot_msgs.trajectory import PoseTrajectory, ServoStateTrajectory

from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._sim2real_helpers import Envelope, Ruler, Sim2RealView, position_error_mm

KEY = "sim2real_five_bar"
JOINTS = ("left_motor", "right_motor")
BASE = 0.2  # m between the motors
PROXIMAL, DISTAL = 0.25, 0.35  # m
PICK, PLACE = np.array([-0.12, 0.38]), np.array([0.14, 0.44])  # m, in the plane of the linkage
DWELL_S = 0.1
MOVES_S = (1.0, 0.7, 0.45)  # the move's duration at each speed
RATE_HZ, CAMERA_HZ = 200.0, 60.0
SEED = 23
HEADLINE = "move_0.45s"
ENVELOPE = Envelope(floor=1.5, growth=1.0, unit="mm")  # 1.5 mm ≈ 3σ of the camera's 0.4 mm noise, +1 mm per second
LINK_INERTIA = np.array([0.012, 0.012])  # kg·m² each base motor carries
VISCOUS = np.array([0.05, 0.05])  # N·m·s/rad


def inverse(point: np.ndarray) -> np.ndarray:
    """The two motor angles (rad) that put the tool at `point` (N, 2), elbows out."""
    angles = []
    for side, sign in ((-1.0, 1.0), (1.0, -1.0)):
        base = np.array([side * BASE / 2.0, 0.0])
        to = point - base
        reach = np.linalg.norm(to, axis=1)
        heading = np.arctan2(to[:, 1], to[:, 0])
        elbow = np.arccos(np.clip((PROXIMAL**2 + reach**2 - DISTAL**2) / (2 * PROXIMAL * reach), -1.0, 1.0))
        # left elbow out to the left (heading + elbow), right elbow out to the right (heading - elbow)
        angles.append(heading + sign * elbow)
    return np.column_stack(angles)


def forward(angles: np.ndarray) -> np.ndarray:
    """The tool point (N, 2) for motor angles (N, 2): where the two distal links meet, the branch away from the base."""
    left = np.column_stack([-BASE / 2 + PROXIMAL * np.cos(angles[:, 0]), PROXIMAL * np.sin(angles[:, 0])])
    right = np.column_stack([BASE / 2 + PROXIMAL * np.cos(angles[:, 1]), PROXIMAL * np.sin(angles[:, 1])])
    middle = (left + right) / 2.0
    half = np.linalg.norm(right - left, axis=1) / 2.0
    rise = np.sqrt(np.clip(DISTAL**2 - half**2, 0.0, None))
    along = (right - left) / (2.0 * half[:, None])
    normal = np.column_stack([-along[:, 1], along[:, 0]])  # left-to-right turned +90°: away from the base (up)
    return middle + rise[:, None] * normal


def _plan(move_s: float) -> tuple[np.ndarray, np.ndarray]:
    """t (s) and the tool point (N, 2) at 1 kHz: dwell, minimum-jerk straight line, dwell."""
    t = np.arange(round((move_s + 2 * DWELL_S) * 1000) + 1) / 1000.0
    s = np.clip((t - DWELL_S) / move_s, 0.0, 1.0)
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    return t, PICK + blend[:, None] * (PLACE - PICK)


def _lag(t: np.ndarray, command: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty_like(command)
    out[0] = command[0]
    for k, step in enumerate(np.diff(t)):
        out[k + 1] = out[k] + (1.0 - math.exp(-step / tau)) * (command[k + 1] - out[k])
    return out


def _play(motor: np.ndarray, backlash_deg: float) -> np.ndarray:
    """The link side of a gear with backlash: it moves only once the motor has taken up the play (a play operator)."""
    half = math.radians(backlash_deg) / 2.0
    out = np.empty_like(motor)
    out[0] = motor[0]
    for k in range(1, len(motor)):
        out[k] = np.clip(out[k - 1], motor[k] - half, motor[k] + half)
    return out


# (servo lag s, backlash deg, deflection m per m/s², camera noise m)
MODELS = {
    "real": (0.012, 0.4, 0.0009, 0.0004),
    "sim_flex": (0.011, 0.35, 0.0008, 0.0),
    "sim": (0.010, 0.0, 0.0, 0.0),
    "target": (0.0, 0.0, 0.0, 0.0),
}


def _recording(source: str, move_s: float, index: int) -> Recording:
    lag, backlash, deflection, noise_m = MODELS[source]
    noise = np.random.default_rng(SEED + 17 * index + list(MODELS).index(source))
    t, planned = _plan(move_s)
    command = inverse(planned)
    motor = _lag(t, command, lag) if lag else command
    link = np.column_stack([_play(motor[:, j], backlash) for j in range(2)]) if backlash else motor
    # the distal links bend against the tool's acceleration; WHY the motor side's (smooth) acceleration: the play
    # operator's kinks are impulses in the link side's, which a link's own inertia never passes on
    acceleration = np.gradient(np.gradient(forward(motor), t, axis=0), t, axis=0)
    tool = forward(link) - deflection * acceleration
    stamps = np.arange(int(t[-1] * RATE_HZ) + 1) / RATE_HZ
    position = np.column_stack([np.interp(stamps, t, motor[:, j]) for j in range(2)])
    velocity = np.gradient(position, stamps, axis=0)
    effort = LINK_INERTIA * np.gradient(velocity, stamps, axis=0) + VISCOUS * velocity
    if source == "real":
        position = position + np.radians(noise.normal(0.0, 0.01, position.shape))
        effort = effort + noise.normal(0.0, 0.02, effort.shape)
    state = ServoStateTrajectory(t=stamps, joint_names=JOINTS, positions=position, velocities=velocity,
                                 effort=None if source == "target" else effort,
                                 header=Header(stamp=0.0, frame_id="base_link"))  # fmt: skip
    frames = np.arange(int(t[-1] * CAMERA_HZ) + 1) / CAMERA_HZ
    seen = np.column_stack([np.interp(frames, t, tool[:, axis]) for axis in range(2)])
    if noise_m:
        seen = seen + noise.normal(0.0, noise_m, seen.shape)
    # the linkage's plane is the cell's x-z plane (the tool reaches out along z, above the belt); y = 0
    xyz = np.column_stack([seen[:, 0], np.zeros(len(frames)), seen[:, 1]])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (len(frames), 1))
    camera = PoseTrajectory(t=frames, positions=xyz, orientations=quaternions, child_frame_id="tool",
                            header=Header(stamp=0.0, frame_id="base_link", meta={"sensor": "ArUco on the tool, 60 Hz"}))
    meta = {"real": {"rig": "BAM five-bar pilot cell"}, "sim_flex": {"model": "flexible distal links + backlash"},
            "sim": {"model": "rigid links, ideal gears"}, "target": {"planner": "minimum-jerk line"}}[source]  # fmt: skip
    return Recording(streams={"servo_state": state, "tool": camera}, header=Header(meta=meta))


@functools.cache
def sample_dataset() -> Dataset:
    runs = []
    for index, move_s in enumerate(MOVES_S):
        peak = 1.875 * float(np.linalg.norm(PLACE - PICK)) / move_s  # minimum-jerk peak speed
        runs.append(Run(
            name=f"move_{move_s:g}s", title=f"pick → place in {move_s:g} s (peak {peak:.2f} m/s)",
            recordings={source: _recording(source, move_s, index) for source in MODELS},
            pairs=(("real", "sim_flex"), ("real", "sim"), ("real", "target")),
            tags={"move_s": move_s, "peak_m_s": round(peak, 2)},
        ))  # fmt: skip
    return Dataset(runs=tuple(runs), title="five-bar pick → place speed sweep")


# ── the real linkage, playing the headline run (the answer's picture of the morphology) ──────────────────────────────


def _plane(points: np.ndarray) -> np.ndarray:
    """(N, 2) points of the linkage's plane as (N, 3) in the cell: x across, z up the plane, y = 0."""
    return np.column_stack([points[:, 0], np.zeros(len(points)), points[:, 1]])


def _log_linkage(stream: Any, dataset: Dataset) -> None:
    run = next((r for r in dataset.runs if r.name == HEADLINE), dataset.runs[0])
    real = run.recordings["real"].streams
    state, tool = real.get("servo_state"), real.get("tool")
    if not isinstance(state, ServoStateTrajectory) or not isinstance(tool, PoseTrajectory):
        return
    angles = np.asarray(state.positions, dtype=np.float64)
    keep = np.arange(0, len(angles), 4)  # 50 Hz is plenty to watch it move
    bases = np.array([[-BASE / 2, 0.0], [BASE / 2, 0.0]])
    stream.log(f"{KEY}/linkage/base", rr.LineStrips3D([_plane(bases)], colors=[(120, 120, 125)],
                                                       radii=rr.Radius.ui_points(4.0)), static=True)  # fmt: skip
    stream.log(f"{KEY}/linkage/tool_path", rr.LineStrips3D([np.asarray(tool.positions, dtype=np.float64)],
                                                            colors=[(95, 200, 120)], radii=rr.Radius.ui_points(1.5)),
               static=True)  # fmt: skip
    seconds = np.asarray(state.t, dtype=np.float64) - float(state.t[0])
    tips = forward(angles[keep])
    for second, (left, right), tip in zip(seconds[keep], angles[keep], tips, strict=True):
        elbows = bases + PROXIMAL * np.array([[np.cos(left), np.sin(left)], [np.cos(right), np.sin(right)]])
        chain = np.array([bases[0], elbows[0], tip, elbows[1], bases[1]])
        stream.set_time(TIMELINE, duration=float(second))
        stream.log(f"{KEY}/linkage/arms", rr.LineStrips3D([_plane(chain)], colors=[(230, 230, 235)],
                                                           radii=rr.Radius.ui_points(3.0)))  # fmt: skip
        stream.log(f"{KEY}/linkage/joints", rr.Points3D(_plane(chain), colors=[(255, 160, 60)],
                                                         radii=rr.Radius.ui_points(6.0)))  # fmt: skip
    stream.reset_time()


def _linkage_views(dataset: Dataset) -> list[Any]:
    # face-on to the linkage's plane, a little above it, close enough that the linkage fills the view
    camera = rrb.EyeControls3D(kind="Orbital", look_target=[0.01, 0.0, 0.24], eye_up=[0.0, 0.0, 1.0],
                               position=[0.01, -0.62, 0.36])  # fmt: skip
    return [rrb.Spatial3DView(origin=f"/{KEY}/linkage", eye_controls=camera,
                              name=f"{HEADLINE} · the real five-bar from its encoders; green: the tool the camera saw")]


VIEW = Sim2RealView(
    key=KEY,
    ruler=Ruler(stream="tool", label="tool position (camera on an ArUco marker)", error=position_error_mm,
                envelope=ENVELOPE),
    models=("sim_flex", "sim", "target"),
    headline=HEADLINE,
    question="Does the sim put the five-bar's tool where the real one goes, and does it stop the same way?",
    overlay=(HEADLINE,),
    paths=False,  # WHY: millimetres apart on a 0.28 m line, the paths draw as one; the linkage reads better
    extra_log=_log_linkage,
    extra_views=_linkage_views,
    notes=(
        "Two motors share one tool, so the encoders can agree while the tool does not: the ruler is the camera on the "
        "tool. The flexible-link sim also predicts the overshoot as the tool stops; the rigid one cannot."
    ),
    sample=sample_dataset,
)


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    return [(None, {"dataset": sample_dataset()})]


register(Dashboard(
    key=KEY,
    title="Sim-to-real · five-bar",
    category="motion",
    summary="Does the sim put the five-bar's tool where the real one goes, and does it stop the same way? A "
            "parallel arm (BAM's pilot cell): models ranked on the tool position from a camera, in mm, with a "
            "tolerance that grows with time, where each source's tool went, then each run's motors and tool compared.",
    consumes=(Dataset,),
    inputs={"dataset": "a robot_msgs Dataset: one Run per experiment, recordings real (first) and each model, "
                       "streams servo_state (the two motors) and tool (a PoseTrajectory from the camera)"},
    viewers={"rerun": RerunView(blueprint=VIEW.sample_blueprint, log=VIEW.log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n"
        "from robot_msgs.compare import Dataset, Recording, Run\n\n"
        'real = Recording(streams={"servo_state": motors, "tool": camera_tool_poses})\n'
        'run = Run(name="move_0.45s", recordings={"real": real, "sim": sim, "target": plan})\n'
        'dash.show("sim2real_five_bar", dataset=Dataset(runs=(run, ...)))'
    ),
))  # fmt: skip
