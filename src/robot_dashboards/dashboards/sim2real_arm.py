"""Sim-to-real · 6-DOF arm: across a speed sweep, does the sim predict the real UR5e better than the plan does, and
which sensor streams does it get wrong?

    import robot_dashboards as dash

    dash.show("sim2real_arm", dataset=runs)        # a robot_msgs Dataset: runs × {real, sim, target}

- ``dataset`` (`robot_msgs.compare.Dataset`): one `Run` per experiment, recordings ``real`` (first: the reference),
  ``sim`` and ``target`` (the plan itself, as the do-nothing model: a sim that cannot beat "the arm does what it was
  told" is not earning its keep). Streams by name: ``servo_state`` (the drives' report), ``servo_command`` (what was
  sent), ``imu`` (a wrist IMU on the flange), ``wrench`` (the wrist F/T sensor); the target records ``servo_state``
  only, so its other pairs are listed as not compared, with why.

The page (robot_dashboards' sim-to-real layout, `_sim2real_helpers`): the answer first (the models ranked on the
servo encoders, the worst joint in degrees, with a tolerance that grows with time), the real arm on the URDF playing
the headline run, then robot_msg_viz's DatasetComparison across runs (every stream, worst first), then each run's
comparisons and its recorded streams overlaid.

The sample is robot_msg_viz's own sim-to-real story (`robot_msg_viz.views._sim2real`), called rather than copied so
the dashboard and its stock views show one dataset: a UR5e pick at 0.5×, 0.75× and 1× speed; sim lags 8 ms with no sag
and clean sensors, the real arm lags 15 ms, sags under gravity, runs warmer, its sensors are noisy and biased, its
wrist rings at 6 Hz after each stop, and at full speed wrist 1 trips a following-error fault for 60 ms.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import rerun.blueprint as rrb
from robot_msg_viz import TIMELINE

# WHY robot_msg_viz's sample module: the arm's sim-to-real story is already built there for its stream and dataset
# views (seeded, physical: friction, gravity sag, a following-error fault); CI pins robot_msg_viz to a commit
from robot_msg_viz.views import _sim2real as story
from robot_msgs.compare import Dataset, Recording, Run
from robot_msgs.std import Header
from robot_msgs.trajectory import ServoStateTrajectory

from robot_dashboards.assets import UR5E_URDF
from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._motion_state_helpers import eye, forward_kinematics, log_arm, pose_arm
from robot_dashboards.dashboards._sim2real_helpers import Envelope, Ruler, Sim2RealView, joint_error_deg

KEY = "sim2real_arm"
URDF = UR5E_URDF
ARM = "real_arm"
HEADLINE = "pick_1x"
ENVELOPE = Envelope(floor=0.1, growth=0.1, unit="°")  # 0.1° ≈ 1.5 mm at the UR5e's 0.85 m reach; +0.1°/s


def _recording(source: str, speed: float) -> Recording:
    if source == "target":
        plan = story.plan(speed)
        state = ServoStateTrajectory(t=plan.t, joint_names=plan.joint_names, positions=plan.positions,
                                     velocities=plan.velocities, header=Header(stamp=0.0, frame_id="base_link"))
        return Recording(streams={"servo_state": state}, header=Header(meta={"planner": "min-jerk pick v3"}))
    streams = {"servo_state": story.servo_states(source, speed), "servo_command": story.servo_commands(source, speed),
               "imu": story.imu(source, speed), "wrench": story.wrench(source, speed)}  # fmt: skip
    meta = {"engine": "mujoco 3.3"} if source == "sim" else {"rig": "ur5e cell 2", "firmware": "5.21"}
    return Recording(streams=streams, header=Header(meta=meta))


@functools.cache
def sample_dataset() -> Dataset:
    runs = tuple(
        Run(name=f"pick_{speed:g}x", title=f"pick at {speed:g}× speed",
            recordings={source: _recording(source, speed) for source in ("real", "sim", "target")},
            pairs=(("real", "sim"), ("real", "target")), tags={"speed": speed, "payload_kg": 0.8})
        for speed in story.SPEEDS
    )  # fmt: skip
    return Dataset(runs=runs, title="UR5e pick speed sweep: real vs sim vs the plan")


# ── the real arm on the URDF, playing the headline run (the answer's picture of what was compared) ─────────────────


def _log_arm(stream: Any, dataset: Dataset) -> None:
    run = next((r for r in dataset.runs if r.name == HEADLINE), dataset.runs[0])
    state = run.recordings["real"].streams.get("servo_state")
    if not isinstance(state, ServoStateTrajectory):
        return
    tree = log_arm(stream, URDF, ARM)
    seconds = np.asarray(state.t, dtype=np.float64) - float(state.t[0])
    for second, positions in zip(seconds, np.asarray(state.positions), strict=True):
        stream.set_time(TIMELINE, duration=float(second))
        pose_arm(stream, tree, ARM, tuple(state.joint_names), positions)
    stream.reset_time()


@functools.cache
def _eye() -> rrb.EyeControls3D:
    """robot_state's camera: on a ring of the arm's reach about its base, so the arm never leaves the view."""
    state = sample_dataset().runs[-1].recordings["real"].streams["servo_state"]
    links = forward_kinematics(URDF, tuple(state.joint_names), np.asarray(state.positions)[::10])
    points = np.vstack([poses[:, :3, 3] for poses in links.values()])
    reach = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    low, high = float(points[:, 2].min()), float(points[:, 2].max())
    ring = np.array([[reach * np.cos(a), reach * np.sin(a), z] for a in np.linspace(0, 2 * np.pi, 16)
                     for z in (low, high)])  # fmt: skip
    return eye(ring, distance_radii=1.5)


def _arm_views(dataset: Dataset) -> list[Any]:
    return [rrb.Spatial3DView(origin="/", name=f"{HEADLINE} · the real arm (its encoders) on the UR5e URDF",
                              contents=[f"/world/{ARM}/**"], eye_controls=_eye())]  # fmt: skip


VIEW = Sim2RealView(
    key=KEY,
    ruler=Ruler(stream="servo_state", label="servo encoders, worst joint", error=joint_error_deg, envelope=ENVELOPE),
    models=("sim", "target"),
    headline=HEADLINE,
    question="Does the sim predict the real arm better than the plan does, and on which streams is it wrong?",
    overlay=(HEADLINE,),  # WHY the headline only: four 6-joint streams per run, ~1 MB of .rrd each
    notes=(
        "`target` is the plan used as a model (the arm does exactly what it was told): the baseline a sim must beat. "
        "Across runs: every stream (servo state and command, wrist IMU, wrist F/T), worst first. At 1× the real "
        "wrist 1 trips a following-error fault the sim never shows: see servo_state's fault lane."
    ),
    sample=sample_dataset,
    extra_log=_log_arm,
    extra_views=_arm_views,
)


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    return [(None, {"dataset": sample_dataset()})]


register(Dashboard(
    key=KEY,
    title="Sim-to-real · 6-DOF arm",
    category="motion",
    summary="Across a speed sweep, does the sim predict the real UR5e better than the plan does, and which sensor "
            "streams does it get wrong? Models ranked on the encoders with a tolerance that grows with time, the "
            "real arm on its URDF, then servo state, servo command, wrist IMU and wrist F/T compared per run.",
    consumes=(Dataset,),
    inputs={"dataset": "a robot_msgs Dataset: one Run per experiment, recordings real (first), sim and target, "
                       "streams servo_state, servo_command, imu, wrench"},
    viewers={"rerun": RerunView(blueprint=VIEW.sample_blueprint, log=VIEW.log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n"
        "from robot_msgs.compare import Dataset, Recording, Run\n\n"
        'real = Recording(streams={"servo_state": drives, "servo_command": commands, "imu": wrist_imu, "wrench": ft})\n'
        'run = Run(name="pick_1x", recordings={"real": real, "sim": sim, "target": plan}, tags={"speed": 1.0})\n'
        'dash.show("sim2real_arm", dataset=Dataset(runs=(run, ...)))'
    ),
))  # fmt: skip
