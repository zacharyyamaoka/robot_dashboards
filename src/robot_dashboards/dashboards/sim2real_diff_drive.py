"""Sim-to-real · differential-drive base: driven open loop, how far does each model's base drift from the real one?

    import robot_dashboards as dash

    dash.show("sim2real_diff_drive", dataset=runs)  # a robot_msgs Dataset: runs × {real, sim_slip, sim}

WHY a differential-drive base: the commonest mobile robot, and the clearest case of error that grows with time. Its
pose is the integral of its wheel speeds, so a 1 % wheel-radius error is a heading error that keeps compounding: the
envelope must grow, and a model is judged on how long it stays usable for dead reckoning.

- ``dataset`` (`robot_msgs.compare.Dataset`): one `Run` per experiment, recordings ``real`` (first: the reference)
  and each model. Streams by name: ``wheels`` (`ServoStateTrajectory`: the two wheel drives' angle, speed, torque),
  ``imu`` (the base IMU: yaw rate, acceleration) and ``pose`` (a `PoseTrajectory` of the base: motion capture on the
  real robot, the simulated pose in a model).

The page (robot_dashboards' sim-to-real layout, `_sim2real_helpers`): the models ranked on the base position in mm
with a tolerance that grows with time, where each source's base went (top view), robot_msg_viz's DatasetComparison
across runs, then each run's comparisons and its recorded streams overlaid.

The sample (seeded, deterministic), all driven open loop by the same wheel-speed commands (no feedback on the pose):
a slalom at 0.3, 0.6 and 0.9 m/s for 6 s, wheels 0.1 m across, 0.3 m apart, an on-board wheel-speed loop (a first-
order lag). Three sources per run:

- ``real``: worn wheels (radius 49.0 / 49.7 mm against the nominal 50), the tyres skid in turns (effective track
  7 % wider), 1.5 % longitudinal slip, a 70 ms wheel loop; motion capture with 1 mm of noise, an IMU with noise and
  gyro bias.
- ``sim_slip``: the calibrated model: radii 49.1 / 49.6 mm, 6 % skid, 1.4 % slip, 70 ms.
- ``sim``: the nominal model: 50 mm wheels, no skid, no slip, 60 ms.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
from robot_msgs.compare import Dataset, Recording, Run
from robot_msgs.std import Header
from robot_msgs.trajectory import ImuTrajectory, PoseTrajectory, ServoStateTrajectory

from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._sim2real_helpers import Envelope, Ruler, Sim2RealView, position_error_mm

KEY = "sim2real_diff_drive"
WHEELS = ("left_wheel", "right_wheel")
RADIUS, TRACK = 0.05, 0.30  # m, nominal
SPEEDS = (0.3, 0.6, 0.9)  # m/s
SECONDS, RAMP_S, PERIOD_S, TURN_RADIUS = 6.0, 0.5, 3.0, 0.8  # the slalom: curvature up to 1 / 0.8 m
DT, RATE_HZ, POSE_HZ = 0.001, 100.0, 50.0
WHEEL_INERTIA, ROLLING = 0.004, 0.08  # kg·m² per wheel (with the base's share), N·m rolling resistance
G = 9.80665
SEED = 41
HEADLINE = "slalom_0.9mps"
ENVELOPE = Envelope(floor=5.0, growth=5.0, unit="mm")  # 5 mm: mocap + start placement; +5 mm/s ≈ 1 % at 0.5 m/s

# (left radius m, right radius m, track multiplier from skid, longitudinal slip, wheel loop lag s)
MODELS = {
    "real": (0.0490, 0.0497, 1.07, 0.015, 0.070),
    "sim_slip": (0.0491, 0.0496, 1.06, 0.014, 0.070),
    "sim": (RADIUS, RADIUS, 1.0, 0.0, 0.060),
}


def _commands(speed: float) -> tuple[np.ndarray, np.ndarray]:
    """t and the commanded wheel speeds (rad/s, left / right) for the slalom at `speed`: what every source is sent."""
    t = np.arange(round(SECONDS / DT) + 1) * DT
    v = speed * np.clip(t / RAMP_S, 0.0, 1.0)
    w = (v / TURN_RADIUS) * np.sin(2 * np.pi * t / PERIOD_S)
    return t, np.column_stack([(v - w * TRACK / 2) / RADIUS, (v + w * TRACK / 2) / RADIUS])


def _drive(source: str, speed: float) -> dict[str, np.ndarray]:
    left_r, right_r, skid, slip, lag = MODELS[source]
    t, command = _commands(speed)
    wheel = np.empty_like(command)
    wheel[0] = command[0]
    blend = 1.0 - math.exp(-DT / lag)
    for k in range(1, len(t)):
        wheel[k] = wheel[k - 1] + blend * (command[k] - wheel[k - 1])
    ground = wheel * np.array([left_r, right_r]) * (1.0 - slip)  # each wheel's contact speed (m/s)
    v = ground.mean(axis=1)
    yaw_rate = (ground[:, 1] - ground[:, 0]) / (TRACK * skid)
    yaw = np.concatenate([[0.0], np.cumsum(yaw_rate[:-1]) * DT])
    x = np.concatenate([[0.0], np.cumsum(v[:-1] * np.cos(yaw[:-1])) * DT])
    y = np.concatenate([[0.0], np.cumsum(v[:-1] * np.sin(yaw[:-1])) * DT])
    angle = np.concatenate([[np.zeros(2)], np.cumsum(wheel[:-1], axis=0) * DT])
    accel = np.gradient(wheel, t, axis=0)
    effort = WHEEL_INERTIA * accel + ROLLING * np.tanh(wheel / 0.5)
    return {"t": t, "x": x, "y": y, "yaw": yaw, "v": v, "yaw_rate": yaw_rate, "dv": np.gradient(v, t),
            "angle": angle, "wheel": wheel, "effort": effort}  # fmt: skip


def _quaternions(yaw: np.ndarray) -> np.ndarray:
    return np.column_stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)])


def _recording(source: str, speed: float, index: int) -> Recording:
    drive = _drive(source, speed)
    real = source == "real"
    noise = np.random.default_rng(SEED + 13 * index + list(MODELS).index(source))

    def at(stamps: np.ndarray, key: str) -> np.ndarray:
        values = drive[key]
        if values.ndim == 1:
            return np.interp(stamps, drive["t"], values)
        return np.column_stack([np.interp(stamps, drive["t"], values[:, j]) for j in range(values.shape[1])])

    stamps = np.arange(int(SECONDS * RATE_HZ) + 1) / RATE_HZ
    angle, speed_now, effort = at(stamps, "angle"), at(stamps, "wheel"), at(stamps, "effort")
    if real:
        angle = angle + noise.normal(0.0, 0.0005, angle.shape)
        effort = effort + noise.normal(0.0, 0.01, effort.shape)
    wheels = ServoStateTrajectory(t=stamps, joint_names=WHEELS, positions=angle, velocities=speed_now, effort=effort,
                                  header=Header(stamp=0.0, frame_id="base_link"))  # fmt: skip

    yaw, yaw_rate, v, dv = (at(stamps, key) for key in ("yaw", "yaw_rate", "v", "dv"))
    gyro = np.column_stack([np.zeros_like(yaw_rate), np.zeros_like(yaw_rate), yaw_rate])
    specific = np.column_stack([dv, v * yaw_rate, np.full_like(v, G)])  # forward, centripetal, gravity (base frame)
    if real:
        gyro = gyro + noise.normal(0.0, 0.004, gyro.shape) + [0.0, 0.0, 0.003]
        specific = specific + noise.normal(0.0, 0.06, specific.shape) + [0.05, -0.03, 0.02]
    imu = ImuTrajectory(t=stamps, orientations=_quaternions(yaw), angular_velocities=gyro,
                        linear_accelerations=specific, header=Header(stamp=0.0, frame_id="imu_link"))  # fmt: skip

    frames = np.arange(int(SECONDS * POSE_HZ) + 1) / POSE_HZ
    xyz = np.column_stack([at(frames, "x"), at(frames, "y"), np.zeros(len(frames))])
    if real:
        xyz = xyz + np.column_stack([noise.normal(0.0, 0.001, (len(frames), 2)), np.zeros(len(frames))])
    pose = PoseTrajectory(t=frames, positions=xyz, orientations=_quaternions(at(frames, "yaw")), child_frame_id="base",
                          header=Header(stamp=0.0, frame_id="odom" if not real else "mocap"))  # fmt: skip
    meta = {"real": {"rig": "mobile base, mocap room"}, "sim_slip": {"model": "calibrated radii, skid and slip"},
            "sim": {"model": "nominal kinematics"}}[source]  # fmt: skip
    return Recording(streams={"wheels": wheels, "imu": imu, "pose": pose}, header=Header(meta=meta))


@functools.cache
def sample_dataset() -> Dataset:
    runs = tuple(
        Run(name=f"slalom_{speed:g}mps", title=f"open-loop slalom at {speed:g} m/s",
            recordings={source: _recording(source, speed, index) for source in MODELS},
            pairs=(("real", "sim_slip"), ("real", "sim")), tags={"speed_m_s": speed, "control": "open loop"})
        for index, speed in enumerate(SPEEDS)
    )  # fmt: skip
    return Dataset(runs=runs, title="diff-drive slalom speed sweep")


VIEW = Sim2RealView(
    key=KEY,
    ruler=Ruler(stream="pose", label="base position (motion capture)", error=position_error_mm, envelope=ENVELOPE),
    models=("sim_slip", "sim"),
    headline=HEADLINE,
    question="Driven open loop, how far does each model's base drift from the real one, and how soon?",
    overlay=(HEADLINE,),
    notes=(
        "Open loop on purpose: the same wheel-speed commands go to every source, nothing corrects the pose, so the "
        "drift is the model's alone. A wheel-radius error is a heading error, and heading error compounds: compare "
        "the envelope's straight line with the rigid sim's curve."
    ),
    sample=sample_dataset,
)


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    return [(None, {"dataset": sample_dataset()})]


register(Dashboard(
    key=KEY,
    title="Sim-to-real · differential drive",
    category="motion",
    summary="Driven open loop, how far does each model's mobile base drift from the real one, and how soon? Models "
            "ranked on the base position from motion capture, in mm, with a tolerance that grows with time, the "
            "paths from above, then each run's wheels, IMU and pose compared.",
    consumes=(Dataset,),
    inputs={"dataset": "a robot_msgs Dataset: one Run per experiment, recordings real (first) and each model, "
                       "streams wheels (ServoStateTrajectory), imu and pose (PoseTrajectory)"},
    viewers={"rerun": RerunView(blueprint=VIEW.sample_blueprint, log=VIEW.log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n"
        "from robot_msgs.compare import Dataset, Recording, Run\n\n"
        'real = Recording(streams={"wheels": drives, "imu": base_imu, "pose": mocap})\n'
        'run = Run(name="slalom_0.9mps", recordings={"real": real, "sim": sim}, tags={"speed_m_s": 0.9})\n'
        'dash.show("sim2real_diff_drive", dataset=Dataset(runs=(run, ...)))'
    ),
))  # fmt: skip
