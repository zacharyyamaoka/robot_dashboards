"""Sim-to-real · 1-DOF pendulum: how long, and how closely, does each model predict the real pendulum?

    import robot_dashboards as dash

    dash.show("sim2real_pendulum", dataset=runs)       # a robot_msgs Dataset: runs × {real, sim, ..., analytical}

- ``dataset`` (`robot_msgs.compare.Dataset`): one `Run` per experiment, each with a `Recording` per source. The first
  recording is the reference (``real``); every other is a model of it. Streams by name, the same on every side:
  ``servo_state`` (the motor encoder, velocity, torque), ``link_angle`` (the link itself, seen by a camera on an ArUco
  marker: the ruler that does not go through the gearbox), ``imu`` (an IMU on the link's tip).

The page answers first, then opens up (Zach, 2026-11-25: "a bunch of runs … analysis across those runs … drill down
and open each run"):

1. **Answer**: which model tracks the real pendulum best, ranked by a score that favours short horizons; the error of
   each model at 0.25 / 0.5 / 1 / 2 s; how long each stays inside a tolerance that grows with time; the headline run's
   link angle, every source overlaid, and its error against the envelope.
2. **Across runs**: robot_msg_viz's DatasetComparison summary: the worst comparison of every stream, runs ranked.
3. **One tab per run**: its envelope, every stream compared (robot_msg_viz's RunComparison tabs), every stream
   recorded with all sources overlaid (robot_msg_viz's Run view).

WHY a tolerance that grows with time (Zach's note): even a good simulator drifts from reality, and from its own
analytical model, the longer it runs, so a fixed error bound would fail every model at some horizon and say nothing
about the first second, where a model is used. The envelope (`ENVELOPE`) is ``tol(t) = 0.5° + 1.0°/s · t``: 0.5° is
about 3σ of the camera's angle noise, and each second of horizon allows one more degree.

The sample, all physics (seeded, deterministic): a 0.5 kg link, its centre of mass 0.2 m from the pivot, on a 9:1
quasi-direct-drive actuator with 1.1° of gear backlash, motor-side Coulomb + Stribeck friction and an encoder on the
motor. Four runs: open-loop releases from 60° and 90° with the motor unpowered (the natural response; at 90° the
small-angle model is far off), an open-loop torque pulse (a dumb input signal, no feedback), and a closed-loop PD step
to 45° (the feedback case). Four sources per run, recorded at 100 Hz (the camera at 30 Hz):

- ``real``: the two-mass backlash model above, integrated at 0.2 ms, read through its sensors: encoder noise, the
  camera at 30 Hz with 0.15° of noise, the IMU's noise and bias; its logger starts 23 ms before the release, which
  the calibration step (`calibrate`, robot_msgs' `estimate_time_offset` against the rigid sim) estimates at 21-25 ms
  into ``Run.time_offset_s`` (the backlash's free fall at the release biases it a little).
- ``sim_backlash``: a MuJoCo-like rigid-body step (semi-implicit Euler at 2 ms, MuJoCo's default timestep) of the
  same two-mass model with its identified parameters (1.0° of backlash, Coulomb friction only).
- ``sim``: the same integrator with a rigid gear (no backlash).
- ``analytical``: the small-angle linear ODE (sin θ ≈ θ, viscous damping only), no actuator dynamics.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from robot_msgs.compare import Dataset, Recording, Run, estimate_time_offset
from robot_msgs.std import Header
from robot_msgs.trajectory import ImuTrajectory, JointTrajectory, ServoStateTrajectory

from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._sim2real_helpers import Envelope, Ruler, Sim2RealView, joint_error_deg

KEY = "sim2real_pendulum"
JOINT = "pivot"
G = 9.80665
RATE_HZ, CAMERA_HZ, SECONDS = 100.0, 30.0, 4.0  # WHY 100 Hz: the .rrd is per-entity heavy; 200 Hz was +25 %
IMU_RADIUS = 0.25  # m from the pivot
LOGGER_LEAD_S = 0.023  # the real logger starts this long before the release (calibrate() estimates it)
SEED = 7
ENVELOPE = Envelope(floor=0.5, growth=1.0, unit="°")


@dataclass(frozen=True)
class Pendulum:
    """One model of the pendulum and its actuator (SI, angles at the output shaft)."""

    mass: float = 0.5  # kg
    com: float = 0.2  # m, pivot to centre of mass
    link_inertia: float = 0.0222  # kg·m² about the pivot
    motor_inertia: float = 0.0049  # kg·m², the rotor reflected through the 9:1 gear
    link_viscous: float = 0.0006  # N·m·s/rad, the pivot bearing
    motor_viscous: float = 0.0022  # N·m·s/rad
    coulomb: float = 0.034  # N·m, motor side
    stribeck: float = 0.045  # N·m breakaway (= coulomb: no Stribeck)
    stribeck_speed: float = 0.1  # rad/s
    backlash_deg: float = 1.1  # 0 = a rigid gear
    gear_stiffness: float = 250.0  # N·m/rad once the teeth touch
    gear_damping: float = 0.4  # N·m·s/rad in contact
    max_torque: float = 3.0  # N·m

    @property
    def gravity(self) -> float:
        return self.mass * G * self.com


REAL = Pendulum()
# the identified model: close, not equal (a fit is never exact), backlash measured at 1.0°
SIM_BACKLASH = Pendulum(link_inertia=0.0220, motor_inertia=0.0050, link_viscous=0.0005, motor_viscous=0.0020,
                        coulomb=0.032, stribeck=0.032, backlash_deg=1.0, gear_stiffness=300.0, gear_damping=0.5)
SIM = Pendulum(link_inertia=0.0220, motor_inertia=0.0050, link_viscous=0.0005, motor_viscous=0.0020, coulomb=0.032,
               stribeck=0.032, backlash_deg=0.0)  # fmt: skip


# ── the experiments: what the motor is told to do ───────────────────────────────────────────────────────────────────

KP, KD = 3.0, 0.35  # the closed-loop PD on the motor encoder (N·m/rad, N·m·s/rad)
PULSE_NM, PULSE_S = 0.5, 0.25


@dataclass(frozen=True)
class Experiment:
    name: str
    title: str
    start_deg: float
    mode: str  # "release" | "pulse" | "step"
    target_deg: float = 0.0

    def torque(self, t: float, motor: float, motor_rate: float, model: Pendulum) -> float:
        """The motor torque at `t` (s after the release) from the encoder (`motor`, `motor_rate`)."""
        if self.mode == "release" or t < 0.0:
            return 0.0
        if self.mode == "pulse":
            return PULSE_NM if t < PULSE_S else 0.0
        target = math.radians(self.target_deg)
        u = KP * (target - motor) - KD * motor_rate + model.gravity * math.sin(target)
        return max(-model.max_torque, min(model.max_torque, u))


EXPERIMENTS = (
    Experiment("release_60deg", "open-loop release from 60°", 60.0, "release"),
    Experiment("release_90deg", "open-loop release from 90°", 90.0, "release"),
    Experiment("pulse", f"open-loop torque pulse ({PULSE_NM:g} N·m, {PULSE_S * 1000:g} ms)", 0.0, "pulse"),
    Experiment("step_45deg", "closed-loop PD step to 45°", 0.0, "step", 45.0),
)
HEADLINE = "release_60deg"


def _friction(model: Pendulum, rate: float) -> float:
    breakaway = model.coulomb + (model.stribeck - model.coulomb) * math.exp(-((rate / model.stribeck_speed) ** 2))
    return breakaway * math.tanh(rate / 0.02) + model.motor_viscous * rate


def _simulate(model: Pendulum, experiment: Experiment, dt: float, lead_s: float = 0.0) -> dict[str, np.ndarray]:
    """Semi-implicit Euler (MuJoCo's integrator) from `lead_s` before the release to SECONDS after it.

    Returns on every step: t (s after the release), the link angle θ and rate, its acceleration, the motor angle φ
    and rate (output side), the motor torque."""
    steps = round((SECONDS + lead_s) / dt) + 1
    out = {key: np.empty(steps) for key in ("t", "link", "link_rate", "link_accel", "motor", "motor_rate", "torque")}
    gap = math.radians(model.backlash_deg) / 2.0
    theta = phi = math.radians(experiment.start_deg)
    if gap > 0.0 and experiment.start_deg > 0.0:
        # held by hand at the start, the rotor's teeth at the far side of the gap: on release the link falls through
        # the backlash before it picks the rotor up (what a rigid-gear model cannot show)
        phi = theta - gap
    omega = nu = 0.0
    rigid = gap == 0.0
    for k in range(steps):
        t = k * dt - lead_s
        u = experiment.torque(t, phi, nu, model)
        if t < 0.0 and experiment.mode == "release":  # held still until the release
            alpha = 0.0
            out_vals = (t, theta, 0.0, 0.0, phi, 0.0, 0.0)
        elif rigid:
            inertia = model.link_inertia + model.motor_inertia
            alpha = (u - model.gravity * math.sin(theta) - model.link_viscous * omega - _friction(model, omega)) / inertia
            out_vals = (t, theta, omega, alpha, theta, omega, u)
            omega += alpha * dt
            theta += omega * dt
            phi, nu = theta, omega
        else:
            delta = phi - theta
            depth = delta - gap if delta > gap else delta + gap if delta < -gap else 0.0
            gear = model.gear_stiffness * depth + (model.gear_damping * (nu - omega) if depth else 0.0)
            alpha = (gear - model.gravity * math.sin(theta) - model.link_viscous * omega) / model.link_inertia
            motor_alpha = (u - gear - _friction(model, nu)) / model.motor_inertia
            out_vals = (t, theta, omega, alpha, phi, nu, u)
            omega += alpha * dt
            nu += motor_alpha * dt
            theta += omega * dt
            phi += nu * dt
        for key, value in zip(out, out_vals, strict=True):
            out[key][k] = value
    return out


def _analytical(experiment: Experiment) -> dict[str, np.ndarray]:
    """The small-angle linear ODE, (I + J) θ̈ = u − m g l θ − b θ̇, solved with RK4 at 1 ms (exact to plot precision)."""
    model = SIM
    inertia = model.link_inertia + model.motor_inertia
    damping = model.link_viscous + model.motor_viscous
    dt = 0.001
    steps = round(SECONDS / dt) + 1
    theta, omega = math.radians(experiment.start_deg), 0.0

    def accel(th: float, om: float, t: float) -> float:
        return (experiment.torque(t, th, om, model) - model.gravity * th - damping * om) / inertia

    out = {key: np.empty(steps) for key in ("t", "link", "link_rate", "link_accel", "motor", "motor_rate", "torque")}
    for k in range(steps):
        t = k * dt
        a = accel(theta, omega, t)
        for key, value in zip(out, (t, theta, omega, a, theta, omega, experiment.torque(t, theta, omega, model)),
                              strict=True):  # fmt: skip
            out[key][k] = value
        k1 = (omega, a)
        k2 = (omega + 0.5 * dt * k1[1], accel(theta + 0.5 * dt * k1[0], omega + 0.5 * dt * k1[1], t + 0.5 * dt))
        k3 = (omega + 0.5 * dt * k2[1], accel(theta + 0.5 * dt * k2[0], omega + 0.5 * dt * k2[1], t + 0.5 * dt))
        k4 = (omega + dt * k3[1], accel(theta + dt * k3[0], omega + dt * k3[1], t + dt))
        theta += dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        omega += dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return out


# ── the sensors: what each source records ───────────────────────────────────────────────────────────────────────────


def _at(state: dict[str, np.ndarray], t: np.ndarray, key: str) -> np.ndarray:
    return np.interp(t, state["t"], state[key])


def _recording(state: dict[str, np.ndarray], source: str, run_index: int, clock_s: float = 0.0) -> Recording:
    """The three streams `source` records, on its own clock (`clock_s` = its clock's reading at the release)."""
    real = source == "real"
    noise = np.random.default_rng(SEED + 31 * run_index + (1 if real else 0))
    clock = np.arange(round(SECONDS * RATE_HZ) + 1) / RATE_HZ  # what the logger stamps
    t = clock - clock_s  # physical time since the release at each stamp
    motor = _at(state, t, "motor")
    motor_rate = _at(state, t, "motor_rate")
    torque = _at(state, t, "torque")
    if real:
        motor = motor + np.radians(noise.normal(0.0, 0.005, motor.shape))
        motor_rate = np.gradient(motor, clock)  # the drive differentiates its encoder
        torque = torque + noise.normal(0.0, 0.015, torque.shape)  # current-sensed torque
    servo = ServoStateTrajectory(t=clock, joint_names=(JOINT,), positions=motor[:, None], velocities=motor_rate[:, None],
                                 effort=torque[:, None], header=Header(stamp=0.0, frame_id="base_link"))  # fmt: skip

    frames = np.arange(int(SECONDS * CAMERA_HZ) + 1) / CAMERA_HZ
    link = _at(state, frames - clock_s, "link")
    if real:
        link = link + np.radians(noise.normal(0.0, 0.15, link.shape))
    # WHY velocities (differenced frames, as a tracker reports them): robot_msg_viz's Run view of a JointTrajectory
    # always lays out a velocity plot, and an empty plot fails the contract
    camera = JointTrajectory(t=frames, joint_names=(JOINT,), positions=link[:, None],
                             velocities=np.gradient(link, frames)[:, None],
                             header=Header(stamp=0.0, frame_id="camera", meta={"sensor": "ArUco on the link, 30 Hz"}))

    theta, rate, accel = (_at(state, t, key) for key in ("link", "link_rate", "link_accel"))
    gyro = np.column_stack([np.zeros_like(rate), np.zeros_like(rate), rate])
    specific = np.column_stack([-IMU_RADIUS * rate**2 - G * np.cos(theta), IMU_RADIUS * accel + G * np.sin(theta),
                                np.zeros_like(theta)])  # fmt: skip
    if real:
        gyro = gyro + noise.normal(0.0, 0.003, gyro.shape) + [0.0, 0.0, 0.002]
        specific = specific + noise.normal(0.0, 0.05, specific.shape) + [0.04, -0.03, 0.0]
    half = theta / 2.0
    quaternions = np.column_stack([np.zeros_like(half), np.zeros_like(half), np.sin(half), np.cos(half)])
    imu = ImuTrajectory(t=clock, orientations=quaternions, angular_velocities=gyro, linear_accelerations=specific,
                        header=Header(stamp=0.0, frame_id="imu_link"))  # fmt: skip
    meta = {
        "real": {"rig": "pendulum bench", "actuator": "9:1 QDD"},
        "sim_backlash": {"engine": "semi-implicit Euler 2 ms", "actuator": "two-mass, 1.0° backlash"},
        "sim": {"engine": "semi-implicit Euler 2 ms", "actuator": "rigid gear"},
        "analytical": {"model": "small-angle linear ODE"},
    }[source]
    return Recording(streams={"servo_state": servo, "link_angle": camera, "imu": imu}, header=Header(meta=meta))


SOURCES = ("real", "sim_backlash", "sim", "analytical")
# real against each model (the ranking); on the headline run also sim against the analytical model ("even the
# simulator diverges from the analytical model"). WHY only there: each comparison is ~70 KB of .rrd (60 were 10 MB)
PAIRS = (("real", "sim_backlash"), ("real", "sim"), ("real", "analytical"))
MODEL_PAIR = ("sim", "analytical")


def calibrate(real: Recording, model: Recording, window_s: float = 0.5) -> float:
    """How late the real clock runs against `model`'s (s), from the encoder over the first `window_s` after the
    start: robot_msgs' `estimate_time_offset` (velocity cross-correlation, then position-RMS refinement).

    WHY a short window: over a long run the model's own drift (what this dashboard measures) would pull the estimate."""

    def joints(recording: Recording) -> JointTrajectory:
        servo = recording.streams["servo_state"]
        keep = np.asarray(servo.t) <= window_s
        return JointTrajectory(t=np.asarray(servo.t)[keep], joint_names=servo.joint_names,
                               positions=np.asarray(servo.positions)[keep])  # fmt: skip

    return float(estimate_time_offset(joints(model), joints(real), max_offset_s=0.1).offset_s)


@functools.cache
def _run(index: int) -> Run:
    experiment = EXPERIMENTS[index]
    states = {
        "real": _simulate(REAL, experiment, 0.0002, lead_s=LOGGER_LEAD_S),
        "sim_backlash": _simulate(SIM_BACKLASH, experiment, 0.002),
        "sim": _simulate(SIM, experiment, 0.002),
        "analytical": _analytical(experiment),
    }
    recordings = {source: _recording(states[source], source, index, LOGGER_LEAD_S if source == "real" else 0.0)
                  for source in SOURCES}  # fmt: skip
    offset = calibrate(recordings["real"], recordings["sim"])
    tags = {"mode": "closed loop" if experiment.mode == "step" else "open loop", "input": experiment.mode,
            "start_deg": experiment.start_deg}  # fmt: skip
    if experiment.mode == "step":
        tags["target_deg"] = experiment.target_deg
    pairs = (*PAIRS, MODEL_PAIR) if experiment.name == HEADLINE else PAIRS
    return Run(name=experiment.name, title=experiment.title, recordings=recordings, pairs=pairs, tags=tags,
               time_offset_s={"real": round(offset, 4)})  # fmt: skip


@functools.cache
def sample_dataset() -> Dataset:
    return Dataset(runs=tuple(_run(i) for i in range(len(EXPERIMENTS))), title="pendulum: real vs three models")


VIEW = Sim2RealView(
    key=KEY,
    ruler=Ruler(stream="link_angle", label="link angle (camera on an ArUco marker)", error=joint_error_deg,
                envelope=ENVELOPE),
    models=("sim_backlash", "sim", "analytical"),
    headline=HEADLINE,
    question="Which model of the pendulum tracks the real one best, and for how long?",
    overlay=(HEADLINE,),  # WHY the headline only: each run's overlay is ~0.8 MB of .rrd
    notes=(
        "Open loop first (the natural response, no controller to hide a model error), then a closed-loop step "
        "(the feedback case). Clocks calibrated before comparing: the real logger's lead is estimated from the "
        "encoder (`estimate_time_offset`, first 0.5 s) into each run's `time_offset_s`."
    ),
    sample=sample_dataset,
)


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    return [(None, {"dataset": sample_dataset()})]


register(Dashboard(
    key=KEY,
    title="Sim-to-real · pendulum",
    category="motion",
    summary="Which model of the 1-DOF pendulum (sim with backlash, rigid sim, the analytical ODE) tracks the real "
            "one best, and for how long? Models ranked by a short-horizon score, the error at 0.25 / 0.5 / 1 / 2 s, "
            "time inside a tolerance that grows with time, then each run: every sensor stream compared and overlaid.",
    consumes=(Dataset,),
    inputs={"dataset": "a robot_msgs Dataset: one Run per experiment, recordings real (first) and each model, "
                       "streams servo_state, link_angle, imu"},
    viewers={"rerun": RerunView(blueprint=VIEW.sample_blueprint, log=VIEW.log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n"
        "from robot_msgs.compare import Dataset, Recording, Run\n\n"
        "# one Run per experiment; the first recording (real) is the reference, every other a model of it\n"
        'run = Run(name="release_60deg", recordings={"real": real, "sim": sim, "analytical": ode})\n'
        'dash.show("sim2real_pendulum", dataset=Dataset(runs=(run, ...)))'
    ),
))  # fmt: skip
