"""Babble round 2026-11-28: five read-only Rerun "robot viewer" layouts over one robot-description tree.

    env -u VIRTUAL_ENV uv run --directory /home/bam/robot_dashboards python docs/proposals/robot-viewer-2026-11-28/build.py

Writes reports/media/robot-viewer-2026-11-28/rrd/<variant>-<robot>.rrd (5 variants x 3 robots, each with its
blueprint inside) and rrd/tabs/<variant>-<robot>-<tab>.rrd (the same recording opened on another tab, for stills).

Zach (2026-11-28): "a viewer that just streams the stuff ... for all of our actuators we want to see the commanded,
and states, and all the sensors we want to see the measurements, and then also ... the configurations ... Its really
like tview ... read only ... a 3d view on the left, and then tabs ... follow the structure of the robotdescription ...
as you compose the robotdescription so should their visualization compose."

Exploration only: nothing here is registered as a dashboard. The recording is identical across the five variants
(same entities, same samples); only the blueprint differs, so the comparison is the layout and nothing else.

Entity layout (shared by every variant), one subtree per part, at its cell-unique path (``right_arm/elbow``):
  /cell/<servo>/position/{cmd,state,error}   degrees     /cell/<servo>/velocity/{cmd,state}  deg/s
  /cell/<servo>/effort/{cmd,state}           N·m         /cell/<servo>/temperature  °C   /cell/<servo>/fault  code
  /cell/<part>/config                        markdown (static, read-only)
  /cell/<camera>/image                       JPEG frames  /world/...  the 3D scene  /now  the tview-style table
WHY scalars under the part's own path rather than robot_msg_viz's /<field>/<column>/<name>: the whole question of
this round is whether the viewer follows the description tree, and that needs every part's signals under its path.
A winner would grow a tree-aware ServoCommand/ServoState view in robot_msg_viz instead of this file.

Simulated, honestly: the UR5e motion is the real bam_ruckig trajectory in assets/; commands are that trajectory,
states are it delayed 90 ms with gravity sag and noise (a real tracking loop is not run); the pendulum's state comes
from a PD loop integrated at 1 kHz; camera frames are a pinhole projection of the same stick geometry; temperatures,
faults and the configs are invented sample numbers. Nothing opens a bus; there is no command path.
"""

from __future__ import annotations

import io
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image as PILImage, ImageDraw
from robot_msgs._core import Msg
from robot_msgs.control import ServoCommand, ServoState
from robot_msgs.robot.robot_config import RobotConfig
from robot_msgs.sensor import CameraInfo, CameraSettings
from robot_msgs.std import Header

from robot_dashboards.assets import UR5E_TRAJECTORY, UR5E_URDF
from robot_dashboards.dashboards._motion_state_helpers import eye, forward_kinematics, log_arm, pose_arm, short_joint

REPO = Path(__file__).resolve().parents[3]
OUT = Path("/home/bam/robot_dashboards/reports/media/robot-viewer-2026-11-28/rrd")
RATE_HZ, SECONDS, CAMERA_HZ = 25.0, 5.0, 10.0
DEG = 180.0 / math.pi
WARN_C = 60.0
FAULT_NAMES = {33: "motor driver fault", 39: "outside limit"}
# one colour per joint index (state solid, command the same hue as dots), Rerun-dark friendly
PALETTE = [(255, 120, 80), (250, 200, 60), (90, 210, 120), (70, 180, 250), (180, 130, 255), (255, 110, 190)]


# ── the robot description: a tree of parts ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, eq=False, repr=False, kw_only=True)
class ServoConfig(Msg):
    """Proposal-local: a servo's board config (robot_msgs has no such message yet; RobotConfig wants a Msg)."""

    kp: float  # N·m/rad
    kd: float  # N·m·s/rad
    max_current: float  # A
    min_position: float  # rad
    max_position: float  # rad
    bus: str
    can_id: int
    header: Header = field(default_factory=Header)


@dataclass(frozen=True)
class Part:
    """One node of the robot description: a group (an arm, a cell), a servo, a camera or an IMU."""

    name: str
    kind: str  # group | servo | camera | imu
    children: tuple[Part, ...] = ()
    config: Msg | None = None
    info: CameraInfo | None = None  # cameras only
    joint: str | None = None  # servos: the URDF joint it drives


def walk(part: Part, prefix: str = "") -> list[tuple[str, Part]]:
    """(cell-unique path, part) for every node below `part` (the root itself is not named in paths)."""
    out = []
    for child in part.children:
        path = f"{prefix}{child.name}"
        out.append((path, child))
        out += walk(child, path + "/")
    return out


def servos_of(part: Part, prefix: str = "") -> list[tuple[str, Part]]:
    return [(p, c) for p, c in walk(part, prefix) if c.kind == "servo"]


def cameras_of(part: Part, prefix: str = "") -> list[tuple[str, Part]]:
    return [(p, c) for p, c in walk(part, prefix) if c.kind == "camera"]


def servo_groups(root: Part) -> list[tuple[str, list[tuple[str, Part]]]]:
    """Every node whose direct children include servos, with those servos: (path or root name, [(path, servo)])."""
    groups = []
    direct = [(c.name, c) for c in root.children if c.kind == "servo"]
    if direct:
        groups.append((root.name, direct))
    for path, node in walk(root):
        mine = [(f"{path}/{c.name}", c) for c in node.children if c.kind == "servo"]
        if mine:
            groups.append((path, mine))
    return groups


def robot_config(root: Part) -> RobotConfig:
    return RobotConfig(components={path: part.config for path, part in walk(root) if part.config is not None})


def _camera(name: str, w: int, h: int, fov_deg: float, exposure_us: float, fps: float) -> Part:
    f = 0.5 * w / math.tan(math.radians(fov_deg) / 2)
    info = CameraInfo(width=w, height=h, k=np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]))
    settings = CameraSettings(exposure_time=exposure_us * 1e-6, gain=4.0, frame_rate=fps, auto_exposure=False)
    return Part(name, "camera", config=settings, info=info)


UR5E_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
               "wrist_3_joint")  # fmt: skip
_UR_GAINS = [(400, 8, 20), (400, 8, 20), (250, 5, 14), (60, 1.2, 6), (60, 1.2, 6), (40, 0.8, 6)]


def ur5e_arm(name: str, bus: str) -> Part:
    """The UR5e as a description subtree: six servos and its wrist camera. Composing a cell composes this."""
    servos = tuple(
        Part(short_joint(j).replace("_joint", ""), "servo", joint=j,
             config=ServoConfig(kp=kp, kd=kd, max_current=imax, min_position=-2 * math.pi if i != 2 else -math.pi,
                                max_position=2 * math.pi if i != 2 else math.pi, bus=bus, can_id=i + 1))
        for i, (j, (kp, kd, imax)) in enumerate(zip(UR5E_JOINTS, _UR_GAINS, strict=True))
    )  # fmt: skip
    return Part(name, "group", servos + (_camera("wrist_camera", 256, 192, 70, 4000, 10),))


ROBOTS: dict[str, Part] = {
    "pendulum": Part("pendulum_rig", "group", (
        Part("pendulum", "servo", joint="pivot",
             config=ServoConfig(kp=4.0, kd=0.08, max_current=6.0, min_position=-math.pi / 2,
                                max_position=math.pi / 2, bus="can11", can_id=11)),
        _camera("side_camera", 256, 192, 60, 8000, 10),
    )),
    "ur5e": Part("ur5e_cell", "group", (ur5e_arm("arm", "can0"),)),
    "dual_arm": Part("dual_arm_cell", "group", (ur5e_arm("left_arm", "can0"), ur5e_arm("right_arm", "can1"),
                                                _camera("overhead_camera", 256, 192, 75, 6000, 10))),
}  # fmt: skip
ARM_BASE_Y = {"arm": 0.0, "left_arm": 0.42, "right_arm": -0.42}


# ── the samples ────────────────────────────────────────────────────────────────────────────────────────

T = np.round(np.arange(0.0, SECONDS + 1e-9, 1.0 / RATE_HZ), 6)


@dataclass
class ServoTrace:
    cmd_q: np.ndarray
    q: np.ndarray
    cmd_qd: np.ndarray
    qd: np.ndarray
    cmd_eff: np.ndarray
    eff: np.ndarray
    temp: np.ndarray
    fault: np.ndarray


def _ur5e_motion(shift_s: float, mirror: bool) -> dict[str, np.ndarray]:
    data = np.load(UR5E_TRAJECTORY)
    names = [str(n) for n in data["names"]]
    local = np.clip(T - 0.5 - shift_s, data["t"][0], data["t"][-1])
    q = np.stack([np.interp(local, data["t"], data["q"][:, names.index(j)]) for j in UR5E_JOINTS], axis=1)
    if mirror:
        q[:, 0] = -q[:, 0]
    return {j: q[:, i] for i, j in enumerate(UR5E_JOINTS)}


def _arm_traces(arm: str, rng: np.random.Generator) -> dict[str, ServoTrace]:
    motion = _ur5e_motion(shift_s=0.6 if arm == "left_arm" else 0.0, mirror=arm == "left_arm")
    traces = {}
    hot = {"arm": "wrist_3_joint", "right_arm": "wrist_3_joint", "left_arm": None}[arm]
    for i, joint in enumerate(UR5E_JOINTS):
        cmd = motion[joint]
        delayed = np.interp(T - 0.09, T, cmd)  # the 90 ms tracking lag
        sag = {1: -1.6, 2: -1.1, 3: -0.5}.get(i, 0.0) / DEG
        q = delayed + sag + rng.normal(0, 0.04 / DEG, len(T))
        cmd_qd, qd = np.gradient(cmd, T), np.gradient(q, T)
        q2, q3 = motion["shoulder_lift_joint"], motion["elbow_joint"]
        ff = {1: 38 * np.cos(q2) + 9 * np.cos(q2 + q3), 2: 9 * np.cos(q2 + q3)}.get(i, 0.4 * np.cos(cmd))
        kp = _UR_GAINS[i][0]
        eff = ff + kp * (cmd - q) * 0.1 + rng.normal(0, 0.2, len(T))
        rate = 0.3 if joint == hot else 0.03 + 0.01 * i
        temp = np.round((58.9 if joint == hot else 36 + 2.5 * i) + rate * T + rng.normal(0, 0.03, len(T)), 1)
        fault = np.zeros(len(T))
        if arm == "right_arm" and joint == "elbow_joint":  # the driver faults at 2.6 s: state stops, command goes on
            k = int(np.searchsorted(T, 2.6))
            q[k:] = q[k] - np.linspace(0, 6.0 / DEG, len(T) - k)  # it sags as it goes limp
            qd[k:], eff[k:], fault[k:] = 0.0, 0.0, 33
        traces[short_joint(joint)] = ServoTrace(cmd, q, cmd_qd, qd, ff, eff, temp, fault)
    return traces


def _pendulum_trace() -> ServoTrace:
    """A PD loop on a real pendulum (m 0.5 kg, l 0.35 m), integrated at 1 kHz: it lags and sags under gravity."""
    m, l, b, kp, kd = 0.5, 0.35, 0.01, 4.0, 0.08
    inertia, dt = m * l * l, 1e-3
    fine = np.arange(0, SECONDS + dt, dt)
    ramp = np.clip(fine / 0.8, 0, 1)
    cmd = math.radians(35) * np.sin(2 * math.pi * 0.5 * fine) * ramp
    cmd_qd = np.gradient(cmd, fine)
    theta, omega, out_q, out_qd, out_u = 0.0, 0.0, [], [], []
    for k in range(len(fine)):
        u = float(np.clip(kp * (cmd[k] - theta) + kd * (cmd_qd[k] - omega), -1.2, 1.2))
        alpha = (u - m * 9.81 * l * math.sin(theta) - b * omega) / inertia
        omega += alpha * dt
        theta += omega * dt
        out_q.append(theta), out_qd.append(omega), out_u.append(u)
    pick = lambda a: np.interp(T, fine, np.asarray(a))  # noqa: E731
    ff = m * 9.81 * l * np.sin(pick(cmd))
    temp = np.round(41.0 + 0.8 * T + 0.0 * T, 1)
    return ServoTrace(pick(cmd), pick(out_q), pick(cmd_qd), pick(out_qd), ff, pick(out_u), temp, np.zeros(len(T)))


def traces_for(robot: str) -> dict[str, ServoTrace]:
    rng = np.random.default_rng(20261128)
    if robot == "pendulum":
        return {"pendulum": _pendulum_trace()}
    out = {}
    for arm in [c.name for c in ROBOTS[robot].children if c.kind == "group"]:
        out.update({f"{arm}/{j}": tr for j, tr in _arm_traces(arm, rng).items()})
    return out


def messages(robot: str, traces: dict[str, ServoTrace], k: int) -> tuple[ServoCommand, ServoState]:
    """The tick as the robot would publish it: one ServoCommand and one ServoState over cell-unique names."""
    names = tuple(traces)
    col = lambda attr: np.array([getattr(traces[n], attr)[k] for n in names])  # noqa: E731
    header = Header(stamp=1_795_867_200.0 + float(T[k]))
    command = ServoCommand(name=names, position=col("cmd_q"), velocity=col("cmd_qd"),
                           feedforward_effort=col("cmd_eff"), header=header)  # fmt: skip
    state = ServoState(name=names, position=col("q"), velocity=col("qd"), effort=col("eff"),
                       temperature=col("temp"), fault=col("fault"), seq=k, header=header)  # fmt: skip
    return command, state


# ── the 3D scene and the camera frames (a pinhole projection of the same geometry) ─────────────────────

# items on the floor around each arm's base (radius m, azimuth deg): where the sample trajectory's tool swings
ITEMS = [((0.45, 60), (230, 80, 60)), ((0.5, 0), (60, 160, 230)), ((0.45, -80), (240, 200, 50)),
         ((0.5, -140), (120, 210, 110))]  # fmt: skip


def _chain(fk: dict[str, np.ndarray], k: int, base_y: float) -> np.ndarray:
    order = ["base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
             "wrist_3_link", "flange", "tool0"]  # fmt: skip
    pts = [fk[link][k, :3, 3] for link in order if link in fk]
    return np.array(pts) + np.array([0.0, base_y, 0.0])


def _look_at(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """camera-to-world rotation, RDF (x right, y down, z forward)."""
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def _render(info: CameraInfo, rotation: np.ndarray, position: np.ndarray, strokes: list, items: list,
            sky: tuple[int, int, int]) -> np.ndarray:  # fmt: skip
    w, h = info.width, info.height
    img = PILImage.new("RGB", (w, h), sky)
    draw = ImageDraw.Draw(img)
    k = np.asarray(info.k)

    def project(points: np.ndarray) -> list[tuple[float, float]] | None:
        cam = (np.asarray(points) - position) @ rotation
        if np.any(cam[:, 2] < 0.03):
            return None
        uv = (cam @ k.T)[:, :2] / cam[:, 2:3]
        return [tuple(p) for p in uv]

    for x in np.linspace(-1.0, 1.4, 13):  # the floor grid
        for line in (np.array([[x, -1.2, 0], [x, 1.2, 0]]), np.array([[-1.0, x - 0.2, 0], [1.4, x - 0.2, 0]])):
            seg = project(np.linspace(line[0], line[1], 24))
            if seg:
                draw.line(seg, fill=(90, 96, 108), width=1)
    for (cx, cy), colour in items:
        corners = np.array([[cx - 0.035, cy - 0.035, 0], [cx + 0.035, cy - 0.035, 0], [cx + 0.035, cy + 0.035, 0],
                            [cx - 0.035, cy + 0.035, 0]])  # fmt: skip
        poly = project(corners)
        if poly:
            draw.polygon(poly, fill=colour)
    for points, colour, width in strokes:
        poly = project(points)
        if poly:
            draw.line(poly, fill=colour, width=width, joint="curve")
    return np.asarray(img)


@dataclass
class Scene:
    eye: rrb.EyeControls3D
    log_static: Callable[[rr.RecordingStream], None]
    log_tick: Callable[[rr.RecordingStream, int], None]
    frames: dict[str, list[np.ndarray]]  # camera path -> frames at CAMERA_HZ


def scene_for(robot: str, traces: dict[str, ServoTrace]) -> Scene:
    root = ROBOTS[robot]
    cams = dict(cameras_of(root))
    every = int(RATE_HZ / CAMERA_HZ)
    cam_ticks = list(range(0, len(T), every))
    frames: dict[str, list[np.ndarray]] = {p: [] for p in cams}
    cam_pose: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {p: [] for p in cams}

    if robot == "pendulum":
        pivot = np.array([0.0, -0.07, 0.9])  # on a short axle in front of the post
        q = traces["pendulum"].q
        bob = lambda k: pivot + 0.35 * np.array([math.sin(q[k]), 0.0, -math.cos(q[k])])  # noqa: E731
        side_pos = np.array([0.0, -1.1, 0.7])
        side_rot = _look_at(side_pos, np.array([0.0, 0.0, 0.7]), np.array([0, 0, 1.0]))
        for k in cam_ticks:
            strokes = [(np.array([[0, 0, 0], [0, 0, 0.92]]), (150, 155, 165), 4),
                       (np.array([pivot, bob(k)]), (240, 240, 245), 3),
                       (np.array([bob(k) + [0.02, 0, 0], bob(k) - [0.02, 0, 0]]), (255, 120, 80), 9)]  # fmt: skip
            frames["side_camera"].append(_render(cams["side_camera"].info, side_rot, side_pos, strokes, [],
                                                 (36, 40, 48)))  # fmt: skip
            cam_pose["side_camera"].append((side_rot, side_pos))

        def log_static(rec: rr.RecordingStream) -> None:
            rec.log("/world/rig/stand", rr.LineStrips3D([[[0, 0, 0], [0, 0, 0.92]]], radii=0.012,
                                                        colors=[(150, 155, 165)]), static=True)  # fmt: skip
            rec.log("/world/rig/axle", rr.LineStrips3D([[[0, 0, 0.9], [0, -0.07, 0.9]]], radii=0.012,
                                                       colors=[(150, 155, 165)]), static=True)  # fmt: skip
            rec.log("/world/rig/base", rr.Boxes3D(centers=[[0, 0, 0.01]], half_sizes=[[0.15, 0.12, 0.01]],
                                                  colors=[(90, 95, 105)], fill_mode="solid"), static=True)  # fmt: skip

        def log_tick(rec: rr.RecordingStream, k: int) -> None:
            rec.log("/world/rig/rod", rr.LineStrips3D([[pivot, bob(k)]], radii=0.008, colors=[(235, 235, 240)]))
            rec.log("/world/rig/bob", rr.Points3D([bob(k)], radii=0.035, colors=[(255, 120, 80)]))

        look = eye(np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, 1.3]]), distance_radii=1.9)
    else:
        arms = [c.name for c in root.children if c.kind == "group"]
        fks = {}
        for arm in arms:
            q = np.stack([traces[f"{arm}/{short_joint(j)}"].q for j in UR5E_JOINTS], axis=1)
            fks[arm] = forward_kinematics(UR5E_URDF, UR5E_JOINTS, q)
        items = []
        for a in arms:
            sign = -1.0 if a == "left_arm" else 1.0  # the left arm's shoulder_pan is mirrored
            for (radius, azimuth), c in ITEMS:
                y = radius * math.sin(math.radians(sign * azimuth))
                if len(arms) > 1 and y * np.sign(ARM_BASE_Y[a]) < -0.2:
                    continue  # it would sit under the other arm
                items.append(((radius * math.cos(math.radians(sign * azimuth)),
                               ARM_BASE_Y[a] + radius * math.sin(math.radians(sign * azimuth))), c))
        for k in cam_ticks:
            strokes = [(_chain(fks[a], k, ARM_BASE_Y[a]), (200, 205, 215), 5) for a in arms]
            for path, cam in cams.items():
                if path.endswith("wrist_camera"):
                    arm = path.split("/")[0]
                    flange = fks[arm][tip_name(fks[arm])][k]
                    position = flange[:3, 3] + np.array([0, ARM_BASE_Y[arm], 0]) + 0.06 * flange[:3, 1]
                    # WHY pitched down: the camera is mounted looking ~40° below the flange axis, at the workspace
                    forward = flange[:3, 2] - np.array([0.0, 0.0, 0.85])
                    rot = _look_at(position, position + forward, np.array([0.0, 0.0, 1.0]))
                    own = [s for s in strokes if s is not strokes[arms.index(arm)]]
                    image = _render(cam.info, rot, position, own, items, (44, 46, 52))
                else:
                    position = np.array([0.0, 0.0, 2.3])
                    rot = _look_at(position, np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0, 0]))
                    image = _render(cam.info, rot, position, strokes, items, (30, 33, 40))
                frames[path].append(image)
                cam_pose[path].append((rot, position))

        def log_static(rec: rr.RecordingStream) -> None:
            for arm in arms:
                rec.log(f"/world/{arm}", rr.Transform3D(translation=[0.0, ARM_BASE_Y[arm], 0.0]), static=True)
                log_arm(rec, UR5E_URDF, arm)
            rec.log("/world/items", rr.Boxes3D(centers=[[x, y, 0.02] for (x, y), _ in items],
                                               half_sizes=[[0.035, 0.035, 0.02]] * len(items),
                                               colors=[c for _, c in items], fill_mode="solid"), static=True)  # fmt: skip

        def log_tick(rec: rr.RecordingStream, k: int) -> None:
            for arm in arms:
                tree = log_arm(rec, UR5E_URDF, arm)
                q = [traces[f"{arm}/{short_joint(j)}"].q[k] for j in UR5E_JOINTS]
                pose_arm(rec, tree, arm, UR5E_JOINTS, np.asarray(q))

        reach = np.vstack([_chain(fks[a], k, ARM_BASE_Y[a]) for a in arms for k in range(0, len(T), 5)]
                          + [[[x, y, 0.0] for (x, y), _ in items]])  # fmt: skip
        look = eye(reach, distance_radii=3.0 if len(arms) > 1 else 3.2, elevation=0.75)

    def log_cameras(rec: rr.RecordingStream, k: int) -> None:
        if k % every:
            return
        i = k // every
        for path, cam in cams.items():
            rot, position = cam_pose[path][i]
            rec.log(f"/world/cameras/{path}", rr.Transform3D(translation=position, mat3x3=rot))
            rec.log(f"/cell/{path}/image", rr.Image(frames[path][i]).compress(jpeg_quality=82))

    def log_static_all(rec: rr.RecordingStream) -> None:
        log_static(rec)
        for path, cam in cams.items():
            rec.log(f"/world/cameras/{path}", rr.Pinhole(image_from_camera=np.asarray(cam.info.k),
                                                         resolution=[cam.info.width, cam.info.height],
                                                         camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.12,
                                                         color=(120, 200, 255)), static=True)  # fmt: skip
            rec.log(f"/cell/{path}/image", rr.Pinhole(image_from_camera=np.asarray(cam.info.k),
                                                      resolution=[cam.info.width, cam.info.height]), static=True)

    def log_tick_all(rec: rr.RecordingStream, k: int) -> None:
        log_tick(rec, k)
        log_cameras(rec, k)

    return Scene(look, log_static_all, log_tick_all, frames)


def tip_name(fk: dict[str, np.ndarray]) -> str:
    for link in ("tool0", "flange", "wrist_3_link"):
        if link in fk:
            return link
    raise KeyError("no tip link")


# ── the text: per-part config (static) and the tview-style "now" table ─────────────────────────────────

def _servo_config_md(path: str, cfg: ServoConfig) -> str:
    return (f"**{path}** · servo · {cfg.bus} id {cfg.can_id}\n\n| kp N·m/rad | kd N·m·s/rad | max current | limits |\n"
            f"|---:|---:|---:|---|\n| {cfg.kp:g} | {cfg.kd:g} | {cfg.max_current:g} A | "
            f"{cfg.min_position * DEG:+.0f}° … {cfg.max_position * DEG:+.0f}° |")  # fmt: skip


def _camera_config_md(path: str, cam: Part) -> str:
    s: CameraSettings = cam.config  # type: ignore[assignment]
    fov = 2 * math.degrees(math.atan(0.5 * cam.info.width / cam.info.k[0][0]))
    return (f"**{path}** · camera\n\n| resolution | fps | exposure | gain | h-FOV |\n|---|---:|---:|---:|---:|\n"
            f"| {cam.info.width}×{cam.info.height} | {s.frame_rate:g} | {s.exposure_time * 1e3:g} ms | "
            f"{s.gain:g} | {fov:.0f}° |")  # fmt: skip


def _parts_config_md(title: str, items: list[tuple[str, Part]]) -> str:
    """One table for the servos in `items`, then one for its cameras: the Config tab's text."""
    servos = [(p, c) for p, c in items if c.kind == "servo"]
    cams = [(p, c) for p, c in items if c.kind == "camera"]
    lines = [f"### {title}", ""]
    if servos:
        lines += ["| servo | bus/id | kp | kd | max A | limits |", "|---|---|---:|---:|---:|---|"]
        for path, part in servos:
            c: ServoConfig = part.config  # type: ignore[assignment]
            lines.append(f"| {path} | {c.bus}/{c.can_id} | {c.kp:g} | {c.kd:g} | {c.max_current:g} | "
                         f"{c.min_position * DEG:+.0f}…{c.max_position * DEG:+.0f}° |")  # fmt: skip
        lines.append("")
    if cams:
        lines += ["| camera | resolution | fps | exposure | gain |", "|---|---|---:|---:|---:|"]
        for path, cam in cams:
            s: CameraSettings = cam.config  # type: ignore[assignment]
            lines.append(f"| {path} | {cam.info.width}×{cam.info.height} | {s.frame_rate:g} | "
                         f"{s.exposure_time * 1e3:g} ms | {s.gain:g} |")  # fmt: skip
    return "\n".join(lines)


def _group_config_md(title: str, root: Part, prefix: str) -> str:
    return _parts_config_md(title, walk(root, prefix))


def config_docs(root: Part) -> list[str]:
    """The Config tab's documents: one per top-level group, plus the cell's own direct parts (if any)."""
    docs = [c.name for c in root.children if c.kind == "group"]
    if any(c.kind != "group" for c in root.children):
        docs.append(root.name)
    return docs


def _now_md(root: Part, command: ServoCommand, state: ServoState, k: int) -> str:
    """tview-style: every servo's command, state, error, velocity, effort, temperature and fault at the playhead."""
    index = {n: i for i, n in enumerate(state.name)}
    lines = [f"**{root.name}** · t = {T[k]:.2f} s · read-only", ""]
    for group, servos in servo_groups(root):
        lines += [f"**{group}**", "", "| servo | cmd ° | state ° | err ° | vel °/s | effort N·m | temp °C | fault |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
        for path, _ in servos:
            i = index[path]
            err = (command.position[i] - state.position[i]) * DEG
            temp = state.temperature[i]
            fault = int(state.fault[i])
            lines.append(
                f"| {path.split('/')[-1]} | {command.position[i] * DEG:+7.1f} | {state.position[i] * DEG:+7.1f} | "
                f"{'**' if abs(err) > 2 else ''}{err:+.2f}{'**' if abs(err) > 2 else ''} | "
                f"{state.velocity[i] * DEG:+.0f} | {state.effort[i]:+.1f} | "
                f"{'⚠ **' if temp >= WARN_C else ''}{temp:.1f}{'**' if temp >= WARN_C else ''} | "
                f"{'⛔ **' + str(fault) + ' ' + FAULT_NAMES.get(fault, '') + '**' if fault else 'ok'} |")  # fmt: skip
        lines.append("")
    cams = cameras_of(root)
    if cams:
        lines += ["| camera | frame | resolution | fps | exposure |", "|---|---:|---|---:|---:|"]
        for path, cam in cams:
            s: CameraSettings = cam.config  # type: ignore[assignment]
            lines.append(f"| {path} | {k // int(RATE_HZ / CAMERA_HZ)} | {cam.info.width}×{cam.info.height} | "
                         f"{s.frame_rate:g} | {s.exposure_time * 1e3:g} ms |")  # fmt: skip
    lines += [""]
    lines += [_group_config_md(root.name, root, "")]
    return "\n".join(lines)


# ── the recording (identical for every variant) ────────────────────────────────────────────────────────

def _color(path: str, root: Part) -> tuple[int, int, int]:
    for _, servos in servo_groups(root):
        for i, (p, _) in enumerate(servos):
            if p == path:
                return PALETTE[i % len(PALETTE)]
    return PALETTE[0]


def _tint(c: tuple[int, int, int], f: float = 0.55) -> tuple[int, int, int]:
    return tuple(int(v + (255 - v) * f) for v in c)  # type: ignore[return-value]


def write(path: Path, robot: str, blueprint: rrb.Blueprint) -> Path:
    root = ROBOTS[robot]
    traces = traces_for(robot)
    scene = scene_for(robot, traces)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = rr.RecordingStream(f"robot_viewer_{robot}")
    rec.save(str(path))
    rec.send_blueprint(blueprint, make_active=True, make_default=True)
    rec.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    scene.log_static(rec)
    times = rr.TimeColumn("t", duration=T)
    for spath, part in servos_of(root):
        tr, colour, short = traces[spath], _color(spath, root), spath.split("/")[-1]
        base = f"/cell/{spath}"
        for signal, cmd, st, scale in (("position", tr.cmd_q, tr.q, DEG), ("velocity", tr.cmd_qd, tr.qd, DEG),
                                       ("effort", tr.cmd_eff, tr.eff, 1.0)):  # fmt: skip
            rec.log(f"{base}/{signal}/state", rr.SeriesLines(colors=[colour], widths=[2.0], names=[f"{short} state"]),
                    static=True)  # fmt: skip
            rec.log(f"{base}/{signal}/cmd", rr.SeriesPoints(colors=[_tint(colour)], marker_sizes=[1.3],
                                                            markers=["circle"], names=[f"{short} cmd"]), static=True)
            rec.send_columns(f"{base}/{signal}/state", indexes=[times], columns=rr.Scalars.columns(scalars=st * scale))
            rec.send_columns(f"{base}/{signal}/cmd", indexes=[times], columns=rr.Scalars.columns(scalars=cmd * scale))
        rec.log(f"{base}/position/error", rr.SeriesLines(colors=[colour], widths=[1.5], names=[f"{short} error"]),
                static=True)  # fmt: skip
        rec.send_columns(f"{base}/position/error", indexes=[times],
                         columns=rr.Scalars.columns(scalars=(tr.cmd_q - tr.q) * DEG))  # fmt: skip
        rec.log(f"{base}/temperature", rr.SeriesLines(colors=[colour], widths=[1.5], names=[f"{short} °C"]),
                static=True)  # fmt: skip
        rec.send_columns(f"{base}/temperature", indexes=[times], columns=rr.Scalars.columns(scalars=tr.temp))
        rec.log(f"{base}/fault", rr.SeriesLines(colors=[(255, 70, 70) if tr.fault.any() else colour], widths=[2.0],
                                                names=[f"{short} fault"], interpolation_mode="StepAfter"),
                static=True)  # fmt: skip
        rec.send_columns(f"{base}/fault", indexes=[times], columns=rr.Scalars.columns(scalars=tr.fault))
        rec.log(f"{base}/config", rr.TextDocument(_servo_config_md(spath, part.config), media_type="text/markdown"),
                static=True)  # fmt: skip
    for cpath, cam in cameras_of(root):
        rec.log(f"/cell/{cpath}/config", rr.TextDocument(_camera_config_md(cpath, cam), media_type="text/markdown"),
                static=True)  # fmt: skip
    for doc in config_docs(root):
        node = next((c for c in root.children if c.name == doc), None)
        text = (_group_config_md(doc, node, f"{doc}/") if node is not None
                else _parts_config_md(f"{root.name} (cell-level parts)", [(c.name, c) for c in root.children
                                                                         if c.kind != "group"]))  # fmt: skip
        rec.log(f"/config/{doc}", rr.TextDocument(text, media_type="text/markdown"), static=True)
    config = robot_config(root)  # the RobotConfig the text is drawn from (one component per configured part)
    assert len(config.components) == len([p for p, n in walk(root) if n.config is not None])
    for k in range(len(T)):
        rec.set_time("t", duration=float(T[k]))
        scene.log_tick(rec, k)
        command, state = messages(robot, traces, k)
        rec.log("/now", rr.TextDocument(_now_md(root, command, state, k), media_type="text/markdown"))
    rec.flush()
    rec.disconnect()
    return path


# ── the five layouts: each a function (description tree, scene) -> Blueprint ───────────────────────────

LEGEND_OFF = rrb.PlotLegend(visible=False)


def _three_d(robot: str, scene: Scene) -> rrb.Spatial3DView:
    return rrb.Spatial3DView(origin="/world", name=f"3D · {ROBOTS[robot].name}", eye_controls=scene.eye)


def _frame(robot: str, scene: Scene, right: rrb.Container | rrb.View) -> rrb.Blueprint:
    """Every variant: the 3D view on the left (a hard gate), panels collapsed, the playhead on t."""
    return rrb.Blueprint(
        rrb.Horizontal(_three_d(robot, scene), right, column_shares=[1, 1.7]),
        rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline="t"), collapse_panels=False,
    )  # fmt: skip


def _position_plot(path: str, name: str | None = None) -> rrb.TimeSeriesView:
    return rrb.TimeSeriesView(origin=f"/cell/{path}/position", contents=[f"/cell/{path}/position/cmd",
                              f"/cell/{path}/position/state"], name=name or f"{path} · ° cmd ··· state —",
                              plot_legend=LEGEND_OFF)  # fmt: skip


def _overlay(name: str, paths: list[str], leaves: list[str], legend: bool = True) -> rrb.TimeSeriesView:
    contents = [f"/cell/{p}/{leaf}" for p in paths for leaf in leaves]
    return rrb.TimeSeriesView(origin="/cell", contents=contents, name=name,
                              plot_legend=rrb.PlotLegend(corner="LeftTop", visible=legend))  # fmt: skip


def _camera_view(path: str) -> rrb.Spatial2DView:
    return rrb.Spatial2DView(origin=f"/cell/{path}/image", name=f"{path} · image")


def _config_view(path: str, name: str | None = None) -> rrb.TextDocumentView:
    return rrb.TextDocumentView(origin=f"/cell/{path}/config", name=name or f"{path} · config")


def v1_tabs_by_type(robot: str, scene: Scene, tab: int = 0, sub_tab: int = 0) -> rrb.Blueprint:
    """V1 Zach's sketch: Servos / Cameras / Config. Servos grouped by the subtree they hang off."""
    root = ROBOTS[robot]
    groups = []
    for group, servos in servo_groups(root):
        cells = [_position_plot(p, f"{p.split('/')[-1]} · ° cmd ··· state —") for p, _ in servos]
        cells.append(_overlay(f"{group} · temperature °C", [p for p, _ in servos], ["temperature"], legend=False))
        groups.append(rrb.Grid(*cells, grid_columns=min(4, len(cells)), name=group))
    # WHY sub-tabs per subtree: Zach, "on servos you can break down by right arm etc so you don't have a crazy huge list"
    servos_tab = rrb.Tabs(*groups, name="Servos", active_tab=sub_tab) if len(groups) > 1 else rrb.Vertical(groups[0], name="Servos")
    cams = [_camera_view(p) for p, _ in cameras_of(root)]
    cameras_tab = rrb.Grid(*cams, grid_columns=min(2, len(cams)), name="Cameras")
    config_tab = rrb.Vertical(*[rrb.TextDocumentView(origin=f"/config/{d}", name=f"{d} · config")
                                for d in config_docs(root)], name="Config")  # fmt: skip
    return _frame(robot, scene, rrb.Tabs(servos_tab, cameras_tab, config_tab, active_tab=tab))


# V2's registry: a part kind -> its stock view. A new kind is one entry; a new part is no edit at all.
def _stock_servo(path: str, part: Part) -> rrb.Container:
    return rrb.Vertical(_position_plot(path, f"{path.split('/')[-1]} · ° cmd ··· state —"), _config_view(path),
                        row_shares=[1.5, 1], name=path)  # fmt: skip


def _stock_camera(path: str, part: Part) -> rrb.Container:
    return rrb.Vertical(_camera_view(path), _config_view(path), row_shares=[2.2, 1], name=path)


def _stock_group(path: str, part: Part) -> rrb.Container:
    kids = [stock_view(f"{path}/{c.name}" if path else c.name, c) for c in part.children]
    return rrb.Grid(*kids, grid_columns=math.ceil(math.sqrt(len(kids))) if len(kids) > 3 else len(kids), name=path)


STOCK_VIEWS: dict[str, Callable[[str, Part], rrb.Container]] = {
    "servo": _stock_servo, "camera": _stock_camera, "group": _stock_group}  # fmt: skip


def stock_view(path: str, part: Part) -> rrb.Container:
    return STOCK_VIEWS[part.kind](path, part)


def v2_tabs_mirror_tree(robot: str, scene: Scene, tab: int = 0) -> rrb.Blueprint:
    """V2: one tab per top-level part; each tab is that part's stock view, composed recursively from its children."""
    root = ROBOTS[robot]
    return _frame(robot, scene, rrb.Tabs(*[stock_view(c.name, c) for c in root.children], active_tab=tab))


def v3_tview(robot: str, scene: Scene, tab: int = 0) -> rrb.Blueprint:
    """V3: one read-only page like moteus tview: the live table first, camera thumbnails, two summary plots."""
    root = ROBOTS[robot]
    servos = [p for p, _ in servos_of(root)]
    cams = [_camera_view(p) for p, _ in cameras_of(root)]
    right = rrb.Vertical(
        rrb.TextDocumentView(origin="/now", name="now · every servo and camera at the playhead"),
        rrb.Horizontal(*cams, name="cameras"),
        rrb.Horizontal(_overlay("tracking error ° (cmd − state), every servo", servos, ["position/error"], False),
                       _overlay("temperature °C, every servo", servos, ["temperature"], False)),
        row_shares=[4.6, 1.1, 1.0],
    )  # fmt: skip
    return _frame(robot, scene, right)


COLOUR_WORDS = ("red", "amber", "green", "blue", "violet", "pink")


def _key(servos: list[tuple[str, Part]]) -> str:
    """WHY a colour key in the title, not a legend: a 12-entry legend (cmd + state per joint) hid half the curves."""
    return " ".join(f"{p.split('/')[-1]} {COLOUR_WORDS[i % 6]}" for i, (p, _) in enumerate(servos)) if len(servos) > 1 \
        else ""


SIGNAL_TABS = ("Position", "Velocity", "Effort", "Health", "Cameras", "Config")


def v4_tabs_by_signal(robot: str, scene: Scene, tab: int = 0) -> rrb.Blueprint:
    """V4: a tab per signal; in each, one plot per subtree overlays all its joints, command dots vs state lines."""
    root = ROBOTS[robot]
    groups = servo_groups(root)

    def per_group(signal: str, unit: str) -> rrb.Vertical:
        return rrb.Vertical(*[_overlay(f"{g} · {signal} {unit} · cmd ··· state — · {_key(s)}", [p for p, _ in s],
                                       [f"{signal}/cmd", f"{signal}/state"], legend=False) for g, s in groups],
                            name=signal.title())  # fmt: skip

    health = rrb.Vertical(*[rrb.Horizontal(
        _overlay(f"{g} · tracking error ° · {_key(s)}", [p for p, _ in s], ["position/error"], legend=False),
        _overlay(f"{g} · temperature °C", [p for p, _ in s], ["temperature"], legend=False),
        _overlay(f"{g} · fault code", [p for p, _ in s], ["fault"], legend=False),
        column_shares=[1.3, 1, 0.8]) for g, s in groups], name="Health")  # fmt: skip
    cams = [_camera_view(p) for p, _ in cameras_of(root)]
    config = rrb.Vertical(*[rrb.TextDocumentView(origin=f"/config/{d}", name=f"{d} · config")
                            for d in config_docs(root)], name="Config")  # fmt: skip
    return _frame(robot, scene, rrb.Tabs(per_group("position", "°"), per_group("velocity", "°/s"),
                                         per_group("effort", "N·m"), health,
                                         rrb.Grid(*cams, grid_columns=min(2, len(cams)), name="Cameras"), config,
                                         active_tab=tab))  # fmt: skip


def _card(path: str, part: Part, depth: int) -> rrb.Container:
    if part.kind == "servo":
        return rrb.Horizontal(_position_plot(path, f"{path.split('/')[-1]} · ° cmd ··· state —"),
                              _config_view(path, "config"), column_shares=[2, 1.3], name=path)  # fmt: skip
    if part.kind == "camera":
        return rrb.Vertical(_camera_view(path), _config_view(path, "config"), row_shares=[2, 1], name=path)
    kids = [_card(f"{path}/{c.name}" if path else c.name, c, depth + 1) for c in part.children]
    box = rrb.Horizontal if depth % 2 == 0 else rrb.Vertical
    return box(*kids, name=path or part.name)


def v5_cards(robot: str, scene: Scene, tab: int = 0) -> rrb.Blueprint:
    """V5: every part a card (plot + config, or image + config), nested to mirror the tree, all visible at once."""
    return _frame(robot, scene, _card("", ROBOTS[robot], 0))


VARIANTS: dict[str, tuple[str, Callable[..., rrb.Blueprint], tuple[str, ...]]] = {
    "v1": ("Tabs by part type", v1_tabs_by_type, ("servos", "cameras", "config")),
    "v2": ("Tabs mirror the tree", v2_tabs_mirror_tree, ()),
    "v3": ("tview overview", v3_tview, ()),
    "v4": ("Tabs by signal", v4_tabs_by_signal, tuple(s.lower() for s in SIGNAL_TABS)),
    "v5": ("Part cards", v5_cards, ()),
}


def tabs_of(variant: str, robot: str) -> tuple[str, ...]:
    if variant == "v2":
        return tuple(c.name for c in ROBOTS[robot].children)
    return VARIANTS[variant][2]


def main(argv: list[str]) -> int:
    only = set(argv)
    for robot in ROBOTS:
        scene = scene_for(robot, traces_for(robot))
        for key, (title, make, _) in VARIANTS.items():
            if only and key not in only and robot not in only:
                continue
            out = write(OUT / f"{key}-{robot}.rrd", robot, make(robot, scene))
            print(f"{out.relative_to(OUT.parent)}  {out.stat().st_size / 1e6:.2f} MB  {title}")
            for i, tab in enumerate(tabs_of(key, robot)[1:], start=1):
                extra = write(OUT / "tabs" / f"{key}-{robot}-{tab}.rrd", robot, make(robot, scene, tab=i))
                print(f"  {extra.relative_to(OUT.parent)}")
            if key == "v1" and robot == "dual_arm":  # the Servos tab's second subtree
                extra = write(OUT / "tabs" / f"{key}-{robot}-servos_right_arm.rrd", robot, make(robot, scene, sub_tab=1))
                print(f"  {extra.relative_to(OUT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
