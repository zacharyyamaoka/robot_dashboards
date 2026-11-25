"""Shared by the manipulation dashboards (grasping, pick_and_place); the leading _ keeps it out of discovery.

- `Arm`: forward kinematics built from a URDF's own joint origins and axes (the same numbers Rerun's loader
  draws the arm from), and a damped-least-squares IK for a tool point beyond the last link.
- `plan_cycle()`: a pick-and-place cycle for the sample: IK at each waypoint on the shipped UR5e URDF, quintic
  rest-to-rest legs (Cartesian straight lines for the short approach / lift / place / retreat moves, joint space
  for the long ones), each leg stretched until it keeps the joints' velocity and acceleration limits.
- `phases()`: where each waypoint was reached along a tool path, and how each phase went against its plan.
- The gripper's volumes and `grasp_checks()`: is a grasp clear of a point cloud, and is there something to grip.
- `mug_scene()`: a mug on a table as one RGB-D camera sees it (z-buffered, so the far side and the shadow are
  missing, as in a real single-view cloud).

WHY our own FK / IK: robot_dashboards depends on numpy, rerun and robot_msgs only (pyproject.toml), and the
robot-sdk kinematics packages are not among them; the FK reads the URDF through Rerun's own parser.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun.urdf as rurdf
from robot_msgs.geometry import Pose
from robot_msgs.geometry.rotations import euler_to_matrix, quat_from_matrix
from robot_msgs.manipulation import GraspPose, Waypoint

DEG = 180.0 / math.pi

# ── kinematics ────────────────────────────────────────────────────────────────


def _rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotations by ``angles`` about the unit ``axis`` (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3)[None] + s * skew[None] + (1.0 - c) * (skew @ skew)[None]


def _rotvec(matrix: np.ndarray) -> np.ndarray:
    """The rotation vector (axis × angle, rad) of a 3×3 rotation."""
    q = quat_from_matrix(matrix)  # xyzw
    q = q if q[3] >= 0.0 else -q
    s = float(np.linalg.norm(q[:3]))
    if s < 1e-12:
        return np.zeros(3)
    return q[:3] / s * 2.0 * math.atan2(s, q[3])


def pose_matrix(pose: Pose) -> np.ndarray:
    return pose.to_matrix()


@dataclass
class Arm:
    """A serial arm read from its URDF: the chain from the root link to ``joints[-1]``'s child, then a tool point
    ``tool`` metres along that link's z (the UR5e's flange z points out of the wrist)."""

    urdf: Path
    joints: tuple[str, ...]
    tool: float = 0.0
    steps: list[tuple[np.ndarray, np.ndarray | None, int | None]] = field(init=False)
    lower: np.ndarray = field(init=False)
    upper: np.ndarray = field(init=False)
    max_velocity: np.ndarray = field(init=False)
    root: str = field(init=False)

    def __post_init__(self) -> None:
        tree = rurdf.UrdfTree.from_file_path(str(self.urdf))
        self.root = tree.root_link().name
        by_child = {joint.child_link: joint for joint in tree.joints()}
        link, chain = tree.get_joint_by_name(self.joints[-1]).child_link, []
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].parent_link
        chain.reverse()
        self.steps = []
        for joint in chain:
            origin = np.eye(4)
            origin[:3, :3] = euler_to_matrix(*joint.origin_rpy)
            origin[:3, 3] = joint.origin_xyz
            moving = joint.name in self.joints and joint.joint_type in ("revolute", "continuous")
            index = self.joints.index(joint.name) if moving else None
            self.steps.append((origin, np.asarray(joint.axis, dtype=np.float64) if moving else None, index))
        found = [tree.get_joint_by_name(name) for name in self.joints]
        self.lower = np.array([j.limit_lower for j in found], dtype=np.float64)
        self.upper = np.array([j.limit_upper for j in found], dtype=np.float64)
        self.max_velocity = np.array([j.limit_velocity for j in found], dtype=np.float64)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """(N, 4, 4) tool poses in the root link's frame for (N, J) joint values (or one (J,) → (4, 4))."""
        q = np.asarray(q, dtype=np.float64)
        single = q.ndim == 1
        q = np.atleast_2d(q)
        pose = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        for origin, axis, index in self.steps:
            pose = pose @ origin
            if axis is not None:
                motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
                motion[:, :3, :3] = _rotations(axis, q[:, index])
                pose = pose @ motion
        tool = np.eye(4)
        tool[2, 3] = self.tool
        pose = pose @ tool
        return pose[0] if single else pose

    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The tool pose and the 6×J geometric Jacobian (linear rows first) at one configuration."""
        pose = np.eye(4)
        axes, origins, order = [], [], []
        for origin, axis, index in self.steps:
            pose = pose @ origin
            if axis is not None:
                axes.append(pose[:3, :3] @ (axis / np.linalg.norm(axis)))
                origins.append(pose[:3, 3].copy())
                order.append(index)
                motion = np.eye(4)
                motion[:3, :3] = _rotations(axis, np.array([q[index]]))[0]
                pose = pose @ motion
        tool = np.eye(4)
        tool[2, 3] = self.tool
        pose = pose @ tool
        jacobian = np.zeros((6, len(self.joints)))
        for z, o, index in zip(axes, origins, order, strict=True):
            jacobian[:3, index] = np.cross(z, pose[:3, 3] - o)
            jacobian[3:, index] = z
        return pose, jacobian

    def ik(self, target: np.ndarray, seed: np.ndarray, *, iterations: int = 300) -> np.ndarray:
        """Joint values putting the tool at the 4×4 ``target``, from ``seed`` (damped least squares, so it stays on
        the seed's branch: elbow up, wrist down for a pick)."""
        q = np.asarray(seed, dtype=np.float64).copy()
        for _ in range(iterations):
            pose, jacobian = self.jacobian(q)
            error = np.concatenate([target[:3, 3] - pose[:3, 3], _rotvec(target[:3, :3] @ pose[:3, :3].T)])
            if np.linalg.norm(error[:3]) < 1e-8 and np.linalg.norm(error[3:]) < 1e-8:
                return q
            damping = 1e-4
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
            q = q + np.clip(step, -0.2, 0.2)
        pose = self.fk(q)
        miss = float(np.linalg.norm(target[:3, 3] - pose[:3, 3]))
        raise ValueError(f"IK did not converge: {miss * 1000:.2f} mm from the target")


# ── a planned pick-and-place cycle (the sample) ──────────────────────────────

LINEAR = ("pre_grasp", "grasp", "lift", "pre_place", "place", "retreat")  # short moves: straight tool lines


def _quintic(steps: int) -> np.ndarray:
    """s(τ) = 10τ³ − 15τ⁴ + 6τ⁵ at ``steps`` samples after τ = 0 (rest to rest: peak ṡ = 1.875 / T, s̈ = 5.774 / T²)."""
    tau = np.arange(1, steps + 1) / steps
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


@dataclass(frozen=True)
class Cycle:
    t: np.ndarray  # (N,) s
    q: np.ndarray  # (N, J) rad
    qd: np.ndarray
    qdd: np.ndarray
    tool: np.ndarray  # (N, 4, 4) the tool in the root frame


def plan_cycle(arm: Arm, waypoints: tuple[Waypoint, ...], q_start: np.ndarray, *, rate: float,
               max_acceleration: np.ndarray) -> Cycle:  # fmt: skip
    """Plan the waypoints in order from ``q_start``: each leg takes its waypoint's ``duration``, stretched (never
    shortened) until every joint keeps its URDF velocity limit and ``max_acceleration``; ``dwell`` holds."""
    q = np.asarray(q_start, dtype=np.float64)
    path = [q[None]]
    for previous, waypoint in zip(waypoints, waypoints[1:], strict=False):
        start, goal = pose_matrix(previous.pose), pose_matrix(waypoint.pose)
        planned = waypoint.duration or 0.0
        same = np.linalg.norm(goal[:3, 3] - start[:3, 3]) < 1e-9 and np.allclose(goal[:3, :3], start[:3, :3])
        if same:  # the arm holds while the jaws move
            path.append(np.repeat(q[None], max(1, round(planned * rate)), axis=0))
        elif waypoint.label in LINEAR:
            duration = planned
            for _ in range(20):  # stretch until the joints keep their limits
                steps = max(2, math.ceil(duration * rate - 1e-9))
                s = _quintic(steps)
                leg, seed = [], q
                rotation = matrix_to_rotation_path(start[:3, :3], goal[:3, :3], s)
                for k in range(steps):
                    target = np.eye(4)
                    target[:3, :3] = rotation[k]
                    target[:3, 3] = start[:3, 3] + s[k] * (goal[:3, 3] - start[:3, 3])
                    seed = arm.ik(target, seed)
                    leg.append(seed)
                leg = np.array(leg)
                stretch = _stretch(np.vstack([q[None], leg]), rate, arm.max_velocity, max_acceleration)
                if stretch <= 1.0 + 1e-6:
                    break
                duration = steps / rate * stretch
            path.append(leg)
            q = leg[-1]
        else:
            goal_q = arm.ik(goal, q)
            delta = np.abs(goal_q - q)
            duration = max(planned, float(np.max(1.875 * delta / arm.max_velocity)),
                           float(np.max(np.sqrt(5.7735 * delta / max_acceleration))))  # fmt: skip
            steps = max(2, math.ceil(duration * rate - 1e-9))
            path.append(q[None] + _quintic(steps)[:, None] * (goal_q - q)[None])
            q = goal_q
        if waypoint.dwell:
            path.append(np.repeat(q[None], max(1, round(waypoint.dwell * rate)), axis=0))
    positions = np.vstack(path)
    t = np.arange(len(positions)) / rate
    velocities = np.gradient(positions, t, axis=0, edge_order=1)
    accelerations = np.gradient(velocities, t, axis=0, edge_order=1)
    return Cycle(t=t, q=positions, qd=velocities, qdd=accelerations, tool=arm.fk(positions))


def matrix_to_rotation_path(start: np.ndarray, goal: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(len(s), 3, 3): the rotation from ``start`` to ``goal`` about one fixed axis, ``s`` of the way (a slerp)."""
    rotvec = _rotvec(goal @ start.T)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.broadcast_to(start, (len(s), 3, 3)).copy()
    return _rotations(rotvec / angle, s * angle) @ start[None]


def _stretch(q: np.ndarray, rate: float, max_velocity: np.ndarray, max_acceleration: np.ndarray) -> float:
    """How much longer a sampled leg must take to keep the limits (1 = it already does)."""
    t = np.arange(len(q)) / rate
    qd = np.gradient(q, t, axis=0, edge_order=1)
    qdd = np.gradient(qd, t, axis=0, edge_order=1)
    return max(1.0, float(np.max(np.abs(qd) / max_velocity)), float(np.sqrt(np.max(np.abs(qdd) / max_acceleration))))


# ── phases: where each waypoint was reached, and how each phase went ─────────

STOP_TOLERANCE = 0.002  # m: a stop waypoint is reached within 2 mm (a UR's repeatability is ±0.03 mm; 2 mm is the
# check's own slack for a sampled trajectory, 100 Hz at the speeds of a pick)
ANGLE_TOLERANCE = 2.0  # deg
TIME_TOLERANCE = 0.10  # a phase is on plan when its time is within 10 % of the planned duration (or 50 ms)


@dataclass(frozen=True)
class Phase:
    index: int  # the waypoint it ends at
    name: str
    start: float  # s
    end: float
    planned: float | None  # s, the waypoint's duration (+ its dwell)
    miss_mm: float
    miss_deg: float
    tolerance_mm: float
    peak: tuple[str, float, str, float] | None  # (joint, peak / limit, "speed" | "acceleration", limit) here

    @property
    def actual(self) -> float:
        return self.end - self.start

    @property
    def reached(self) -> bool:
        return self.miss_mm <= self.tolerance_mm and self.miss_deg <= ANGLE_TOLERANCE

    @property
    def on_time(self) -> bool | None:
        if self.planned is None:
            return None
        return abs(self.actual - self.planned) <= max(TIME_TOLERANCE * self.planned, 0.05)

    @property
    def as_planned(self) -> bool:
        return self.reached and self.on_time is not False


def _groups(waypoints: tuple[Waypoint, ...]) -> list[list[int]]:
    """Consecutive waypoints at one pose (a grasp and its close, a place and its release) as one stop."""
    groups: list[list[int]] = []
    for i, w in enumerate(waypoints):
        if groups:
            last = waypoints[groups[-1][-1]].pose
            if np.linalg.norm(last.position.as_array() - w.pose.position.as_array()) < 1e-6:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def phases(t: np.ndarray, tool: np.ndarray, waypoints: tuple[Waypoint, ...], *, joint_t: np.ndarray | None = None,
           qd: np.ndarray | None = None, qdd: np.ndarray | None = None, joints: tuple[str, ...] = (),
           max_velocity: np.ndarray | None = None, max_acceleration: np.ndarray | None = None) -> list[Phase]:
    """One `Phase` per waypoint after the first: from when the tool left the previous waypoint to when it left
    this one (so a waypoint's dwell, and a close's hold, belong to its phase). ``t`` are the tool path's times;
    ``joint_t`` and ``qd`` / ``qdd`` (with their limits) name the joint nearest its limit in each phase.

    Each stop is found in order along the tool path: the first stretch after the previous stop where the tool is
    within 2 mm of it (a blended waypoint: within its blend radius); the tool arrives where that stretch comes
    closest (to within 20 µm, the settle band) and leaves when it moves more than that off it again. A
    waypoint never come within tolerance of is "reached" at its closest approach, and fails. Consecutive waypoints
    at one pose (a grasp and its close) share the stop: the first is left on arrival, the last on leaving."""
    t = np.asarray(t, dtype=np.float64)
    xyz = tool[:, :3, 3]
    left = [0.0] * len(waypoints)  # when the tool left each waypoint
    measured = [len(xyz) - 1] * len(waypoints)  # the sample each waypoint's miss is measured at
    cursor = 0
    # WHY 20 µm: a quintic stop covers its last 20 µm in ~15 ms, so arrival and departure each land within two
    # 100 Hz samples of the true stop (a 0.1 mm band cost 30 ms a side and flagged a 0.3 s release as late)
    settle = 2e-5
    for group in _groups(waypoints):
        first = waypoints[group[0]]
        target = pose_matrix(first.pose)[:3, 3]
        allowed = max(STOP_TOLERANCE, first.blend_radius)
        distance = np.linalg.norm(xyz[cursor:] - target, axis=1)
        inside = np.flatnonzero(distance <= allowed)
        if inside.size:
            breaks = np.flatnonzero(np.diff(inside) > 1)
            stretch = np.arange(inside[0], (inside[breaks[0]] if breaks.size else inside[-1]) + 1)
            closest = float(distance[stretch].min())
            settled = stretch[distance[stretch] <= closest + settle]
            arrive, leave = cursor + int(settled[0]), cursor + int(settled[-1])
        else:
            arrive = leave = cursor + int(np.argmin(distance))
        for k, index in enumerate(group):
            last = k == len(group) - 1
            at = leave if last else min(leave, arrive + round(k * (leave - arrive) / max(len(group) - 1, 1)))
            left[index] = float(t[at])
            measured[index] = arrive if k == 0 else at
        cursor = leave
    out = []
    for i in range(1, len(waypoints)):
        w = waypoints[i]
        planned = None if w.duration is None else w.duration + w.dwell
        pose, goal = tool[measured[i]], pose_matrix(w.pose)
        miss_m = float(np.linalg.norm(goal[:3, 3] - pose[:3, 3]))
        miss_deg = float(np.linalg.norm(_rotvec(goal[:3, :3] @ pose[:3, :3].T))) * DEG
        peak = None
        if joint_t is not None and len(joints):
            window = (joint_t >= left[i - 1] - 1e-9) & (joint_t <= left[i] + 1e-9)
            for values, limit, what in ((qd, max_velocity, "speed"), (qdd, max_acceleration, "acceleration")):
                if values is None or limit is None or not window.any():
                    continue
                fraction = np.max(np.abs(values[window]), axis=0) / limit
                j = int(np.argmax(fraction))
                if peak is None or fraction[j] > peak[1]:
                    peak = (joints[j], float(fraction[j]), what, float(limit[j]))
        tolerance = max(STOP_TOLERANCE, w.blend_radius) * 1000.0
        out.append(Phase(i, w.name, left[i - 1], left[i], planned, miss_m * 1000.0, miss_deg, tolerance, peak))
    return out


# ── the gripper's volumes, and is a grasp feasible on a cloud ────────────────

# A two-finger parallel gripper the size of a Robotiq 2F-85: the glyph's 40 mm fingers (robot_msg_viz's
# FINGER_DEPTH), each 8 mm thick and 20 mm wide, a 30 mm palm behind them, and an 85 mm stroke.
FINGER = 0.040
FINGER_THICK = 0.008
FINGER_WIDE = 0.020
PALM = 0.030
STROKE = 0.085
STANDOFF = 0.10  # m: the approach starts this far back along the grasp's -z (a pre-grasp offset)
MIN_BETWEEN = 15  # points between the jaws: fewer and the jaws close on air


@dataclass(frozen=True)
class GraspCheck:
    fingers: int  # cloud points inside the open fingers at the grasp
    palm: int
    approach: int  # inside the open gripper swept back along the approach (fingers + palm), not at the grasp
    between: int  # points between the open jaws (what they close on)
    open_width: float
    hits: np.ndarray  # the points the gripper would hit (fingers, palm or approach), for drawing

    @property
    def within_stroke(self) -> bool:
        return self.open_width <= STROKE + 1e-9

    @property
    def feasible(self) -> bool:
        return self.fingers == 0 and self.palm == 0 and self.approach == 0 and self.between >= MIN_BETWEEN \
            and self.within_stroke  # fmt: skip

    def reasons(self) -> list[str]:
        out = []
        if self.fingers:
            out.append(f"fingers hit {self.fingers} points")
        if self.palm:
            out.append(f"palm hits {self.palm} points")
        if self.approach:
            out.append(f"approach path hits {self.approach} points")
        if self.between < MIN_BETWEEN:
            out.append(f"only {self.between} points between the jaws")
        if not self.within_stroke:
            out.append(f"opens {self.open_width * 1000:.0f} mm, past the {STROKE * 1000:.0f} mm stroke")
        return out


def _in_box(local: np.ndarray, low, high) -> np.ndarray:
    return np.all((local >= np.asarray(low)) & (local <= np.asarray(high)), axis=1)


def grasp_checks(grasp: GraspPose, points: np.ndarray) -> GraspCheck:
    """The open gripper (pre-grasp width, else the closing width) at the grasp and swept back along its approach,
    against the cloud's points (in the grasp's frame: fingers along ±x, approach along +z, tips at the origin)."""
    matrix = pose_matrix(grasp.pose)
    local = (np.asarray(points, dtype=np.float64) - matrix[:3, 3]) @ matrix[:3, :3]
    half = (grasp.pre_grasp_width or grasp.width) / 2.0
    y = FINGER_WIDE / 2.0
    fingers = _in_box(local, (half, -y, -FINGER), (half + FINGER_THICK, y, 0.0)) | \
        _in_box(local, (-half - FINGER_THICK, -y, -FINGER), (-half, y, 0.0))  # fmt: skip
    palm = _in_box(local, (-half - FINGER_THICK, -y, -FINGER - PALM), (half + FINGER_THICK, y, -FINGER))
    swept = _in_box(local, (half, -y, -FINGER - STANDOFF), (half + FINGER_THICK, y, 0.0)) | \
        _in_box(local, (-half - FINGER_THICK, -y, -FINGER - STANDOFF), (-half, y, 0.0)) | \
        _in_box(local, (-half - FINGER_THICK, -y, -FINGER - PALM - STANDOFF), (half + FINGER_THICK, y, -FINGER))  # fmt: skip
    # WHY the approach excludes what is between the open jaws at the grasp: the object sits there by design;
    # anything between them on the way in (before the tips arrive) would be pushed, so it counts
    between = _in_box(local, (-half, -y, -FINGER), (half, y, 0.0))
    approach = swept & ~fingers & ~palm & ~between
    hits = fingers | palm | approach
    return GraspCheck(int(fingers.sum()), int(palm.sum()), int(approach.sum()), int(between.sum()),
                      2.0 * half, np.asarray(points)[hits])  # fmt: skip


# ── a mug on a table, as one RGB-D camera sees it ────────────────────────────

MUG_RADIUS = 0.040  # m, outer
MUG_WALL = 0.005
MUG_HEIGHT = 0.095
HANDLE_CENTRE = (0.044, 0.050)  # (radial, height) of the handle's arc centre
HANDLE_ARC = 0.022  # the arc's radius
HANDLE_TUBE = 0.006
HANDLE_ANGLE = math.radians(-35.0)  # the handle's side of the mug, about z from world +x
CAMERA_EYE = np.array([0.05, -0.42, 0.40])
CAMERA_LOOK = np.array([0.0, 0.0, 0.04])


def _mug_surface(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense surface samples of the mug and the table (points, unit normals, part index; the table is 5), in the
    world (table top at z = 0)."""
    parts = []

    def wall(radius: float, sign: float, count: int) -> None:
        a = rng.uniform(0, 2 * np.pi, count)
        z = rng.uniform(0.006, MUG_HEIGHT, count)
        p = np.column_stack([radius * np.cos(a), radius * np.sin(a), z])
        n = sign * np.column_stack([np.cos(a), np.sin(a), np.zeros(count)])
        parts.append((p, n))

    wall(MUG_RADIUS, 1.0, 60000)
    wall(MUG_RADIUS - MUG_WALL, -1.0, 40000)
    r = np.sqrt(rng.uniform((MUG_RADIUS - MUG_WALL) ** 2, MUG_RADIUS**2, 6000))
    a = rng.uniform(0, 2 * np.pi, 6000)
    parts.append((np.column_stack([r * np.cos(a), r * np.sin(a), np.full(6000, MUG_HEIGHT)]),
                  np.tile([0.0, 0.0, 1.0], (6000, 1))))  # the rim  # fmt: skip
    r = (MUG_RADIUS - MUG_WALL) * np.sqrt(rng.uniform(0, 1, 15000))
    a = rng.uniform(0, 2 * np.pi, 15000)
    parts.append((np.column_stack([r * np.cos(a), r * np.sin(a), np.full(15000, 0.008)]),
                  np.tile([0.0, 0.0, 1.0], (15000, 1))))  # the inside bottom  # fmt: skip
    # the handle: a tube along an arc in the (radial, z) plane at HANDLE_ANGLE, its ends in the wall
    count = 30000
    alpha = rng.uniform(math.radians(-75), math.radians(75), count)
    phi = rng.uniform(0, 2 * np.pi, count)
    centre_r = HANDLE_CENTRE[0] + HANDLE_ARC * np.cos(alpha)
    centre_z = HANDLE_CENTRE[1] + HANDLE_ARC * np.sin(alpha)
    out_r, out_z = np.cos(alpha), np.sin(alpha)  # the arc's outward normal in (radial, z)
    radial = centre_r + HANDLE_TUBE * (np.cos(phi) * out_r)
    tangential = HANDLE_TUBE * np.sin(phi)
    height = centre_z + HANDLE_TUBE * (np.cos(phi) * out_z)
    u = np.array([math.cos(HANDLE_ANGLE), math.sin(HANDLE_ANGLE), 0.0])
    v = np.array([-math.sin(HANDLE_ANGLE), math.cos(HANDLE_ANGLE), 0.0])
    p = radial[:, None] * u + tangential[:, None] * v + height[:, None] * np.array([0, 0, 1.0])
    n = (np.cos(phi) * out_r)[:, None] * u + np.sin(phi)[:, None] * v + (np.cos(phi) * out_z)[:, None] * [0, 0, 1.0]
    keep = radial > MUG_RADIUS - 0.001
    parts.append((p[keep], n[keep]))
    count = 160000
    table = np.column_stack([rng.uniform(-0.16, 0.16, count), rng.uniform(-0.13, 0.17, count), np.zeros(count)])
    table = table[np.hypot(table[:, 0], table[:, 1]) > MUG_RADIUS]
    parts.append((table, np.tile([0.0, 0.0, 1.0], (len(table), 1))))
    points = np.vstack([p for p, _ in parts])
    normals = np.vstack([n for _, n in parts])
    kinds = np.concatenate([np.full(len(p), k) for k, (p, _) in enumerate(parts)])
    return points, normals, kinds


def mug_scene(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """(points (N, 3) float32, rgb (N, 3) uint8): the mug and table seen from `CAMERA_EYE`, one point per pixel of a
    320 × 240 camera (nearest surface wins), with ~1 mm of depth noise, shaded by a light above-left; the table
    cropped to the 32 × 30 cm around the mug, as a tabletop pipeline crops to its workspace."""
    rng = np.random.default_rng(seed)
    points, normals, kinds = _mug_surface(rng)
    forward = (CAMERA_LOOK - CAMERA_EYE) / np.linalg.norm(CAMERA_LOOK - CAMERA_EYE)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rel = points - CAMERA_EYE
    depth = rel @ forward
    facing = np.einsum("ij,ij->i", normals, -rel) > 0.0
    width, height, focal = 320, 240, 380.0
    u = np.round(rel @ right / depth * focal + width / 2).astype(int)
    v = np.round(rel @ down / depth * focal + height / 2).astype(int)
    ok = facing & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    pixel = v[ok] * width + u[ok]
    index = np.flatnonzero(ok)
    order = np.lexsort((depth[index], pixel))  # by pixel, nearest first
    pixel, index = pixel[order], index[order]
    first = np.concatenate([[True], pixel[1:] != pixel[:-1]])
    seen = index[first]
    ray = rel[seen] / np.linalg.norm(rel[seen], axis=1, keepdims=True)
    noisy = points[seen] + ray * rng.normal(0.0, 0.001, (len(seen), 1))
    light = np.array([-0.4, -0.5, 1.0]) / np.linalg.norm([-0.4, -0.5, 1.0])
    shade = 0.55 + 0.45 * np.clip(normals[seen] @ light, 0.0, 1.0)
    base = np.where((kinds[seen] == 5)[:, None], [[112, 92, 72]], [[244, 240, 230]])  # a walnut table, a cream mug
    rgb = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)
    return noisy.astype(np.float32), rgb

