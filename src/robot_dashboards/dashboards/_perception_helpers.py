"""Shared by the perception dashboards (object_detection, detection_3d, rgbd_camera).

- Recording plumbing: where static scenery goes, and one stable recording per stream so robot_msg_viz keeps a
  class's colour for the whole stream.
- One class -> colour map per recording, logged at the root as an `rr.AnnotationContext`, in the same
  first-appearance order and palette as robot_msg_viz's vision views, so a class has one colour in every view.
- A small numpy ray caster for the sample scenes (boxes and cylinders on a ground plane, hard shadows, contact
  shadows, highlights): it returns a consistent RGB picture, depth along the optical axis and which object each
  pixel sees, so the sample detections can be cut from what is actually visible.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rerun as rr
from robot_msg_viz._engine import Timed
from robot_msgs.sensor import CameraInfo
from robot_msgs.std import Header

# ── recordings ───────────────────────────────────────────────────────────────


@dataclass
class _State:
    """What a dashboard remembers about one recording (stored on the recording itself, so it dies with it)."""

    timed: Timed | None = None
    classes: dict[str, int] = field(default_factory=dict)  # class name -> class id (1, 2, ...), first appearance
    logged_classes: tuple[str, ...] = ()
    memory: dict[str, Any] = field(default_factory=dict)  # per dashboard: tracks seen, trails, ...


def inner(rec: Any) -> rr.RecordingStream:
    """The real recording under a `Timed` wrapper: where static scenery (class map, legends) is logged."""
    return rec._inner if isinstance(rec, Timed) else rec


def state(rec: Any) -> _State:
    stream = inner(rec)
    found = getattr(stream, "_robot_dashboards_perception", None)
    if found is None:
        found = _State()
        stream._robot_dashboards_perception = found  # type: ignore[attr-defined]
    return found


def drawing(rec: Any) -> Any:
    """The recording to hand to robot_msg_viz: the same object for every tick of one stream.

    WHY: robot_msg_viz's vision views keep a class's colour per recording object (by id), and
    `DashboardStream.log` wraps the stream in a fresh `Timed` every tick, so without this a class could change
    colour from one frame to the next in a live stream.
    """
    if not isinstance(rec, Timed):
        return rec
    memo = state(rec)
    if memo.timed is None:
        memo.timed = Timed(inner(rec))
    return memo.timed


# ── class colours ────────────────────────────────────────────────────────────

try:  # one palette with robot_msg_viz's vision views, so the boxes they draw and our plots agree
    from robot_msg_viz.views.vision import _NO_CLASS as NO_CLASS
    from robot_msg_viz.views.vision import _PALETTE as PALETTE
except ImportError:  # pragma: no cover - the vision views moved; keep their published colours
    PALETTE = (
        (20, 130, 205), (225, 120, 0), (0, 155, 110), (200, 80, 160),
        (205, 75, 30), (120, 95, 205), (150, 140, 0), (120, 120, 120),
    )  # fmt: skip
    NO_CLASS = (140, 140, 140)


def class_colour(class_id: int) -> tuple[int, int, int]:
    return tuple(NO_CLASS) if class_id == 0 else tuple(PALETTE[(class_id - 1) % len(PALETTE)])


def class_ids(rec: Any, names: Sequence[str]) -> list[int]:
    """Class ids for `names` in first-appearance order (robot_msg_viz's rule), and the root class map updated.

    WHY the root: every view (the image, the 3D boxes, the plots) then reads one class -> colour map.
    """
    memo = state(rec)
    ids = [0 if not name else memo.classes.setdefault(name, len(memo.classes) + 1) for name in names]
    if tuple(memo.classes) != memo.logged_classes:
        memo.logged_classes = tuple(memo.classes)
        infos = [rr.AnnotationInfo(id=0, label="no class", color=NO_CLASS)]
        infos += [rr.AnnotationInfo(id=i, label=n, color=class_colour(i)) for n, i in memo.classes.items()]
        inner(rec).log("/", rr.AnnotationContext(infos), static=True)
    return ids


# ── the ray caster for the sample scenes ─────────────────────────────────────

_EPS = 1e-6


def _mv(rows: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """(N, 3) @ (3,) without BLAS. WHY: numpy's matmul hands this to a threaded BLAS that ran 5x slower than einsum
    on a busy machine (measured 2026-11-24, 300k rows)."""
    return np.einsum("ij,j->i", rows, vector)


Paint = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (local coordinates (N, 3), part (N,)) -> linear albedo


def srgb(colour: Sequence[float]) -> np.ndarray:
    """An sRGB colour (0-255) as linear light, the space the shading runs in."""
    return (np.asarray(colour, dtype=float) / 255.0) ** 2.2


@dataclass(frozen=True)
class Box:
    """A box standing on (or above) the ground, turned `yaw` about z. Local coordinates: the box frame, metres."""

    centre: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float
    paint: Paint
    gloss: float = 0.05

    def bound(self) -> tuple[np.ndarray, float]:
        return np.asarray(self.centre, dtype=float), float(np.linalg.norm(self.half))

    def _to_local(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])

    def hit(self, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(t, world normal, local point, part) per ray; t = inf on a miss. Part: 0-5 = -x, -y, -z, +x, +y, +z."""
        to_local = self._to_local()
        local_o = np.einsum("ij,kj->ik", o - np.asarray(self.centre), to_local)
        local_d = np.einsum("ij,kj->ik", d, to_local)
        half = np.asarray(self.half)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (-half - local_o) / local_d, (half - local_o) / local_d
        near_each = np.nan_to_num(np.minimum(t1, t2), nan=-np.inf)
        far_each = np.nan_to_num(np.maximum(t1, t2), nan=np.inf)
        near, far = near_each.max(axis=1), far_each.min(axis=1)
        t = np.where((near <= far) & (near > _EPS), near, np.inf)
        axis = near_each.argmax(axis=1)
        rows = np.arange(len(o))
        sign = -np.sign(local_d[rows, axis])
        local_n = np.zeros_like(o)
        local_n[rows, axis] = sign
        point = local_o + local_d * np.where(np.isfinite(t), t, 0.0)[:, None]
        part = axis + np.where(sign > 0, 3, 0)
        return t, local_n @ to_local, point, part

    def footprint_gap(self, xy: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = xy[:, 0] - self.centre[0], xy[:, 1] - self.centre[1]
        local = np.abs(np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)) - np.asarray(self.half[:2])
        return np.hypot(*np.maximum(local, 0.0).T) + np.minimum(local.max(axis=1), 0.0)

    def corners(self) -> np.ndarray:
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
        return (signs * np.asarray(self.half)) @ self._to_local() + np.asarray(self.centre)


@dataclass(frozen=True)
class Cylinder:
    """A capped cylinder from `base` along unit `axis` for `length`. Local coordinates: (along the axis, angle
    around it in radians, distance from the axis); part 0 = the side, 1 = the base cap, 2 = the far cap."""

    base: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    length: float
    paint: Paint
    gloss: float = 0.1

    def bound(self) -> tuple[np.ndarray, float]:
        centre = np.asarray(self.base) + np.asarray(self.axis) * self.length / 2.0
        return centre, math.hypot(self.radius, self.length / 2.0)

    def _frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        axis = np.asarray(self.axis, dtype=float)
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        first = np.cross(helper, axis)
        first /= np.linalg.norm(first)
        return axis, first, np.cross(axis, first)

    def hit(self, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        axis, first, second = self._frame()
        oc = o - np.asarray(self.base)
        along_d, along_o = _mv(d, axis), _mv(oc, axis)
        perp_d, perp_o = d - along_d[:, None] * axis, oc - along_o[:, None] * axis
        a = np.einsum("ij,ij->i", perp_d, perp_d)
        b = 2.0 * np.einsum("ij,ij->i", perp_d, perp_o)
        c = np.einsum("ij,ij->i", perp_o, perp_o) - self.radius**2
        disc = b * b - 4.0 * a * c
        with np.errstate(divide="ignore", invalid="ignore"):
            side = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
            caps = [(0.0 - along_o) / along_d, (self.length - along_o) / along_d]
        height = along_o + side * along_d
        side = np.where((disc > 0.0) & (side > _EPS) & (height >= 0.0) & (height <= self.length), side, np.inf)
        best, part = side, np.zeros(len(o), dtype=int)
        for index, cap in enumerate(caps, start=1):
            radial = perp_o + np.nan_to_num(cap)[:, None] * perp_d
            ok = (cap > _EPS) & (np.einsum("ij,ij->i", radial, radial) <= self.radius**2)
            cap = np.where(ok, cap, np.inf)
            closer = cap < best
            best, part = np.where(closer, cap, best), np.where(closer, index, part)
        safe = np.where(np.isfinite(best), best, 0.0)
        radial = perp_o + safe[:, None] * perp_d
        normal = radial / self.radius
        normal[part == 1] = -axis
        normal[part == 2] = axis
        local = np.stack([
            along_o + safe * along_d,
            np.arctan2(_mv(radial, second), _mv(radial, first)),
            np.linalg.norm(radial, axis=1),
        ], axis=1)  # fmt: skip
        return best, normal, local, part

    def footprint_gap(self, xy: np.ndarray) -> np.ndarray:
        start = np.asarray(self.base[:2])
        end = start + np.asarray(self.axis[:2]) * self.length
        span = end - start
        share = np.clip(((xy - start) @ span) / max(span @ span, 1e-12), 0.0, 1.0)
        return np.linalg.norm(xy - (start + share[:, None] * span), axis=1) - self.radius

    def corners(self) -> np.ndarray:
        axis, first, second = self._frame()
        ring = [first * x + second * y for x in (-self.radius, self.radius) for y in (-self.radius, self.radius)]
        ends = [np.asarray(self.base), np.asarray(self.base) + axis * self.length]
        return np.array([end + offset for end in ends for offset in ring])


Solid = Box | Cylinder


@dataclass(frozen=True)
class Thing:
    """One object in a scene: its class, its solids, and the box a perfect 3D detector would report."""

    cls: str
    solids: tuple[Solid, ...]
    box_centre: tuple[float, float, float]
    box_size: tuple[float, float, float]
    box_yaw: float

    def corners(self) -> np.ndarray:
        return np.concatenate([s.corners() for s in self.solids])


@dataclass(frozen=True)
class Render:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 metres along the optical axis, 0 where nothing was hit
    owner: np.ndarray  # (H, W) int: index of the Thing each pixel sees, -1 for the ground


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 pose of a camera optical frame (x right, y down, z forward) at `eye` looking at `target`."""
    eye_v = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye_v
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    if np.linalg.norm(right) < 1e-9:  # looking straight along `up`: image right is world +x
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, down, forward], axis=1)
    pose[:3, 3] = eye_v
    return pose


def _first_hits(
    solids: Sequence[tuple[int, Solid]], o: np.ndarray, d: np.ndarray, *, want_paint: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest solid per ray: (t, normal, albedo, gloss, owner). `o` and `d` are (N, 3), or one (3,) shared by
    every ray (the eye of a camera, the direction to the light).

    Only rays passing a solid's bounding sphere get its exact test; WHY the shared-vector cases: the cull is then a
    matrix-vector product instead of per-ray dot products, which halves a 300k-ray frame.
    """
    n = len(o) if o.ndim == 2 else len(d)
    t = np.full(n, np.inf)
    normal, albedo = np.zeros((n, 3)), np.zeros((n, 3))
    gloss, owner = np.zeros(n), np.full(n, -1)
    if o.ndim == 1:
        dd = np.einsum("ij,ij->i", d, d)
    elif d.ndim == 1:
        dd, o_d, o_o = float(d @ d), _mv(o, d), np.einsum("ij,ij->i", o, o)
    else:
        dd = np.einsum("ij,ij->i", d, d)
    for index, solid in solids:
        centre, radius = solid.bound()
        if o.ndim == 1:
            to_centre = centre - o
            along = _mv(d, to_centre)
            miss = to_centre @ to_centre - along**2 / dd
        elif d.ndim == 1:
            along = centre @ d - o_d
            miss = o_o - 2.0 * _mv(o, centre) + centre @ centre - along**2 / dd
        else:
            to_centre = centre - o
            along = np.einsum("ij,ij->i", to_centre, d)
            miss = np.einsum("ij,ij->i", to_centre, to_centre) - along**2 / dd
        near = np.flatnonzero((miss <= radius**2) & (along > -radius * np.sqrt(dd)))
        if near.size == 0:
            continue
        near_o = np.broadcast_to(o, (len(near), 3)) if o.ndim == 1 else o[near]
        near_d = np.broadcast_to(d, (len(near), 3)) if d.ndim == 1 else d[near]
        hit_t, hit_n, local, part = solid.hit(near_o, near_d)
        closer = hit_t < t[near]
        if not closer.any():
            continue
        rows = near[closer]
        t[rows], normal[rows], owner[rows], gloss[rows] = hit_t[closer], hit_n[closer], index, solid.gloss
        if want_paint:
            albedo[rows] = solid.paint(local[closer], part[closer])
    return t, normal, albedo, gloss, owner


def render(
    camera: CameraInfo,
    world_from_optical: np.ndarray,
    things: Sequence[Thing],
    ground: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    supersample: int = 2,
    light: Sequence[float] = (-0.45, 0.35, 1.0),
    exposure: float = 1.0,
    projector_baseline: float | None = None,
) -> Render:
    """Ray-cast `things` on the ground plane z = 0 (painted by `ground(x, y)`) through `camera` at
    `world_from_optical`.

    `projector_baseline` (metres along the optical x axis) marks depth pixels the depth sensor's second view
    cannot see as missing (0), the stereo shadow every RGB-D camera has beside an object's edge.
    """
    s = supersample
    rows, cols = np.mgrid[0 : camera.height * s, 0 : camera.width * s]
    u, v = (cols + 0.5) / s - 0.5, (rows + 0.5) / s - 0.5
    rays = np.stack([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, np.ones_like(u, dtype=float)], -1)
    rotation, eye = world_from_optical[:3, :3], world_from_optical[:3, 3]
    d = np.einsum("ij,kj->ik", rays.reshape(-1, 3), rotation)  # optical z = 1, so the hit distance t IS the depth
    solids = [(index, solid) for index, thing in enumerate(things) for solid in thing.solids]
    t, normal, albedo, gloss, owner = _first_hits(solids, eye, d, want_paint=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ground_t = -eye[2] / d[:, 2]
    ground_t = np.where(ground_t > _EPS, ground_t, np.inf)
    on_ground = ground_t < t
    t = np.where(on_ground, ground_t, t)
    owner[on_ground] = -1
    normal[on_ground] = (0.0, 0.0, 1.0)
    gloss[on_ground] = 0.0
    hit = np.isfinite(t)
    point = eye + d * np.where(hit, t, 0.0)[:, None]
    ground_rows = np.flatnonzero(on_ground)
    albedo[ground_rows] = ground(point[ground_rows, 0], point[ground_rows, 1])
    # hard shadows toward a directional light, a soft contact shadow where things stand, a sky term, highlights
    to_light = np.asarray(light, dtype=float) / np.linalg.norm(light)
    lambert = np.clip(_mv(normal, to_light), 0.0, 1.0)
    lit = np.ones(len(t))
    facing = np.flatnonzero(hit & (lambert > 0.0))
    shadow_t = _first_hits(solids, point[facing] + normal[facing] * 1e-4, to_light, want_paint=False)[0]
    lit[facing[np.isfinite(shadow_t)]] = 0.0
    contact = np.ones(len(t))
    for _, solid in solids:
        if min(c[2] for c in solid.corners()) > 0.01:
            continue  # it does not touch the ground
        centre, radius = solid.bound()
        close = ground_rows[np.hypot(*(point[ground_rows, :2] - centre[:2]).T) < radius + 0.06]
        gap = np.maximum(solid.footprint_gap(point[close, :2]), 0.0)
        contact[close] *= 1.0 - 0.5 * np.exp(-gap / 0.012)
    sky = 0.24 + 0.10 * (0.5 + 0.5 * normal[:, 2])
    linear = exposure * albedo * (sky * contact + 0.8 * lambert * lit)[:, None]
    shiny = np.flatnonzero(gloss > 0.0)  # Blinn-Phong highlights, only where a surface has gloss
    halfway = to_light - d[shiny] / np.linalg.norm(d[shiny], axis=1, keepdims=True)
    halfway /= np.linalg.norm(halfway, axis=1, keepdims=True)
    cosine = np.clip(np.einsum("ij,ij->i", normal[shiny], halfway), 0.0, 1.0)
    linear[shiny] += (gloss[shiny] * cosine**50 * lit[shiny])[:, None]
    linear[~hit] = srgb((40, 41, 44))
    picture = np.clip(linear, 0.0, 1.0) ** (1 / 2.2)
    height, width = camera.height, camera.width
    picture = picture.reshape(height, s, width, s, 3).mean(axis=(1, 3))
    pick = (slice(None), s // 2, slice(None), s // 2)  # the sub-sample nearest each pixel centre
    depth = np.where(hit, t, 0.0).reshape(height, s, width, s)[pick]
    owners = owner.reshape(height, s, width, s)[pick]
    if projector_baseline is not None:
        centre_rows = np.arange(len(t)).reshape(height, s, width, s)[pick].reshape(-1)
        target = point[centre_rows] + normal[centre_rows] * 1e-4
        source = eye + rotation[:, 0] * projector_baseline
        blocked_t = _first_hits(solids, target, source - target, want_paint=False)[0]
        depth = np.where((blocked_t < 1.0).reshape(height, width), 0.0, depth)
    return Render(np.round(picture * 255.0).astype(np.uint8), depth.astype(np.float32), owners.astype(int))


def project(camera: CameraInfo, world_from_optical: np.ndarray, points: np.ndarray) -> np.ndarray:
    """World points -> pixels (u, v) through `camera` at `world_from_optical` (points behind it come back nan)."""
    optical = (np.asarray(points, dtype=float) - world_from_optical[:3, 3]) @ world_from_optical[:3, :3]
    pixels = camera.project(optical)
    return np.where(optical[:, 2:3] > 1e-6, pixels, np.nan)


# ── the sample objects: one vocabulary (bottle, can, box, cup) across the perception dashboards ──


def _flat(colour: Sequence[float]) -> Paint:
    albedo = srgb(colour)
    return lambda local, part: np.tile(albedo, (len(local), 1))


def bottle(x: float, y: float, yaw: float) -> Thing:
    """A 0.5 l PET bottle lying on its side, centred at (x, y), its cap pointing along `yaw`."""
    axis = (math.cos(yaw), math.sin(yaw), 0.0)
    radius = 0.033
    base = np.array([x, y, radius]) - np.asarray(axis) * 0.109

    def body(local: np.ndarray, part: np.ndarray) -> np.ndarray:
        out = np.tile(srgb((150, 196, 226)), (len(local), 1))  # clear blue PET
        along = local[:, 0]
        label = (along > 0.05) & (along < 0.12)
        out[label] = srgb((58, 142, 84))
        out[label & (np.abs(along - 0.085) < 0.006)] = srgb((235, 235, 235))
        out[part == 1] = srgb((120, 170, 205))
        return out

    solids = (
        Cylinder(tuple(base), axis, radius, 0.165, body, 0.5),
        Cylinder(tuple(base + np.asarray(axis) * 0.165), axis, 0.015, 0.034, _flat((160, 200, 228)), 0.5),
        Cylinder(tuple(base + np.asarray(axis) * 0.198), axis, 0.0165, 0.02, _flat((34, 84, 196)), 0.2),
    )
    return Thing("bottle", solids, (x, y, radius), (0.218, 0.066, 0.066), yaw)


def can(x: float, y: float, colour: Sequence[float] = (180, 30, 40)) -> Thing:
    """A 330 ml drinks can standing upright."""

    def paint(local: np.ndarray, part: np.ndarray) -> np.ndarray:
        out = np.tile(srgb(colour), (len(local), 1))
        along = local[:, 0]
        out[(along > 0.108) | (along < 0.008)] = srgb((200, 203, 208))  # the aluminium neck and foot
        out[np.abs(along - 0.06) < 0.006] = srgb((240, 240, 240))
        top = part == 2
        out[top] = srgb((190, 193, 198))
        out[top & (local[:, 2] > 0.028)] = srgb((215, 217, 221))  # the rim
        return out

    solid = Cylinder((x, y, 0.0), (0.0, 0.0, 1.0), 0.033, 0.122, paint, 0.55)
    return Thing("can", (solid,), (x, y, 0.061), (0.066, 0.066, 0.122), 0.0)


def box(x: float, y: float, yaw: float, half: tuple[float, float, float] = (0.08, 0.055, 0.045)) -> Thing:
    """A taped cardboard box."""

    def paint(local: np.ndarray, part: np.ndarray) -> np.ndarray:
        out = np.tile(srgb((196, 152, 98)), (len(local), 1))
        top = part == 5
        out[top & (np.abs(local[:, 1]) < 0.02)] = srgb((218, 196, 150))  # the tape
        mark = top & (local[:, 0] > 0.03) & (local[:, 0] < 0.06) & (local[:, 1] > 0.028) & (local[:, 1] < 0.045)
        out[mark] = srgb((60, 50, 40))
        return out

    solid = Box((x, y, half[2]), half, yaw, paint, 0.06)
    return Thing("box", (solid,), (x, y, half[2]), (2 * half[0], 2 * half[1], 2 * half[2]), yaw)


def cup(x: float, y: float) -> Thing:
    """A paper coffee cup, open, with coffee in it."""

    def paint(local: np.ndarray, part: np.ndarray) -> np.ndarray:
        out = np.tile(srgb((238, 236, 230)), (len(local), 1))
        along = local[:, 0]
        out[(part == 0) & (along > 0.035) & (along < 0.07)] = srgb((0, 128, 128))
        top = part == 2
        out[top & (local[:, 2] < 0.037)] = srgb((92, 62, 40))
        out[top & (local[:, 2] < 0.030) & (local[:, 2] > 0.024)] = srgb((130, 92, 60))
        return out

    solid = Cylinder((x, y, 0.0), (0.0, 0.0, 1.0), 0.042, 0.10, paint, 0.15)
    return Thing("cup", (solid,), (x, y, 0.05), (0.084, 0.084, 0.10), 0.0)


# ── the tabletop scene shared by detection_3d and rgbd_camera ────────────────

TABLE_STAMP = 1_764_028_800.0
OPTICAL_FRAME = "camera_color_optical_frame"
# A RealSense-D435-like colour camera in its 424 x 240 mode (depth aligned to colour, rectified).
TABLE_CAMERA = CameraInfo.from_intrinsics(
    307.0, 307.0, 211.5, 119.5, 424, 240, d=np.zeros(5),
    header=Header(stamp=TABLE_STAMP, frame_id=OPTICAL_FRAME),
)  # fmt: skip
# A head camera above and behind the workspace, in base_link (table top at z = 0), looking down at about 50 degrees.
TABLE_CAMERA_POSE = look_at((0.12, -0.10, 0.62), (0.62, 0.02, 0.02))
STEREO_BASELINE = 0.05  # m: the depth sensor's second view, for the stereo shadow beside each object


def worktop(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A light grey work surface with a faint mottle."""
    mottle = 1.0 + 0.04 * np.sin(x * 23.0 + np.sin(y * 17.0)) * np.cos(y * 19.0)
    return srgb((178, 176, 170))[None, :] * mottle[:, None]


def tabletop_things() -> tuple[Thing, ...]:
    return (
        bottle(0.52, 0.13, math.radians(35.0)),
        can(0.66, -0.12),
        box(0.76, 0.03, math.radians(-20.0)),
        cup(0.50, -0.06),
        can(0.88, 0.17, (30, 70, 170)),
    )


@functools.cache
def tabletop() -> Render:
    """The tabletop through TABLE_CAMERA at TABLE_CAMERA_POSE (computed once: about a second)."""
    return render(
        TABLE_CAMERA, TABLE_CAMERA_POSE, tabletop_things(), worktop, supersample=3,
        projector_baseline=STEREO_BASELINE,
    )  # fmt: skip


@functools.cache
def tabletop_depth_mm() -> np.ndarray:
    """The tabletop's depth as an RGB-D camera reports it: 16-bit millimetres, 0 = no return, with the noise of a
    stereo sensor (growing with the square of the range, ~1 mm at 0.6 m), the stereo shadow beside each object
    (from `tabletop`) and dropouts where stereo matching fails, along depth edges."""
    noise = np.random.default_rng(424)
    metres = tabletop().depth.astype(float)
    sigma = 0.0011 * (metres / 0.6) ** 2
    noisy = metres + noise.normal(0.0, 1.0, metres.shape) * sigma
    rows, cols = np.abs(np.diff(metres, axis=0, prepend=metres[:1])), np.abs(np.diff(metres, axis=1, prepend=metres[:, :1]))
    step = np.maximum(rows, cols)  # the depth jump to the pixel above or to the left
    dropout = noise.random(metres.shape) < np.where(step > 0.02, 0.45, 0.0003)
    return np.where((metres > 0) & ~dropout, np.round(noisy * 1000.0), 0).astype(np.uint16)
