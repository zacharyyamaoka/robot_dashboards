"""Object detection: what did the detector see in each frame, and how sure was it over time?

Inputs per frame: ``image`` (sensor.Image) and ``detections`` (vision.Detection2DArray, in that image's pixels).

Views:
- the camera: the frame with every box in its class colour, labelled "class score" (robot_msg_viz's vision
  view), and a short trail behind each tracked object, so motion and track hand-offs read at a glance;
- score per track over time (0-1), one line per track id in its class colour: a confidence dip, a missed frame
  (a gap) or a short-lived low-score track (a likely false positive) stands out;
- detections per class over time (a step plot): objects entering, leaving, and frames where one was missed;
- this frame, as numbers: each detection best first with its track, score, runner-up class (the margin is how
  sure it was) and box size.

Prior art: Rerun's detect_and_track_objects example (one entity per track, one static class colour map) and
FiftyOne's per-frame review; dataset-level scoring (mAP, TP/FP against ground truth) is out of scope here.

Sample: 3 s at 10 Hz of a top-down camera over a waste-sorting conveyor (a bottle, two cans, a box, a cup on a
moving belt), ray-cast in numpy, with detections cut from what each object shows in the frame: jittered boxes,
scores that wander, a can leaving the frame, a cup entering, one frame where the bottle is missed and two frames
where an oil stain on the belt is called a can at a low score.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz
from robot_msg_viz._engine import part
from robot_msgs.geometry import Pose
from robot_msgs.sensor import CameraInfo, Image
from robot_msgs.std import Header
from robot_msgs.vision import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesis, ObjectHypothesisWithPose

from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._perception_helpers import (
    Thing,
    bottle,
    box,
    can,
    class_colour,
    class_ids,
    cup,
    drawing,
    inner,
    look_at,
    project,
    render,
    srgb,
    state,
)

KEY = "object_detection"
CAMERA = "camera"  # the entity name of the image and its boxes: /image/camera
TRAIL = 10  # frames of box-centre history drawn behind each tracked object


# ── drawing one frame ────────────────────────────────────────────────────────


def _series_key(detection: Detection2D) -> tuple[str, str]:
    """(entity segment, legend name): per track when the detector tracks, else per class."""
    if detection.id:
        return f"track_{part(detection.id)}", f"{detection.class_id or 'no class'} #{detection.id}"
    return f"class_{part(detection.class_id or 'none')}", detection.class_id or "no class"


def log(rec: Any, image: Image | None = None, detections: Detection2DArray | None = None) -> None:
    draw = drawing(rec)
    if image is not None:
        robot_msg_viz.log(image, CAMERA, recording=draw)
    if detections is None:
        return
    memo = state(rec).memory.setdefault(KEY, {"series": {}, "classes": [], "trails": {}})
    names = [d.class_id for d in detections.detections]
    ids = class_ids(rec, names)  # the root class map, in robot_msg_viz's first-appearance order
    robot_msg_viz.log(detections, CAMERA, recording=draw)  # the boxes, class-coloured and labelled on the image

    # score per track (or per class when untracked); a track missing from this frame gets a gap
    best: dict[str, float] = {}
    for detection, class_id in zip(detections.detections, ids, strict=True):
        key, legend = _series_key(detection)
        if key not in memo["series"]:
            colour = class_colour(class_id)
            inner(rec).log(
                f"{KEY}/score/{key}",
                rr.SeriesLines(colors=[colour], names=[legend], widths=[2.0]),
                rr.SeriesPoints(colors=[colour], names=[legend], markers=["circle"], marker_sizes=[3.0]),
                static=True,
            )
        memo["series"][key] = True
        best[key] = max(best.get(key, -math.inf), detection.score)
    for key, score in best.items():
        draw.log(f"{KEY}/score/{key}", rr.Scalars(score))
    for key, shown in memo["series"].items():
        if key not in best and shown:
            draw.log(f"{KEY}/score/{key}", rr.Scalars(math.nan))  # WHY NaN: it breaks the line, so a miss shows
            memo["series"][key] = False

    # detections per class (every class seen so far, 0 when absent from this frame)
    for name, class_id in zip(names, ids, strict=True):
        label = name or "no class"
        if label not in memo["classes"]:
            memo["classes"].append(label)
            inner(rec).log(
                f"{KEY}/count/{part(label)}",
                rr.SeriesLines(colors=[class_colour(class_id)], names=[label], widths=[2.0],
                               interpolation_mode="StepAfter"),
                static=True,
            )  # fmt: skip
    for label in memo["classes"]:
        count = sum(1 for name in names if (name or "no class") == label)
        draw.log(f"{KEY}/count/{part(label)}", rr.Scalars(float(count)))

    # trails: the recent box centres of each track in this frame
    trails: dict[str, list[list[float]]] = memo["trails"]
    present = {}
    for detection, class_id in zip(detections.detections, ids, strict=True):
        if detection.id:
            centre = [detection.bbox.center.x, detection.bbox.center.y]
            trails[detection.id] = (trails.get(detection.id, []) + [centre])[-TRAIL:]
            present[detection.id] = class_id
    for track in list(trails):
        if track not in present:
            del trails[track]
    strips = [trails[track] for track in present if len(trails[track]) > 1]
    draw.log(
        f"image/{CAMERA}/trails",
        rr.LineStrips2D(strips, class_ids=[present[t] for t in present if len(trails[t]) > 1],
                        radii=rr.Radius.ui_points(1.5), draw_order=5.0, show_labels=False),
    )  # fmt: skip

    # this frame, as numbers: best first, with the runner-up class (how sure: the gap to the next guess)
    draw.log(f"{KEY}/frame", rr.TextDocument(_frame_table(detections), media_type="text/markdown"))


def _frame_table(detections: Detection2DArray) -> str:
    found = sorted(detections.detections, key=lambda d: -d.score)
    if not found:
        return "Nothing detected in this frame."
    lines = [
        "| track | class | score | runner-up | box |",
        "|---|---|---:|---|---|",
    ]
    for detection in found:
        guesses = sorted(detection.results, key=lambda r: -r.hypothesis.score)
        runner_up = f"{guesses[1].hypothesis.class_id} {guesses[1].hypothesis.score:.2f}" if len(guesses) > 1 else "–"
        box = detection.bbox
        lines.append(
            f"| {'#' + detection.id if detection.id else '–'} | {detection.class_id or 'no class'} "
            f"| {detection.score:.2f} | {runner_up} | {box.size_x:.0f} × {box.size_y:.0f} px |"
        )
    return "\n".join(lines)


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin=f"/image/{CAMERA}",
                name="camera · what the detector saw",
                defaults=[rr.Image.from_fields(magnification_filter="Linear")],  # a small frame stays smooth
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin=f"/{KEY}/score",
                    name="score per track",
                    axis_y=rrb.ScalarAxis(range=(0.0, 1.0)),
                    # WHY no legend: one entry per track grows without bound on a live stream, and a line's
                    # colour is its class (keyed in the plot below); hover a line for its track.
                    plot_legend=rrb.PlotLegend(visible=False),
                ),
                rrb.TimeSeriesView(
                    origin=f"/{KEY}/count", name="detections per class", plot_legend=rrb.PlotLegend("LeftTop")
                ),
                rrb.TextDocumentView(origin=f"/{KEY}/frame", name="this frame · best first"),
                row_shares=[3.2, 3.8, 4.4],
            ),
            column_shares=[1, 1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="expanded"),
    )


# ── the sample: a top-down camera over a waste-sorting conveyor ──────────────

STAMP = 1_764_028_800.0  # a fixed epoch second, so the sample is deterministic
RATE_HZ = 10.0
DURATION_S = 3.0
BELT_SPEED = 0.12  # m/s, along world +x = image right
BELT_HALF_WIDTH = 0.26
CAMERA_INFO = CameraInfo.from_intrinsics(
    300.0, 300.0, 159.5, 119.5, 320, 240, header=Header(stamp=STAMP, frame_id="camera_color_optical_frame")
)
CAMERA_POSE = look_at((0.0, 0.0, 0.8), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))  # 0.8 m above the belt, looking down
STAIN = (-0.31, -0.19)  # belt coordinates (m) of an oil stain the detector twice calls a can
MISSED = {(21, "1")}  # (frame, track): the bottle (#1) is missed in frame 21 (t = 2.1 s)
FALSE_POSITIVE = {26: 0.33, 27: 0.37}  # frame -> score of the stain's "can"


# (track id, maker, start x (m), y (m), yaw (deg), base score, how much its score wanders)
# WHY the bottle first: classes take colours in first-appearance order, and bottle, can, box, cup is the order the
# other perception dashboards meet them in, so a class has one colour across them.
_ITEMS = (
    ("1", bottle, -0.22, 0.10, 25.0, 0.92, 0.015),
    ("2", lambda x, y, yaw: can(x, y), 0.24, -0.11, 0.0, 0.89, 0.02),
    ("3", lambda x, y, yaw: box(x, y, yaw), -0.08, -0.07, -12.0, 0.84, 0.025),
    ("4", lambda x, y, yaw: cup(x, y), -0.60, -0.10, 0.0, 0.63, 0.045),
    ("5", lambda x, y, yaw: can(x, y, (30, 70, 170)), -0.44, 0.13, 0.0, 0.88, 0.02),
)


def _belt(offset: float) -> Any:
    rubber, cleat, stain, steel, gap, floor = (srgb(c) for c in (
        (54, 56, 60), (76, 78, 84), (33, 33, 35), (150, 154, 160), (28, 28, 30), (44, 45, 48)))  # fmt: skip

    def paint(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        along = x - offset  # belt coordinates move with the belt
        out = np.tile(rubber, (len(x), 1))
        out[np.mod(along, 0.12) < 0.012] = cleat
        out[((along - STAIN[0]) / 0.034) ** 2 + ((y - STAIN[1]) / 0.022) ** 2 < 1.0] = stain
        side = np.abs(y)
        out[side > BELT_HALF_WIDTH] = gap
        out[(side > BELT_HALF_WIDTH + 0.006) & (side < BELT_HALF_WIDTH + 0.05)] = steel  # the side guards
        out[side >= BELT_HALF_WIDTH + 0.05] = floor
        return out

    return paint


def _visible_box(owner: np.ndarray, index: int, thing: Thing) -> tuple[float, float, float, float, float] | None:
    """(x0, y0, x1, y1, visible share) of what `thing` shows in the frame, or None when it is out of view."""
    rows, cols = np.nonzero(owner == index)
    if rows.size == 0:
        return None
    full = project(CAMERA_INFO, CAMERA_POSE, thing.corners())
    (u0, v0), (u1, v1) = np.nanmin(full, axis=0), np.nanmax(full, axis=0)
    w, h = CAMERA_INFO.width, CAMERA_INFO.height
    clipped = max(0.0, min(u1, w) - max(u0, 0.0)) * max(0.0, min(v1, h) - max(v0, 0.0))
    share = clipped / max((u1 - u0) * (v1 - v0), 1e-9)
    return float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1), share


def _hypotheses(cls: str, score: float) -> tuple[ObjectHypothesisWithPose, ...]:
    runner_up = {"cup": "can", "can": "cup", "bottle": "can", "box": "cup"}[cls]
    second = round(max(0.01, (1.0 - score) * (0.6 if cls == "cup" else 0.3)), 3)
    return (
        ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(cls, round(score, 3)), pose=Pose()),
        ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(runner_up, second), pose=Pose()),
    )


@functools.cache
def _ticks() -> tuple[tuple[float, Image, Detection2DArray], ...]:
    noise = np.random.default_rng(20261124)
    wobble = {track: 0.0 for track, *_ in _ITEMS}
    frames = []
    for frame in range(int(round(DURATION_S * RATE_HZ)) + 1):
        t = frame / RATE_HZ
        shift = BELT_SPEED * t
        things = [make(x0 + shift, y, math.radians(yaw)) for _, make, x0, y, yaw, _, _ in _ITEMS]
        picture = render(CAMERA_INFO, CAMERA_POSE, things, _belt(shift), supersample=2)
        header = Header(stamp=STAMP + t, frame_id="camera_color_optical_frame")
        detections = []
        for index, (track, _, _, _, _, base, spread) in enumerate(_ITEMS):
            wobble[track] = 0.7 * wobble[track] + spread * noise.normal()  # scores wander, frame to frame
            seen = _visible_box(picture.owner, index, things[index])
            if seen is None or seen[4] < 0.45 or (frame, track) in MISSED:
                continue
            x0, y0, x1, y1, share = seen
            jitter = noise.normal(0.0, 0.8, 4)
            bbox = BoundingBox2D.from_xyxy(*(round(float(v), 1) for v in (x0, y0, x1, y1) + jitter))
            score = float(np.clip((base + wobble[track]) * (0.55 + 0.45 * min(share, 1.0)), 0.05, 0.99))
            cls = things[index].cls
            detections.append(Detection2D(header=header, results=_hypotheses(cls, score), bbox=bbox, id=track))
        if frame in FALSE_POSITIVE:
            centre = np.array([[STAIN[0] + shift, STAIN[1], 0.0]])
            u, v = project(CAMERA_INFO, CAMERA_POSE, centre)[0]
            half = 0.04 * CAMERA_INFO.fx / CAMERA_POSE[2, 3]
            bbox = BoundingBox2D.from_xyxy(round(u - half, 1), round(v - half * 0.8, 1), round(u + half, 1),
                                           round(v + half * 0.8, 1))  # fmt: skip
            results = (
                ObjectHypothesisWithPose(hypothesis=ObjectHypothesis("can", FALSE_POSITIVE[frame]), pose=Pose()),
                ObjectHypothesisWithPose(hypothesis=ObjectHypothesis("cup", 0.12), pose=Pose()),
            )
            detections.append(Detection2D(header=header, results=results, bbox=bbox, id="6"))
        frames.append((
            round(t, 3),
            Image.from_array(picture.rgb, header=header),
            Detection2DArray(header=header, detections=tuple(detections)),
        ))  # fmt: skip
    return tuple(frames)


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    return [(t, {"image": image, "detections": detections}) for t, image, detections in _ticks()]


register(Dashboard(
    key=KEY,
    title="Object detection",
    category="perception",
    summary="What did the detector see in each frame, and how sure was it over time? The frame with its "
    "class-coloured boxes, score per track, detections per class, and this frame's detections as numbers.",
    consumes=(Image, Detection2DArray),
    inputs={"image": "the camera frame", "detections": "what the detector found in it (boxes in its pixels)"},
    viewers={"rerun": RerunView(blueprint=blueprint, log=log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        'dash.show("object_detection", image=frame, detections=dets)   # one frame\n'
        'with dash.stream("object_detection") as live:                 # live\n'
        "    for t, frame, dets in camera:\n"
        "        live.log(t=t, image=frame, detections=dets)"
    ),
))  # fmt: skip
