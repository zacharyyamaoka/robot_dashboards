"""3D detection: where are the objects in 3D, and what are they?

Inputs: ``cloud`` (sensor.PointCloud), ``detections`` (vision.Detection3DArray, in the cloud's frame), ``camera``
(sensor.CameraInfo), and optionally ``image`` (sensor.Image, that camera's frame) and ``camera_pose``
(geometry.TransformStamped: the camera's optical frame in the cloud's frame, as tf gives it).

Views:
- 3D: the cloud, an oriented box per detection in its class colour labelled "class score", and the camera's
  frustum where it stood, so "where" reads against the scene and the sensor;
- the camera: its image with the 3D boxes projected through its intrinsics (Rerun's objectron pattern), the check
  that a box sits on the object the camera saw;
- the detections as numbers: class, score, runner-up, centre (m), size (mm), yaw (deg), range from the camera.

Without ``camera_pose`` the cloud and boxes are taken to be in the camera's own optical frame.

Sample: a head camera looking down at a work table with a bottle, two cans, a box and a cup, ray-cast in numpy;
its depth back-projected with robot_msgs' ``PointCloud.from_depth``, moved into base_link and cropped to the table;
detections are the true boxes with a few millimetres and degrees of error and plausible scores (the far, small
can lowest).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz
from robot_msgs.geometry import Pose, Transform, TransformStamped, Vector3
from robot_msgs.sensor import CameraInfo, Image, PointCloud
from robot_msgs.std import Header
from robot_msgs.vision import BoundingBox3D, Detection3D, Detection3DArray, ObjectHypothesis, ObjectHypothesisWithPose

from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._perception_helpers import (
    OPTICAL_FRAME,
    TABLE_CAMERA,
    TABLE_CAMERA_POSE,
    TABLE_STAMP,
    class_ids,
    drawing,
    inner,
    tabletop,
    tabletop_depth_mm,
    tabletop_things,
)

KEY = "detection_3d"
CAMERA = "camera"  # /world/camera (pose), /world/camera/image (pinhole), /image/camera (its picture)
CLOUD = "cloud"  # /world/cloud
DETECTIONS = "detections"  # /world/detections
# What robot_msg_viz draws that this question does not need: on the camera's image plane its pixel grid, u/v arrows
# and principal point (clutter over the picture), and around the camera's tf frame its name, its line home and its
# parent triad (useful in 3D, nonsense projected into the camera's own view).
_IMAGE_PLANE_EXTRAS = ("image/grid", "image/uv", "image/principal_point")
_TF_EXTRAS = ("name", "link", "parent")


def _without(extras: tuple[str, ...]) -> list[str]:
    return [f"- /world/{CAMERA}/{extra}/**" for extra in extras]


# ── drawing ──────────────────────────────────────────────────────────────────


def log(
    rec: Any,
    cloud: PointCloud | None = None,
    detections: Detection3DArray | None = None,
    camera: CameraInfo | None = None,
    image: Image | None = None,
    camera_pose: TransformStamped | None = None,
) -> None:
    draw = drawing(rec)
    if camera_pose is not None:
        robot_msg_viz.log(camera_pose, CAMERA, recording=draw)  # places /world/camera
    if camera is not None:
        robot_msg_viz.log(camera, CAMERA, recording=draw)  # the frustum at /world/camera/image
    if image is not None:
        robot_msg_viz.log(image, CAMERA, recording=draw)
        # WHY: the picture lives at /image/camera; this puts it in the pinhole's image frame, so the camera view
        # (origin /world/camera/image) shows it under the projected 3D boxes.
        inner(rec).log(f"image/{CAMERA}", rr.CoordinateFrame(f"tf#/world/{CAMERA}/image"), static=True)
    if cloud is not None:
        robot_msg_viz.log(cloud, CLOUD, recording=draw)
    if detections is not None:
        class_ids(rec, [d.class_id for d in detections.detections])  # the root class map
        robot_msg_viz.log(detections, DETECTIONS, recording=draw)
        draw.log(f"{KEY}/table", rr.TextDocument(_table(detections, camera_pose), media_type="text/markdown"))


def _table(detections: Detection3DArray, camera_pose: TransformStamped | None) -> str:
    found = sorted(detections.detections, key=lambda d: -d.score)
    if not found:
        return "Nothing detected."
    eye = camera_pose.transform.translation.as_array() if camera_pose is not None else np.zeros(3)
    lines = [
        "| class | score | runner-up | centre (m) | size (mm) | yaw | range |",
        "|---|---:|---|---|---|---:|---:|",
    ]
    for detection in found:
        guesses = sorted(detection.results, key=lambda r: -r.hypothesis.score)
        runner_up = f"{guesses[1].hypothesis.class_id} {guesses[1].hypothesis.score:.2f}" if len(guesses) > 1 else "–"
        box = detection.bbox
        centre = box.center.position.as_array()
        size = box.size.as_array() * 1000.0
        yaw = math.degrees(_yaw(box.center))
        lines.append(
            f"| {detection.class_id or 'no class'} | {detection.score:.2f} | {runner_up} "
            f"| {centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f} | {size[0]:.0f} × {size[1]:.0f} × {size[2]:.0f} "
            f"| {yaw:+.0f}° | {np.linalg.norm(centre - eye):.2f} m |"
        )
    return "\n".join(lines)


def _yaw(pose: Pose) -> float:
    """Rotation about z (radians) of a box's frame, from its x axis."""
    x_axis = pose.orientation.to_matrix()[:, 0]
    return math.atan2(x_axis[1], x_axis[0])


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="3D · cloud, boxes and the camera",
                contents=["/world/**", *_without(_IMAGE_PLANE_EXTRAS)],
                eye_controls=rrb.EyeControls3D(
                    position=(-0.12, -0.72, 0.74), look_target=(0.55, 0.02, 0.14), eye_up=(0.0, 0.0, 1.0)
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"/world/{CAMERA}/image",
                    name="camera · the 3D boxes projected",
                    # WHY without the cloud: projected, its points paint over the picture they came from
                    contents=[
                        "/world/**", f"- /world/{CLOUD}/**", f"/image/{CAMERA}",
                        *_without(_IMAGE_PLANE_EXTRAS + _TF_EXTRAS),
                    ],  # fmt: skip
                    defaults=[
                        rr.Image.from_fields(magnification_filter="Linear"),
                        # WHY wireframe: a projected box is otherwise filled, which tints the object it encloses
                        rr.Boxes3D.from_fields(fill_mode="MajorWireframe"),
                    ],
                ),
                rrb.TextDocumentView(origin=f"/{KEY}/table", name="detections · best first"),
                row_shares=[3, 2],
            ),
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),  # one moment: the time panel would only take room
    )


# ── the sample: a head camera over a work table ──────────────────────────────

_SCORES = {0: (0.93, "can", 0.04), 1: (0.90, "cup", 0.05), 2: (0.86, "cup", 0.06), 3: (0.71, "can", 0.18),
           4: (0.58, "cup", 0.22)}  # fmt: skip  # thing index -> (score, runner-up, its score)


@functools.cache
def _inputs() -> dict[str, Any]:
    header = Header(stamp=TABLE_STAMP, frame_id="base_link")
    pose = TransformStamped(header, OPTICAL_FRAME, Transform.from_matrix(TABLE_CAMERA_POSE))
    picture = tabletop()
    camera_header = Header(stamp=TABLE_STAMP, frame_id=OPTICAL_FRAME)
    image = Image.from_array(picture.rgb, header=camera_header)
    depth = Image.from_array(tabletop_depth_mm(), header=camera_header)
    cloud = PointCloud.from_depth(depth, TABLE_CAMERA, image)  # robot_msgs' back-projection
    cloud = cloud.transformed(pose.transform).cropped((0.3, -0.42, -0.02), (1.1, 0.42, 0.3)).downsampled(0.004)
    cloud = cloud.replace(header=header)
    noise = np.random.default_rng(3)
    detections = []
    for index, thing in enumerate(tabletop_things()):
        centre = np.asarray(thing.box_centre) + noise.normal(0.0, 0.004, 3)
        size = np.asarray(thing.box_size) * (1.0 + noise.normal(0.0, 0.03, 3))
        yaw = thing.box_yaw + math.radians(noise.normal(0.0, 3.0))
        box_pose = Pose.from_xyz_rpy(*np.round(centre, 4), 0.0, 0.0, round(yaw, 4))
        score, runner_up, second = _SCORES[index]
        results = (
            ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(thing.cls, score), pose=box_pose),
            ObjectHypothesisWithPose(hypothesis=ObjectHypothesis(runner_up, second), pose=box_pose),
        )
        bbox = BoundingBox3D(center=box_pose, size=Vector3(*np.round(size, 4)))
        detections.append(Detection3D(header=header, results=results, bbox=bbox, id=str(index + 1)))
    return {
        "cloud": cloud,
        "detections": Detection3DArray(header=header, detections=tuple(detections)),
        "camera": TABLE_CAMERA,
        "image": image,
        "camera_pose": pose,
    }


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    inputs = _inputs()
    # the pose first: it places the camera the CameraInfo then draws under
    order = ("camera_pose", "camera", "image", "cloud", "detections")
    return [(None, {name: inputs[name] for name in order})]


register(Dashboard(
    key=KEY,
    title="3D detection",
    category="perception",
    summary="Where are the objects in 3D, and what are they? The cloud with a class-coloured box per detection and "
    "the camera's frustum, the boxes projected onto the camera's image, and each detection's numbers.",
    consumes=(PointCloud, Detection3DArray, CameraInfo, Image, TransformStamped),
    inputs={
        "cloud": "the point cloud the detector ran on",
        "detections": "its 3D boxes, in the cloud's frame",
        "camera": "the camera's intrinsics",
        "image": "the camera's picture (optional)",
        "camera_pose": "where the camera is in the cloud's frame, tf parent -> optical frame (optional)",
    },
    viewers={"rerun": RerunView(blueprint=blueprint, log=log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        'dash.show("detection_3d", cloud=cloud, detections=boxes, camera=info, image=frame, camera_pose=tf)\n'
        'with dash.stream("detection_3d") as live:   # live\n'
        "    live.log(t=t, cloud=cloud, detections=boxes, camera=info, image=frame, camera_pose=tf)"
    ),
))  # fmt: skip
