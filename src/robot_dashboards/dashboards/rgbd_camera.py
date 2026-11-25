"""RGB-D camera: what does the RGB-D camera see, in 2D and in 3D?

Inputs: ``rgb`` (sensor.Image, rgb8), ``depth`` (sensor.Image, 16UC1 millimetres or 32FC1 metres, aligned to the
rgb), ``camera`` (sensor.CameraInfo, their intrinsics) and optionally ``camera_pose`` (geometry.TransformStamped:
the optical frame in the robot's frame, as tf gives it).

Views:
- 3D: the depth back-projected through the intrinsics into a cloud coloured by the rgb (robot_msgs'
  ``PointCloud.from_depth``), inside the camera's frustum: the shape the camera measured, holes and all;
- the rgb picture and the depth picture (coloured by distance, hover for metres), side by side in 2D;
- the depth in numbers: how much of the picture has a return, the near / median / far range, the field of view.

CameraInfo carries no pose: with ``camera_pose`` the camera stands where tf puts it (z up, as in base_link);
without it, it stands level at the origin looking along +x (ROS's camera_link convention), so "up" on screen is up
either way. The cloud stays in the optical frame it was measured in; only the camera entity moves.

Prior art: Rerun's rgbd and ARKit scenes examples (camera on the parent entity, its images as children).

Sample: a work table with a bottle, two cans, a box and a cup, ray-cast in numpy through a RealSense-D435-like
424 x 240 camera into a consistent rgb picture and 16-bit depth with stereo noise (~1 mm at 0.6 m, growing with the
square of the range), the stereo shadow beside each object and a few dropouts.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz
from robot_msgs.geometry import Transform, TransformStamped
from robot_msgs.sensor import CameraInfo, Image, PointCloud
from robot_msgs.std import Header

from robot_dashboards.core import Dashboard, register
from robot_dashboards.dashboards._perception_helpers import (
    OPTICAL_FRAME,
    TABLE_CAMERA,
    TABLE_CAMERA_POSE,
    TABLE_STAMP,
    drawing,
    inner,
    state,
    tabletop,
    tabletop_depth_mm,
)

KEY = "rgbd_camera"
CAMERA, RGB, DEPTH, CLOUD = "camera", "rgb", "depth", "cloud"
_IMAGE_PLANE_EXTRAS = ("image/grid", "image/uv", "image/principal_point")  # robot_msg_viz's, clutter here
# A level camera's optical frame (x right, y down, z forward) in a z-up frame looking along +x: ROS's camera_link ->
# camera_optical_frame rotation, rpy (-90°, 0, -90°). Columns: the optical x, y, z axes.
_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def log(
    rec: Any,
    rgb: Image | None = None,
    depth: Image | None = None,
    camera: CameraInfo | None = None,
    camera_pose: TransformStamped | None = None,
) -> None:
    draw = drawing(rec)
    memo = state(rec).memory.setdefault(KEY, {})
    if camera_pose is not None:
        robot_msg_viz.log(camera_pose, CAMERA, recording=draw)
        memo["posed"] = True
    elif not memo.get("posed") and not memo.get("levelled"):
        memo["levelled"] = True  # no tf (yet): stand the camera level at the origin; a later pose overrides it
        inner(rec).log(f"world/{CAMERA}", rr.Transform3D(mat3x3=_LEVEL), static=True)
    if camera is not None:
        robot_msg_viz.log(camera, CAMERA, recording=draw)
        memo["camera"] = camera
    if rgb is not None:
        robot_msg_viz.log(rgb, RGB, recording=draw)
        memo["rgb"] = rgb
    if depth is None:
        return
    robot_msg_viz.log(depth, DEPTH, recording=draw)
    near_far = _near_far(depth)
    if near_far is not None:
        # WHY: the viewer spreads the colour map from 0, so a table 0.6-1.2 m away used only its upper half; this
        # spreads it over the depths actually seen (dark blue = near, dark red = far).
        draw.log(f"image/{DEPTH}", rr.DepthImage.from_fields(depth_range=near_far))
    camera = memo.get("camera")
    if camera is None or (depth.width, depth.height) != (camera.width, camera.height):
        return  # nothing to back-project through yet
    colour = rgb if rgb is not None else memo.get("rgb")
    if colour is not None and (
        (colour.width, colour.height) != (depth.width, depth.height) or colour.header.stamp != depth.header.stamp
    ):
        colour = None  # not the picture this depth belongs to: an uncoloured cloud beats a wrongly coloured one
    cloud = PointCloud.from_depth(depth, camera, colour)  # robot_msgs' back-projection, in the optical frame
    robot_msg_viz.log(cloud, CLOUD, recording=draw)
    # WHY a frame, not moved points: the cloud is in the camera's optical frame (/world/camera, where the pinhole
    # hangs), so it rides with whatever pose the camera has, and its numbers stay the ones the camera measured.
    inner(rec).log(f"world/{CLOUD}", rr.CoordinateFrame(f"tf#/world/{CAMERA}"), static=True)
    draw.log(f"{KEY}/summary", rr.TextDocument(_summary(depth, camera, len(cloud)), media_type="text/markdown"))


def _metres(depth: Image) -> np.ndarray:
    return depth.data.astype(float) / (1000.0 if depth.encoding == "16UC1" else 1.0)  # ROS: 16UC1 is mm


def _near_far(depth: Image) -> tuple[float, float] | None:
    """The 1st-99th percentile of the depths with a return, in the image's own units (what depth_range takes)."""
    values = depth.data.astype(float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    near, far = np.percentile(values, [1.0, 99.0])
    return (float(near), float(far)) if far > near else None


def _summary(depth: Image, camera: CameraInfo, points: int) -> str:
    metres = _metres(depth)
    valid = np.isfinite(metres) & (metres > 0.0)
    share = 100.0 * valid.mean()
    if valid.any():
        near, median, far = np.percentile(metres[valid], [0.0, 50.0, 100.0])
        ranges = f"{near:.2f} m nearest · {median:.2f} m median · {far:.2f} m farthest"
        scale = 1000.0 if depth.encoding == "16UC1" else 1.0
        low, high = (v / scale for v in (_near_far(depth) or (near * scale, far * scale)))
        ranges += f"  \ncolour: dark blue {low:.2f} m → dark red {high:.2f} m"
    else:
        ranges = "no returns"
    wide = math.degrees(math.atan(camera.cx / camera.fx) + math.atan((camera.width - camera.cx) / camera.fx))
    high = math.degrees(math.atan(camera.cy / camera.fy) + math.atan((camera.height - camera.cy) / camera.fy))
    return "\n".join([
        f"**{share:.1f} %** of pixels have depth · {points:,} points",
        "",
        f"{ranges}",
        "",
        f"{depth.width} × {depth.height} px {depth.encoding} · field of view {wide:.0f}° × {high:.0f}° · "
        f"fx {camera.fx:.0f} px",
    ])  # fmt: skip


def blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                name="3D · depth back-projected, coloured by rgb",
                contents=["/world/**", *[f"- /world/{CAMERA}/{extra}/**" for extra in _IMAGE_PLANE_EXTRAS]],
                # behind and to the left of a camera that looks forward and down at a table 0.5-1 m in front of it
                eye_controls=rrb.EyeControls3D(
                    position=(-0.12, -0.72, 0.74), look_target=(0.55, 0.02, 0.14), eye_up=(0.0, 0.0, 1.0)
                ),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(
                    origin=f"/image/{RGB}", name="rgb",
                    defaults=[rr.Image.from_fields(magnification_filter="Linear")],
                ),  # fmt: skip
                rrb.Spatial2DView(origin=f"/image/{DEPTH}", name="depth · near blue → far red (hover for m)"),
                rrb.TextDocumentView(origin=f"/{KEY}/summary", name="depth in numbers"),
                row_shares=[2, 2, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),  # one moment: the time panel would only take room
    )


@functools.cache
def _inputs() -> dict[str, Any]:
    header = Header(stamp=TABLE_STAMP, frame_id=OPTICAL_FRAME)
    pose = TransformStamped(
        Header(stamp=TABLE_STAMP, frame_id="base_link"), OPTICAL_FRAME, Transform.from_matrix(TABLE_CAMERA_POSE)
    )
    return {
        "camera_pose": pose,
        "camera": TABLE_CAMERA,
        "rgb": Image.from_array(tabletop().rgb, header=header),
        "depth": Image.from_array(tabletop_depth_mm(), header=header),
    }


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    return [(None, dict(_inputs()))]


register(Dashboard(
    key=KEY,
    title="RGB-D camera",
    category="perception",
    summary="What does the RGB-D camera see, in 2D and in 3D? The depth back-projected into a cloud coloured by "
    "the rgb, beside the rgb and depth pictures and how much of the picture has depth.",
    consumes=(Image, CameraInfo, TransformStamped),
    inputs={
        "rgb": "the colour picture (rgb8)",
        "depth": "the depth picture aligned to it (16UC1 mm or 32FC1 m)",
        "camera": "their intrinsics",
        "camera_pose": "where the camera is, tf parent -> optical frame (optional; else level at the origin)",
    },
    blueprint=blueprint,
    log=log,
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        'dash.show("rgbd_camera", rgb=colour, depth=depth, camera=info)\n'
        'with dash.stream("rgbd_camera") as live:   # live\n'
        "    live.log(t=t, rgb=colour, depth=depth, camera=info)"
    ),
))  # fmt: skip
