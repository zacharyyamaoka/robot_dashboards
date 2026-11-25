"""Shared by the motion-state lane's dashboards (trajectory_generation, robot_state): a URDF arm, its forward
kinematics, a camera on it, and joint labels.

WHY our own URDF arm here and not robot_msgs' `_urdf_copy`: that one is private to robot_msgs' JointTrajectory
viewer (it needs a whole trajectory up front); a live JointState arrives one sample at a time. The recipe is the
same, so the two arms look alike: Rerun's own URDF loader (`rerun.urdf.UrdfTree`), its chunks moved under
``/world/<name>/robot``, one transform entity per joint (fixed joints static, moving joints at each sample, from the
loader's own `compute_transform`), and one identity mount from ``tf#/world/<name>`` to the root link.
"""

from __future__ import annotations

import functools
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from robot_msg_viz._engine import part
from robot_msgs.geometry.rotations import euler_to_matrix, quat_from_axis_angle, quat_to_matrix

STEEL = (175, 180, 190, 255)  # robot_msgs' arm colour, so this arm reads like its trajectory viewer's
_DRAWABLE = {"model/stl", "model/obj", "model/gltf-binary", "model/gltf+json"}  # Rerun 0.40 cannot draw .dae
_ARMS: weakref.WeakKeyDictionary[Any, set[str]] = weakref.WeakKeyDictionary()


def inner(recording: Any) -> Any:
    """The RecordingStream behind robot_msg_viz's `Timed` (which lifts every log onto the timeline)."""
    return getattr(recording, "_inner", recording)


def short_joint(joint: str) -> str:
    """``shoulder_pan_joint`` -> ``shoulder_pan``: the URDF name without its suffix."""
    return joint[: -len("_joint")] if joint.endswith("_joint") and len(joint) > len("_joint") else joint


def joint_label(index: int, joint: str) -> str:
    """``J1 shoulder_pan``: the URDF name and its number from the base (prior-art sweep: names plus J1/J2)."""
    return f"J{index + 1} {short_joint(joint)}"


@functools.cache
def urdf_tree(urdf: str, frame_prefix: str = "") -> Any:
    import rerun.urdf as rurdf

    return rurdf.UrdfTree.from_file_path(Path(urdf), frame_prefix=frame_prefix or None)


def forward_kinematics(urdf: str | Path, joint_names: tuple[str, ...], positions: np.ndarray) -> dict[str, np.ndarray]:
    """Every link's pose in the URDF's root link, ``(N, 4, 4)``, from the URDF's own joint origins and axes.

    For framing a camera and for sample physics (a wrist sensor's gravity load); the viewer poses the drawn arm
    from the loader's own joint transforms, never from this."""
    tree = urdf_tree(str(urdf))
    q = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    samples = len(q)
    values = {str(name): q[:, i] for i, name in enumerate(joint_names)}
    poses = {tree.root_link().name: np.broadcast_to(np.eye(4), (samples, 4, 4))}
    pending = list(tree.joints())
    while pending:  # parents first, whatever order the URDF lists its joints in
        joint = next((j for j in pending if j.parent_link in poses), None)
        if joint is None:
            raise ValueError(f"URDF joints not connected to the root link: {[j.name for j in pending]}")
        origin = np.eye(4)
        origin[:3, :3] = euler_to_matrix(*joint.origin_rpy)
        origin[:3, 3] = joint.origin_xyz
        motion = np.broadcast_to(np.eye(4), (samples, 4, 4)).copy()
        value = values.get(joint.name)
        if value is not None and joint.joint_type in ("revolute", "continuous"):
            motion[:, :3, :3] = quat_to_matrix(quat_from_axis_angle(np.asarray(joint.axis, dtype=np.float64), value))
        elif value is not None and joint.joint_type == "prismatic":
            motion[:, :3, 3] = np.asarray(joint.axis, dtype=np.float64)[None, :] * value[:, None]
        poses[joint.child_link] = poses[joint.parent_link] @ origin @ motion
        pending.remove(joint)
    return poses


def tip_link(urdf: str | Path) -> str:
    """The URDF's deepest link (the flange of a serial arm)."""
    tree = urdf_tree(str(urdf))
    parent = {joint.child_link: joint.parent_link for joint in tree.joints()}

    def depth(link: str) -> int:
        count = 0
        while link in parent:
            link, count = parent[link], count + 1
        return count

    return max(parent, key=depth) if parent else tree.root_link().name


def eye(points: np.ndarray, *, distance_radii: float = 2.35, elevation: float = 0.55) -> rrb.EyeControls3D:
    """A camera on everything in `points` (metres, z up), from robot_msgs' front-right, about 29° above."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    low, high = points.min(axis=0), points.max(axis=0)
    look = 0.5 * (low + high)
    radius = max(0.5 * float(np.linalg.norm(high - low)), 0.1)
    direction = np.array([1.5, -1.9, 0.0]) / np.linalg.norm([1.5, -1.9])
    direction[2] = elevation
    direction /= np.linalg.norm(direction)
    return rrb.EyeControls3D(kind="Orbital", look_target=look.tolist(), eye_up=[0.0, 0.0, 1.0],
                             position=(look + distance_radii * radius * direction).tolist())  # fmt: skip


def log_arm(recording: Any, urdf: str | Path, name: str, colour: Any = STEEL) -> Any:
    """Draw the URDF's meshes at ``/world/<name>/robot`` once per recording (static); returns the loaded tree.

    Its frames are prefixed ``<name>/`` (``arm/wrist_3_link``), so a second arm or a sensor in a link's frame
    (``rr.CoordinateFrame("arm/wrist_3_link")``) can be placed against it."""
    stream = inner(recording)
    label = part(name)
    tree = urdf_tree(str(urdf), f"{label}/")
    drawn = _ARMS.setdefault(stream, set())
    if label in drawn:
        return tree
    drawn.add(label)
    entity, old_root = f"/world/{label}/robot", f"/{tree.name}"
    chunks = list(tree.stream(include_joint_transforms=False))
    kinds: dict[str, tuple[str, str | None]] = {}
    for chunk in chunks:
        batch = chunk.to_record_batch()
        names = batch.schema.names
        for column in names:
            archetype = column.split(":")[0] if ":" in column else ""
            if archetype and archetype not in ("InstancePoses3D", "CoordinateFrame"):
                media = None
                if "Asset3D:media_type" in names:
                    rows = batch.column("Asset3D:media_type").to_pylist()
                    media = rows[0][0] if rows and rows[0] else None
                kinds[chunk.entity_path] = (archetype, media)

    def drawable(path: str) -> bool:
        archetype, media = kinds.get(path, ("", None))
        return bool(archetype) and (archetype != "Asset3D" or media in _DRAWABLE)

    # WHY visuals first: a link draws its visual geometry when Rerun can, else its collision geometry, never both
    keep: set[str] = set()
    for link in [tree.root_link().name, *(joint.child_link for joint in tree.joints())]:
        visuals = [path for path in tree.get_visual_geometry_paths(link) if drawable(path)]
        keep.update(visuals or [path for path in tree.get_collision_geometry_paths(link) if drawable(path)])
    dropped = set(kinds) - keep
    stream.send_chunks([
        chunk.with_entity_path(entity + chunk.entity_path[len(old_root):])
        for chunk in chunks
        if not any(chunk.entity_path == path or chunk.entity_path.startswith(path + "/") for path in dropped)
    ])  # fmt: skip
    for path in sorted(keep):
        archetype = kinds[path][0]
        tint = (rr.Asset3D.from_fields(albedo_factor=colour) if archetype == "Asset3D"
                else rr.Mesh3D.from_fields(albedo_factor=colour) if archetype == "Mesh3D"
                else getattr(rr, archetype).from_fields(colors=[colour]))  # fmt: skip
        stream.log(entity + path[len(old_root):], tint, static=True)
    mount = rr.Transform3D(parent_frame=f"tf#/world/{label}", child_frame=f"{label}/{tree.root_link().name}")
    stream.log(f"{entity}/mount", mount, static=True)
    for joint in tree.joints():
        if joint.joint_type == "fixed":
            stream.log(f"{entity}/joint_transforms/{part(joint.name)}", joint.compute_transform(0.0, clamp=False),
                       static=True)  # fmt: skip
    return tree


def pose_arm(recording: Any, tree: Any, name: str, joint_names: tuple[str, ...], positions: np.ndarray) -> list[str]:
    """Put the arm at one sample: one transform per moving joint at the recording's time; returns the joints the
    URDF does not have (they are not drawn)."""
    values = {str(joint): float(value) for joint, value in zip(joint_names, positions, strict=True)}
    for joint in tree.joints():
        mimic = joint.mimic
        if mimic is not None and mimic.joint in values and joint.name not in values:
            values[joint.name] = mimic.multiplier * values[mimic.joint] + mimic.offset
    known = set()
    for joint in tree.joints():
        if joint.joint_type == "fixed" or joint.name not in values:
            continue
        known.add(joint.name)
        recording.log(f"/world/{part(name)}/robot/joint_transforms/{part(joint.name)}",
                      joint.compute_transform(values[joint.name], clamp=False))  # fmt: skip
    return [str(joint) for joint in joint_names if str(joint) not in known]
