"""Trajectory comparison: did the run follow its target, where and by how much did it deviate?

    import robot_dashboards as dash

    dash.show("trajectory_comparison", target=planned, run=measured)                          # tool by FK
    dash.show("trajectory_comparison", target=planned, run=measured, tool=run_tcp, target_tool=planned_tcp)

- ``target``: the JointTrajectory the arm was asked to follow (the plan, the command).
- ``run``: the JointTrajectory it actually did (encoders, a sim rollout); positions alone are enough.
- ``tool`` / ``target_tool`` (optional, both or neither): each one's tool PoseTrajectory in the URDF's root frame;
  without them the tool point is the URDF's last link, by forward kinematics.

What you see is traj_compare's "Target vs run" Rerun viewer (``traj_compare.rerun_target_run``, the pick round's V2),
called, not copied: robot_msgs' stock URDF trajectory viewer for the run (the steel arm, its tool path, the q / qd /
qdd / qddd, limit-card and TCP tabs), with the target as a wide pale line under every plot; the run's deviation from
the target magnified ×N as a thin magenta skeleton on the arm, an arrow from the target's tool point to the ×N one
labelled with the true offset in mm, and the target's tool path dashed blue; and an Error tab, first and open: the
tool error in mm, each joint's error in degrees, and target vs run at the playhead.

WHY a dependency and not a rebuild (Zach, 2026-11-25: "a use case for trajectory comparison (that is basically like
our rerun trajectory comparison viewer exactly)"): traj_compare is public, so the dashboard runs its viewer and the
two can never drift. The arm is ``URDF`` (the shipped UR5e); set it to your URDF for another arm.

Entity layout: traj_compare's V2 (robot_msgs' ``/world/run/...``, ``/positions/<joint>/{reference,run}``, ...,
``/error/{tool,joint}/...``, ``/notes/error``).

The sample: the target is a real UR5e plan (bam_ruckig through four tool poses, the trajectory_tracking_agent
dataset's workspace-dev-029, ``assets/data/workspace-dev-029.npz``); the run is that plan as a tracked arm would
follow it (robot_msg_viz's ``tracked_run`` recipe): each joint lags its command through a 12 ms first-order servo,
the shoulder lift and elbow sag a little under gravity, and the encoders read with 0.01° of noise; positions only,
100 Hz. Both tool paths carry the plan's 145 mm tool: the target's is the plan's own, the run's is the same tool on
the run's joints by forward kinematics.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import rerun.blueprint as rrb
from robot_msgs.std import Header
from robot_msgs.trajectory import JointLimits, JointTrajectory, PoseTrajectory

# WHY traj_compare's _view (its one constructor behind write_target_vs_run_rrd and target_vs_run_blueprint): the public
# entry points write a file or return the layout, and a dashboard logs into the caller's stream. CI pins traj_compare to
# a commit (.github/workflows/site.yml), so a rename there fails this repo's tests before it reaches a user.
from traj_compare.rerun_target_run import _view

from robot_dashboards.assets import UR5E_TRAJECTORY, UR5E_URDF
from robot_dashboards.core import Dashboard, RerunView, register
from robot_dashboards.dashboards._motion_state_helpers import forward_kinematics, inner, tip_link

KEY = "trajectory_comparison"
URDF = UR5E_URDF  # the robot both trajectories are drawn on; set it to your URDF for another arm
# the sample run's servo, sag and encoder (robot_msg_viz.views.compare.tracked_run's numbers)
SERVO_S = 0.012  # first-order lag time constant
SAG_DEG = (0.0, -0.15, 0.08, 0.0, 0.0, 0.0)  # shoulder lift and elbow droop under gravity
NOISE_DEG = 0.01  # encoder noise, 1 sigma
SEED = 11


def _target_run(target: JointTrajectory, run: JointTrajectory, tool: PoseTrajectory | None,
                target_tool: PoseTrajectory | None) -> Any:  # fmt: skip
    return _view(run, target, str(URDF), tool, target_tool, None, None, None, "ruler")


def log(recording: Any, *, target: JointTrajectory | None = None, run: JointTrajectory | None = None,
        tool: PoseTrajectory | None = None, target_tool: PoseTrajectory | None = None) -> None:  # fmt: skip
    if target is None or run is None:
        return  # a comparison needs both; traj_compare's V2 draws nothing for half of one
    # WHY the inner stream: both trajectories carry their own clock (V1's: t = 0 at the run's first sample)
    recording = inner(recording)
    view = _target_run(target, run, tool, target_tool)
    if (target, run, tool, target_tool) != _sample_inputs():
        # WHY only off the sample: V2's layout is built on its inputs (the ×N, the joints, the eye); the fixed blueprint
        # is the sample's, and every copy of the stock robot page is ~0.5 MB in the file
        recording.send_blueprint(view.blueprint())
    view.log(recording)


# ── the sample: a real UR5e plan, and that plan as a tracked arm followed it ────────────────────────────────────────


@functools.cache
def _sample_inputs() -> tuple[JointTrajectory, JointTrajectory, PoseTrajectory, PoseTrajectory]:
    data = np.load(UR5E_TRAJECTORY)
    names = tuple(str(n) for n in data["names"])
    t = np.asarray(data["t"], dtype=np.float64)
    limits = JointLimits(joint_names=names, min_position=data["lo"], max_position=data["hi"],
                         max_velocity=data["vmax"], max_acceleration=data["amax"])  # fmt: skip
    header = Header(meta={"source": "trajectory_tracking_agent workspace/v2 workspace-dev-029",
                          "time_law": "bam_ruckig"})
    target = JointTrajectory(t=t, joint_names=names, positions=data["q"], velocities=data["qd"],
                             accelerations=data["qdd"], limits=limits, header=header)  # fmt: skip
    # the run: a first-order servo on each joint's command, gravity sag, encoder noise (seeded, so deterministic)
    command = np.asarray(data["q"], dtype=np.float64)
    measured = np.empty_like(command)
    measured[0] = command[0]
    for k, step in enumerate(np.diff(t)):
        measured[k + 1] = measured[k] + min(step / SERVO_S, 1.0) * (command[k] - measured[k])
    noise = np.random.default_rng(SEED).normal(0.0, NOISE_DEG, measured.shape)
    measured = measured + np.radians(SAG_DEG) + np.radians(noise)
    run = JointTrajectory(t=t, joint_names=names, positions=measured,
                          header=Header(meta={"source": "workspace-dev-029 through a 12 ms servo, sag and encoder noise"}))
    # the tool: the plan's own path for the target; the same flange -> tool transform on the run's joints
    planned = np.tile(np.eye(4), (len(t), 1, 1))
    planned[:, :3, :3], planned[:, :3, 3] = data["tool_rot"], data["tool_xyz"]
    flange = tip_link(URDF)
    flange_tool = np.linalg.inv(forward_kinematics(URDF, names, command[:1])[flange][0]) @ planned[0]
    followed = forward_kinematics(URDF, names, measured)[flange] @ flange_tool
    target_tool = PoseTrajectory.from_matrices(t, planned, child_frame_id="tcp")
    tool = PoseTrajectory.from_matrices(t, followed, child_frame_id="tcp")
    return target, run, tool, target_tool


def sample() -> list[tuple[float | None, dict[str, Any]]]:
    target, run, tool, target_tool = _sample_inputs()
    return [(None, {"target": target, "run": run, "tool": tool, "target_tool": target_tool})]


@functools.cache
def _sample_layout() -> rrb.Blueprint:
    return _target_run(*_sample_inputs()).blueprint()


register(Dashboard(
    key=KEY,
    title="Trajectory comparison",
    category="motion",
    summary="Did the run follow its target, where and by how much did it deviate? The run on the stock URDF arm with "
            "its target pale under every plot, the deviation magnified ×N as a skeleton and an arrow labelled in mm, "
            "and an Error tab: tool error in mm, each joint's error in degrees, target vs run at the playhead.",
    consumes=(JointTrajectory, PoseTrajectory),
    inputs={
        "target": "the JointTrajectory the arm was asked to follow (the plan, the command)",
        "run": "the JointTrajectory it actually did (encoders, a sim rollout); positions alone are enough",
        "tool": "optional: the run's tool PoseTrajectory (with target_tool; default: the URDF's last link by FK)",
        "target_tool": "optional: the target's tool PoseTrajectory (with tool)",
    },
    viewers={"rerun": RerunView(blueprint=lambda: _sample_layout(), log=log)},
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        "# planned: the JointTrajectory it was asked to follow; measured: what the arm did (positions are enough)\n"
        'dash.show("trajectory_comparison", target=planned, run=measured)'
    ),
))  # fmt: skip
