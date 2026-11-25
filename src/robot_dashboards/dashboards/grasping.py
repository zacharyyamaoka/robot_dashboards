"""Grasping: which grasps did the planner propose on this object, which is best, and is it feasible?

Inputs: the cloud the planner saw (a PointCloud of the object on its table), its candidates (a tuple of
GraspPose, each with a score) and the grasp it chose (optional).

- Left, the 3D scene: the cloud (robot_msg_viz's PointCloud view), every candidate as robot_msg_viz's parallel-gripper
  glyph (the jaws at the closing width, the approach arrow into the palm) coloured by score on its viridis ramp,
  the chosen one drawn by robot_msg_viz's GraspPose view (thick, open jaws faint, labelled). When the best-scored
  candidate is not feasible it is labelled too, and the cloud points it would hit are drawn red.
- Right, top: every candidate's score, best first; a bar is green when the candidate passes every check, red when
  it fails one (pass / fail is the only use of green and red here).
- Right, bottom: the chosen grasp's checks, and why the best-scored one was passed over.

The checks (`_manipulation_helpers.grasp_checks`) put a gripper the size of a Robotiq 2F-85 (40 mm fingers, 8 mm
thick, 20 mm wide, a 30 mm palm, an 85 mm stroke) at the grasp, open to its pre-grasp width, and count cloud points
inside the fingers, the palm, the open gripper swept 100 mm back along the approach, and between the jaws.

The sample: a mug (80 mm across, 95 mm tall, a handle) on a table as one RGB-D camera sees it (the far side and the
mug's shadow on the table are missing), voxel-downsampled to 2.5 mm, and 25 candidates of the kinds
Contact-GraspNet-style planners propose on a mug: pinches on the rim, grips on the handle and wide grips across the
body, scored by hand-made rules plus seeded noise. The top score is a deep rim pinch right over the handle, whose
outer finger lands on the handle; the chosen grasp is the best one that passes every check.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import robot_msg_viz as viz
from robot_msg_viz.views.manipulation import APPROACH, FINGER_DEPTH, jaws, score_colour
from robot_msgs.geometry import Pose
from robot_msgs.manipulation import GraspPose
from robot_msgs.sensor import PointCloud
from robot_msgs.std import Header

from robot_dashboards.core import Dashboard, register
from robot_dashboards.dashboards._manipulation_helpers import (
    HANDLE_ANGLE,
    HANDLE_ARC,
    HANDLE_CENTRE,
    MIN_BETWEEN,
    MUG_HEIGHT,
    MUG_RADIUS,
    MUG_WALL,
    STANDOFF,
    STROKE,
    GraspCheck,
    grasp_checks,
    mug_scene,
)

KEY = "grasping"
PASS, FAIL, INK = (60, 170, 90), (220, 60, 60), (37, 99, 160)
HIT = (235, 50, 50)


def ranked(grasps: tuple[GraspPose, ...]) -> list[tuple[int, GraspPose]]:
    """(rank from 1, grasp), best score first; a grasp without a score sorts last, the planner's order kept."""
    order = sorted(range(len(grasps)), key=lambda i: (grasps[i].score is None, -(grasps[i].score or 0.0), i))
    return [(rank + 1, grasps[i]) for rank, i in enumerate(order)]


def same(a: GraspPose, b: GraspPose) -> bool:
    return a == b or (np.allclose(a.pose.to_matrix(), b.pose.to_matrix(), atol=1e-9) and a.width == b.width)


def score_text(grasp: GraspPose) -> str:
    return "no score" if grasp.score is None else f"score {grasp.score:.2f}"


# ── log ──────────────────────────────────────────────────────────────────────


def log(rec: rr.RecordingStream, cloud: PointCloud | None = None, grasps: tuple[GraspPose, ...] | None = None,
        chosen: GraspPose | None = None) -> None:  # fmt: skip
    if cloud is not None:
        viz.log(cloud, "cloud", recording=rec)
    grasps = tuple(grasps or ())
    points = None if cloud is None else np.asarray(cloud.points, dtype=np.float64)
    checks = {id(g): grasp_checks(g, points) for g in grasps} if points is not None else {}
    order = ranked(grasps)
    for rank, grasp in order:
        draw_candidate(rec, rank, grasp, checks.get(id(grasp)), thick=False)
    if order:
        log_scores(rec, order, checks)
    best = order[0][1] if order else None
    pick = chosen if chosen is not None else best
    rank_of = {id(g): rank for rank, g in order}
    if pick is not None:
        viz.log(pick, "chosen", recording=rec)  # robot_msg_viz's GraspPose view: thick jaws, open width, label
        # its label says which candidate it is (same place and colour as the stock label, which it replaces)
        rank = next((rank_of[id(g)] for _, g in order if same(g, pick)), None)
        where = f"#{rank} of {len(order)}" if rank else "not a candidate"
        who = "chosen" if chosen is not None else "best score, no grasp chosen"
        rec.log("world/chosen/label", rr.Points3D(
            [[0.0, 0.0, -FINGER_DEPTH - APPROACH - 0.012]], radii=0.002, colors=[score_colour(pick.score)],
            labels=[f"{who} · {where} · {score_text(pick)} · close to {pick.width * 1000:.0f} mm"]), static=True)  # fmt: skip
    if best is not None and pick is not None and not same(best, pick):
        check = checks.get(id(best))
        reason = "" if check is None or check.feasible else " · fails: " + ", ".join(check.reasons())
        draw_candidate(rec, 1, best, check, thick=True, label=f"best score #1 · {score_text(best)}{reason}")
        if check is not None and len(check.hits):
            rec.log("world/best_hits", rr.Points3D(check.hits, radii=rr.Radius.ui_points(4.0), colors=[HIT]),
                    static=True)  # fmt: skip
    pick_check = None
    if pick is not None and points is not None:
        pick_check = grasp_checks(pick, points)
    rec.log(f"{KEY}/verdict", rr.TextDocument(verdict_card(order, checks, pick, pick_check, chosen is not None),
                                              media_type="text/markdown"), static=True)  # fmt: skip


def draw_candidate(rec: rr.RecordingStream, rank: int, grasp: GraspPose, check: GraspCheck | None, *, thick: bool,
                   label: str | None = None) -> None:  # fmt: skip
    """robot_msg_viz's glyph without its frame triad and text (25 of them would bury the mug): the jaws at the
    closing width and the approach arrow, coloured by score; the rank, score and verdict ride on a hidden label,
    so hovering the candidate reads them."""
    path = f"world/candidates/{'best' if thick else f'{rank:02d}'}"
    colour = score_colour(grasp.score)
    p = grasp.pose
    rec.log(path, rr.Transform3D(translation=p.position.as_array(),
                                 quaternion=[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
            static=True)  # fmt: skip
    jaws(rec, f"{path}/jaws", grasp.width, colour, 0.0025 if thick else 0.0009)
    behind = -FINGER_DEPTH - APPROACH
    rec.log(f"{path}/approach", rr.Arrows3D(origins=[[0.0, 0.0, behind]], vectors=[[0.0, 0.0, APPROACH]],
                                            radii=0.0016 if thick else 0.0005, colors=[colour]), static=True)  # fmt: skip
    verdict = "" if check is None else (" · feasible" if check.feasible else " · fails: " + ", ".join(check.reasons()))
    text = label or f"#{rank} · {score_text(grasp)} · close to {grasp.width * 1000:.0f} mm{verdict}"
    rec.log(f"{path}/label", rr.Points3D([[0.0, 0.0, behind - 0.006]], radii=0.001, colors=[FAIL if thick else colour],
                                         labels=[text], show_labels=thick), static=True)  # fmt: skip


def log_scores(rec: rr.RecordingStream, order: list[tuple[int, GraspPose]], checks: dict[int, GraspCheck]) -> None:
    """The scores best first, x = rank; green bars pass every check, red ones fail one (no cloud: one ink series)."""
    ranks = np.array([rank for rank, _ in order], dtype=np.float64)
    scores = np.array([g.score if g.score is not None else 0.0 for _, g in order], dtype=np.float64)
    if not checks:
        rec.log(f"{KEY}/scores/candidates", rr.BarChart(scores, abscissa=ranks, color=INK), static=True)
        return
    ok = np.array([checks[id(g)].feasible for _, g in order])
    for name, keep, colour in (("feasible", ok, PASS), ("fails_a_check", ~ok, FAIL)):
        if keep.any():
            rec.log(f"{KEY}/scores/{name}", rr.BarChart(scores[keep], abscissa=ranks[keep], color=colour), static=True)


def verdict_card(order: list[tuple[int, GraspPose]], checks: dict[int, GraspCheck], pick: GraspPose | None,
                 check: GraspCheck | None, given: bool) -> str:  # fmt: skip
    """Markdown: the chosen grasp, each check ✅ / ❌ with its count, then the best-scored one and the tally."""
    if pick is None:
        return "### No grasp\n\nNo candidates and no chosen grasp were given."
    rank = next((r for r, g in order if same(g, pick)), None)
    where = f"#{rank} of {len(order)}" if rank else "not among the candidates"
    who = "Chosen" if given else "Best-scored (no grasp was chosen)"
    lines = [f"### {who}: {where} · {score_text(pick)}" + (f" · {pick.object_id}" if pick.object_id else ""), ""]
    if check is None:
        lines += ["No cloud was given, so nothing can be checked.", ""]
    else:
        lines += [f"**{'Feasible' if check.feasible else 'Not feasible'}** with a 2F-85-sized gripper, "
                  f"open {check.open_width * 1000:.0f} mm, closing to {pick.width * 1000:.0f} mm:", "",
                  "| check | |", "|---|---|"]  # fmt: skip
        rows = [
            (check.fingers == 0, f"open fingers clear of the cloud at the grasp ({check.fingers} points in them)"),
            (check.palm == 0, f"palm clear ({check.palm} points)"),
            (check.approach == 0, f"approach clear, {STANDOFF * 1000:.0f} mm straight in ({check.approach} points)"),
            (check.between >= MIN_BETWEEN, f"something to grip: {check.between} points between the jaws "
                                           f"(≥ {MIN_BETWEEN})"),
            (check.within_stroke, f"opens {check.open_width * 1000:.0f} mm, inside the {STROKE * 1000:.0f} mm stroke"),
        ]  # fmt: skip
        lines += [f"| {'✅' if ok else '❌'} | {text} |" for ok, text in rows] + [""]
    if order and not same(order[0][1], pick):
        best = order[0][1]
        c = checks.get(id(best))
        why = "" if c is None else ("passes every check" if c.feasible else "❌ " + ", ".join(c.reasons()) +
                                    " (drawn red in the scene)")  # fmt: skip
        lines += [f"**Best-scored, #1 · {score_text(best)}:** {why}.", ""]
    if checks:
        fails = [c for c in checks.values() if not c.feasible]
        kinds = {
            "hit the cloud at the grasp": sum(c.fingers > 0 or c.palm > 0 for c in fails),
            "hit it on the way in": sum(c.approach > 0 and not (c.fingers or c.palm) for c in fails),
            "close on air": sum(c.between < MIN_BETWEEN for c in fails),
            "open past the stroke": sum(not c.within_stroke for c in fails),
        }
        tally = "; ".join(f"{n} {kind}" for kind, n in kinds.items() if n)
        lines += [f"**{len(checks)} candidates:** {len(checks) - len(fails)} feasible, {len(fails)} fail "
                  f"({tally}).", ""]  # fmt: skip
    lines.append("*Colour in the scene = score (viridis ramp: dark purple 0 → yellow-green 1); hover a candidate "
                 "for its rank, score and verdict. Green / red = passes / fails a check.*")  # fmt: skip
    return "\n".join(lines)


# ── blueprint ────────────────────────────────────────────────────────────────


def blueprint() -> rrb.Blueprint:
    scene = rrb.Spatial3DView(
        origin="/world", name="Candidates on the cloud · colour = score · thick = chosen · red = what the best hits",
        eye_controls=rrb.EyeControls3D(kind="Orbital", position=[0.17, -0.27, 0.2], look_target=[0.0, -0.01, 0.06],
                                       eye_up=[0.0, 0.0, 1.0]),
    )  # fmt: skip
    scores = rrb.BarChartView(origin=f"/{KEY}/scores", name="Scores, best first (x = rank) · green passes every check",
                              plot_legend=rrb.PlotLegend(visible=False))  # fmt: skip
    card = rrb.TextDocumentView(origin=f"/{KEY}/verdict", name="Is the chosen grasp feasible?")
    return rrb.Blueprint(
        rrb.Horizontal(scene, rrb.Vertical(scores, card, row_shares=[1.0, 1.5]), column_shares=[1.45, 1.0]),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed"),  # one moment: nothing to scrub
        collapse_panels=False,
    )


# ── sample: a mug on a table and 25 candidates around it ─────────────────────


def _frame(origin, z, x) -> Pose:
    """A grasp pose: tips at ``origin``, approaching along ``z``, closing along ``x``."""
    z = np.asarray(z, dtype=np.float64) / np.linalg.norm(z)
    x = np.asarray(x, dtype=np.float64)
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    matrix[:3, 3] = origin
    return Pose.from_matrix(matrix)


def _radial(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle), 0.0])


@functools.cache
def _sample() -> tuple[PointCloud, tuple[GraspPose, ...], GraspPose]:
    points, rgb = mug_scene()
    cloud = PointCloud(points=points, channels={"rgb": rgb}, header=Header(frame_id="world")).downsampled(0.0025)
    rng = np.random.default_rng(11)
    header = Header(frame_id="world")
    middle = MUG_RADIUS - MUG_WALL / 2.0
    out = []

    def add(pose: Pose, width: float, pre: float, score: float) -> None:
        noisy = float(np.clip(score + rng.normal(0.0, 0.015), 0.05, 0.99))
        out.append(GraspPose(pose=pose, width=width, pre_grasp_width=pre, score=round(noisy, 3), object_id="mug",
                             header=header))  # fmt: skip

    # rim pinches straight down, one finger in, one out; deeper pinches score higher (more contact)
    rim = [(math.degrees(HANDLE_ANGLE), 0.025, 0.94), (-60, 0.018, 0.86), (-10, 0.018, 0.84), (-90, 0.020, 0.89),
           (-120, 0.016, 0.82), (-150, 0.014, 0.78), (180, 0.012, 0.74), (150, 0.012, 0.70), (60, 0.014, 0.72),
           (20, 0.016, 0.80)]  # fmt: skip
    for angle, depth, score in rim:
        a = math.radians(angle)
        add(_frame(middle * _radial(a) + [0, 0, MUG_HEIGHT - depth], [0, 0, -1], _radial(a)), 0.005, 0.030, score)
    # the handle: from outside, straight in, closing across the bar
    for dz, turn, score in ((0.0, 0.0, 0.84), (-0.006, 6.0, 0.78), (0.006, -6.0, 0.76), (0.0, 12.0, 0.66),
                            (0.010, 0.0, 0.62)):  # fmt: skip
        a = HANDLE_ANGLE + math.radians(turn)
        tips = (HANDLE_CENTRE[0] + HANDLE_ARC - 0.012) * _radial(HANDLE_ANGLE) + [0, 0, HANDLE_CENTRE[1] + dz]
        add(_frame(tips, -_radial(a), [-math.sin(a), math.cos(a), 0.0]), 0.012, 0.040, score)
    # across the body, from the camera's side: wide, so they score lower; one too low (the table), one too wide
    for angle, height, pre, score in ((-90, 0.050, 0.084, 0.58), (-110, 0.040, 0.084, 0.55),
                                      (-70, 0.060, 0.084, 0.52), (-130, 0.035, 0.084, 0.47),
                                      (-80, 0.006, 0.084, 0.50), (-100, 0.045, 0.092, 0.56)):  # fmt: skip
        a = math.radians(angle)
        add(_frame([0.0, 0.0, height] + 0.002 * _radial(a), -_radial(a), [-math.sin(a), math.cos(a), 0.0]),
            0.079, pre, score)  # fmt: skip
    # pinches that close on air: off the rim outside, and down inside the mug
    add(_frame(0.068 * _radial(math.radians(-140)) + [0, 0, 0.080], [0, 0, -1], _radial(math.radians(-140))),
        0.005, 0.030, 0.42)  # fmt: skip
    add(_frame([0.0, 0.004, 0.060], [0, 0, -1], [1.0, 0.0, 0.0]), 0.010, 0.040, 0.36)
    # two more rim pinches, shallower, from the planner's second pass
    for angle, score in ((-75, 0.81), (-45, 0.77)):
        a = math.radians(angle)
        add(_frame(middle * _radial(a) + [0, 0, MUG_HEIGHT - 0.010], [0, 0, -1], _radial(a)), 0.005, 0.030, score)
    grasps = tuple(out)
    pts = np.asarray(cloud.points, dtype=np.float64)
    feasible = [g for _, g in ranked(grasps) if grasp_checks(g, pts).feasible]
    return cloud, grasps, feasible[0]


def sample() -> list[tuple[float | None, dict]]:
    cloud, grasps, chosen = _sample()
    return [(None, {"cloud": cloud, "grasps": grasps, "chosen": chosen})]


register(Dashboard(
    key=KEY,
    title="Grasping",
    category="manipulation",
    summary="Which grasps did the planner propose on this object, which is best, and is it feasible?",
    consumes=(PointCloud, GraspPose),
    inputs={
        "cloud": "the PointCloud the planner saw: the object on its table",
        "grasps": "the planner's candidates, a tuple of GraspPose, each with a score",
        "chosen": "the GraspPose it picked (optional; else the best-scored is shown as the pick)",
    },
    blueprint=blueprint,
    log=log,
    sample=sample,
    code=(
        "import robot_dashboards as dash\n\n"
        "dash.show(\"grasping\", cloud=cloud, grasps=candidates, chosen=pick)"
    ),
))  # fmt: skip
