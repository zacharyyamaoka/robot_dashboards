"""The babble deck for the 2026-11-28 robot-viewer round: writes gallery-spec.json, then builds and checks the deck.

    python3 /home/bam/robot_dashboards/docs/proposals/robot-viewer-2026-11-28/make_gallery.py

Media are referenced relative to reports/ (the deck lives at reports/robot-viewer-proposals-2026-11-28.html and its
stills and videos in the gitignored reports/media/robot-viewer-2026-11-28/), so the page stays a small text file.
Scores were written after every variant was built and shot, against criteria frozen before building (the brief's).
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GALLERY = Path("/home/bam/.claude-bam/skills/babble/scripts/gallery.py")
OUT = Path("/home/bam/robot_dashboards/reports/robot-viewer-proposals-2026-11-28.html")
M = "media/robot-viewer-2026-11-28"
SPEC = HERE / "gallery-spec.json"
SOURCE = (HERE / "build.py").read_text()

CSS = """<style>
.rv{font:13px/1.4 system-ui,sans-serif;color:#e8e8ea;background:#111;border-radius:8px;overflow:hidden}
.rv img{display:block;width:100%;height:auto}
.rv .st{display:none}.rv .chips{display:flex;gap:6px;padding:8px;flex-wrap:wrap;background:#1b1b1f}
.rv button{font:inherit;padding:4px 10px;border-radius:14px;border:1px solid #555;background:#26262b;color:#eee;cursor:pointer}
.prototype:not([data-story-state]) .st.s1,.prototype[data-story-state=s1] .st.s1,.prototype[data-story-state=s2] .st.s2,
.prototype[data-story-state=s3] .st.s3{display:block}
.prototype:not([data-story-state]) button[data-story-to=s1],.prototype[data-story-state=s1] button[data-story-to=s1],
.prototype[data-story-state=s2] button[data-story-to=s2],.prototype[data-story-state=s3] button[data-story-to=s3]
{background:#3b5bdb;border-color:#748ffc}
.row3{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;background:#111}
.row2{display:grid;grid-template-columns:repeat(2,1fr);gap:4px;background:#111}
.row3 figure,.row2 figure{margin:0}.row3 figcaption,.row2 figcaption{font:12px system-ui;color:#bbb;padding:4px}
pre.code{margin:0;padding:12px;background:#0d0d10;color:#dfe3ea;font:11.5px/1.45 ui-monospace,monospace;
white-space:pre-wrap;max-height:620px;overflow:auto}
</style>"""


def shot(name: str) -> str:
    return f"{M}/shots/{name}.png"


def preview(steps: list[tuple[str, str]]) -> str:
    """Three states over real stills; chips switch them (and the deck's stepper drives the same states)."""
    imgs = "".join(f'<img class="st s{i}" src="{shot(n)}" alt="{html.escape(label)}" loading="eager">'
                   for i, (n, label) in enumerate(steps, start=1))
    chips = "".join(f'<button data-story-to="s{i}">{html.escape(label)}</button>'
                    for i, (_, label) in enumerate(steps, start=1))
    return f'{CSS}<div class="rv">{imgs}<div class="chips">{chips}</div></div>'


def video(variant: str, caption: str) -> dict:
    return {"label": "Dual-arm cell played 0 → 5 s (real Rerun web viewer)", "caption": caption,
            "html": f'<video src="{M}/video/{variant}-dual_arm.mp4" poster="{M}/video/{variant}-dual_arm.webp" '
                    'autoplay muted loop playsinline controls style="width:100%"></video>'}


def three_robots(variant: str) -> dict:
    figs = "".join(f'<figure><img src="{shot(f"{variant}-{r}")}" alt="{r}"><figcaption>{label}</figcaption></figure>'
                   for r, label in (("pendulum", "1-DOF pendulum + side camera"), ("ur5e", "UR5e + wrist camera"),
                                    ("dual_arm", "dual-arm cell")))
    return {"label": "Same layout function, three robots",
            "caption": "Nothing per robot: the one blueprint function walks each description tree. Look at how the "
                       "page grows from one servo to twelve.",
            "html": f'{CSS}<div class="row3">{figs}</div>'}


def stills(label: str, caption: str, names: list[tuple[str, str]]) -> dict:
    cls = "row2" if len(names) <= 2 else "row3"
    figs = "".join(f'<figure><img src="{shot(n)}" alt="{html.escape(c)}"><figcaption>{html.escape(c)}</figcaption>'
                   f"</figure>" for n, c in names)
    return {"label": label, "caption": caption, "html": f'{CSS}<div class="{cls}">{figs}</div>'}


def code(label: str, caption: str, *functions: str) -> dict:
    parts = []
    for name in functions:
        match = re.search(rf"^(def {name}\(.*?)(?=^\S)", SOURCE, re.S | re.M)
        if not match:
            match = re.search(rf"^({name} *[:=].*?)(?=^\S)", SOURCE, re.S | re.M)
        assert match, name
        parts.append(match.group(1).rstrip())
    return {"label": label, "caption": caption,
            "html": f'{CSS}<pre class="code">{html.escape(chr(10).join(parts))}</pre>'}


def story(title: str, steps: list[tuple[str, str]]) -> dict:
    return {"title": title, "steps": [{"label": label, "caption": caption, "state": f"s{i}",
                                       "target": f'button[data-story-to="s{i}"]'}
                                      for i, (label, caption) in enumerate(steps, start=1)]}


def score(s: int, evidence: str, confidence: str) -> dict:
    return {"score": s, "evidence": evidence, "confidence": confidence}


GATES_OK = {
    "g1": {"pass": True, "evidence": "The builder only logs to an .rrd; there is no command path and no transport "
                                     "is imported or opened."},
    "g2": {"pass": True, "evidence": "All 15 recordings (and the tab variants) loaded in the stock Rerun 0.40 web "
                                     "viewer from the saved .rrd with their blueprint, 0 viewer errors, no panic."},
    "g3": {"pass": True, "evidence": "Every blueprint is _frame(): Horizontal(Spatial3DView, …) with the 3D first; "
                                     "seen in every still."},
}

VARIANTS = [
    {
        "id": "v1", "name": "Tabs by part type", "accent": "#4c8dff",
        "thesis": "Zach's sketch: 3D left, tabs Servos / Cameras / Config; Servos splits into one sub-tab per "
                  "subtree, one plot per joint, command dots over the state line.",
        "bestWhen": "You want one joint's tracking at a time, big, and to know where every kind of thing lives.",
        "losesWhen": "You need cameras and servos together, or a new kind of part (IMU, gripper) arrives: its tab "
                     "is an edit.",
        "decisions": [
            {"label": "Primary grouping", "value": "by part kind: Servos, Cameras, Config"},
            {"label": "Big lists", "value": "a sub-tab per subtree (left_arm | right_arm), 6 joints + a temperature "
                                            "plot each"},
            {"label": "Config", "value": "one table per top-level group in its own tab"},
        ],
        "keepParts": ["per-joint cmd-vs-state plots", "sub-tab per subtree", "Config tab tables"],
        "steps": [("v1-dual_arm", "Servos › left_arm"), ("v1-dual_arm-servos_right_arm", "Servos › right_arm"),
                  ("v1-dual_arm-config", "Config")],
        "story": [("Servos, left arm", "One plot per joint: command dots, state line; a temperature plot closes "
                                       "the grid."),
                  ("Servos, right arm", "The faulted elbow: the state line goes flat at 2.6 s while the command "
                                        "dots carry on."),
                  ("Config", "Every servo's kp, kd, current cap and limits, and each camera, per group.")],
        "media": lambda: [video("v1", "Plays the cell, then clicks Servos › right_arm (3.5 s) and Cameras (6.5 s)."),
                          three_robots("v1"),
                          stills("Cameras tab", "Each camera full size, but never beside the joints it watches.",
                                 [("v1-dual_arm-cameras", "dual-arm cell · Cameras"),
                                  ("v1-pendulum-cameras", "pendulum · Cameras")]),
                          code("The layout function", "Groups come from servo_groups(tree); the three tabs are fixed.",
                               "v1_tabs_by_type")],
        "scores": {
            "fr1": score(4, "One joint per plot, ~330 px tall; the right elbow's flat state vs moving command is "
                            "obvious (sub-tab still). Only one arm visible at a time.", "high"),
            "fr2": score(4, "Plots, sub-tabs, cameras and config tables are all derived from the tree; but the tab "
                            "set is a fixed type list, so a new part kind needs a new tab.", "high"),
            "fr3": score(4, "Pendulum: 1 plot; dual-arm: 2 sub-tabs × 7 plots. No long list, but a 4-arm cell "
                            "means 4 sub-tabs.", "high"),
            "fr4": score(3, "Cameras get a whole tab, large and legible, but are never seen with the servos; only "
                            "frustums show in 3D.", "high"),
            "fr5": score(5, "Config tab: a full table per group (bus/id, kp, kd, max A, limits) plus cameras, "
                            "fully readable.", "high"),
        },
    },
    {
        "id": "v2", "name": "Tabs mirror the tree", "accent": "#2fb380",
        "thesis": "One tab per top-level part; each tab is that part's stock view, built recursively from its "
                  "children's stock views by a kind → view registry.",
        "bestWhen": "Descriptions get composed and re-composed (pendulum → arm → cell → line) and you look at one "
                    "subsystem at a time.",
        "losesWhen": "You want the whole cell at once, or the same signal across arms side by side.",
        "decisions": [
            {"label": "Primary grouping", "value": "by the description tree's top level"},
            {"label": "Composition", "value": "STOCK_VIEWS[kind](path, part) recurses; a new kind is one entry"},
            {"label": "Cameras", "value": "inside the part they belong to (wrist camera in its arm's tab)"},
        ],
        "keepParts": ["kind → stock view registry", "wrist camera inside its arm's tab", "tab per top-level part"],
        "steps": [("v2-dual_arm", "left_arm"), ("v2-dual_arm-right_arm", "right_arm"),
                  ("v2-dual_arm-overhead_camera", "overhead_camera")],
        "story": [("left_arm tab", "The arm's six joints and its wrist camera, each with its own config card."),
                  ("right_arm tab", "Same stock view, the other subtree: the faulted elbow sits with its config."),
                  ("overhead_camera tab", "A cell-level part gets its own tab with no layout edit.")],
        "media": lambda: [video("v2", "Plays the cell, then clicks the right_arm (3.5 s) and overhead_camera (6.5 s) "
                                      "tabs."),
                          three_robots("v2"),
                          code("The whole layout: a registry and a recursion",
                               "Adding a part to the description adds its view; adding a kind adds one entry.",
                               "_stock_servo", "_stock_camera", "_stock_group", "STOCK_VIEWS", "stock_view",
                               "v2_tabs_mirror_tree")],
        "scores": {
            "fr1": score(3, "Per-joint plots but ~150 px tall, sharing the tab with config cards; cmd/state "
                            "separation readable, smaller than V1.", "medium"),
            "fr2": score(5, "The layout is STOCK_VIEWS recursion over the tree: the pendulum, the arm and the cell "
                            "all came out of it with no per-robot code; a new kind is one registry line.", "high"),
            "fr3": score(4, "Dual-arm: 3 tabs, 7 cards each. Deep trees nest grids, which will get small past two "
                            "levels.", "medium"),
            "fr4": score(4, "Each wrist camera sits beside its own arm's joints; the overhead camera is its own tab "
                            "(not seen with the arms).", "medium"),
            "fr5": score(3, "A config card per part, right next to its plot, but cut to two lines at 1440×900; the "
                            "full table needs a maximise.", "high"),
        },
    },
    {
        "id": "v3", "name": "tview overview", "accent": "#e0a030",
        "thesis": "No tabs: one read-only page like moteus tview, a live table of every servo's cmd, state, error, "
                  "temperature and fault at the playhead, camera thumbnails, two summary plots.",
        "bestWhen": "You are commissioning or watching live and want the numbers now, every part on one screen.",
        "losesWhen": "You need history per joint, or the cell grows past ~15 servos (the table scrolls).",
        "decisions": [
            {"label": "Primary object", "value": "the playhead: a markdown table regenerated each tick"},
            {"label": "Problems", "value": "bold error > 2°, ⚠ temperature ≥ 60 °C, ⛔ fault code with its name"},
            {"label": "History", "value": "only two overlays: tracking error and temperature, every servo"},
        ],
        "keepParts": ["live now-table with flagged cells", "always-on camera thumbnails", "error overlay plot"],
        "steps": [("v3-dual_arm", "dual-arm cell"), ("v3-ur5e", "UR5e"), ("v3-pendulum", "pendulum")],
        "story": [("Dual-arm cell", "Every servo of both arms in one table; the elbow's ⛔ 33 and −23.8° error "
                                    "stand out."),
                  ("UR5e", "The same page for one arm: one group in the table, one wrist camera."),
                  ("Pendulum", "One row, one camera: the page shrinks with the description.")],
        "media": lambda: [video("v3", "Plays the cell: the table's numbers move with the playhead; the elbow flags "
                                      "at 2.6 s."),
                          three_robots("v3"),
                          code("The layout function", "One page; the table text is built from the tree each tick.",
                               "v3_tview")],
        "scores": {
            "fr1": score(4, "cmd, state and error side by side as numbers with problems flagged: the fastest read "
                            "at the playhead; history is only the error overlay.", "medium"),
            "fr2": score(4, "Table rows, groups and thumbnails are generated from the tree; the page structure "
                            "itself is fixed (fine, it has no per-kind sections).", "medium"),
            "fr3": score(3, "Dual-arm (12 servos) already pushes right_arm's last row below the fold; a bigger cell "
                            "is a scrolling list, the thing Zach wanted to avoid.", "high"),
            "fr4": score(4, "All camera feeds always visible beside the numbers, as thumbnails (~220 px).", "high"),
            "fr5": score(3, "Config is a table under the live one, in the same scroll; readable but out of sight "
                            "on the dual-arm cell.", "medium"),
        },
    },
    {
        "id": "v4", "name": "Tabs by signal", "accent": "#c25cf0",
        "thesis": "Tabs Position / Velocity / Effort / Health / Cameras / Config; in each, one plot per subtree "
                  "overlays all its joints, command dots against state lines.",
        "bestWhen": "You compare the same quantity across arms (is left tracking like right?) or hunt one "
                    "signal's anomaly.",
        "losesWhen": "Six overlaid joints hide one joint's lag; you must pick the colour out of a key.",
        "decisions": [
            {"label": "Primary grouping", "value": "by signal, then by subtree"},
            {"label": "Legends", "value": "off; a colour key in the title (a 12-entry legend hid half the curves)"},
            {"label": "Health", "value": "tracking error, temperature and fault code per subtree"},
        ],
        "keepParts": ["Health tab (error · temp · fault)", "one plot per subtree per signal", "Config tab"],
        "steps": [("v4-dual_arm", "Position"), ("v4-dual_arm-health", "Health"), ("v4-dual_arm-config", "Config")],
        "story": [("Position", "Both arms at once, each a plot of six joints, command dots over state lines."),
                  ("Health", "Tracking error, temperature and fault per arm: the elbow's fault code steps to 33."),
                  ("Config", "The same per-group config tables as V1.")],
        "media": lambda: [video("v4", "Plays the cell, then clicks Health (3.5 s) and Cameras (6.5 s)."),
                          three_robots("v4"),
                          stills("Velocity and effort tabs", "The other signals keep the same per-subtree overlays.",
                                 [("v4-dual_arm-velocity", "Velocity"), ("v4-dual_arm-effort", "Effort")]),
                          code("The layout function", "Subtrees from the tree; the signal tabs are fixed.",
                               "v4_tabs_by_signal")],
        "scores": {
            "fr1": score(3, "Both arms visible together, but six joints per plot: the elbow's divergence reads, a "
                            "single joint's lag needs the colour key or hover.", "high"),
            "fr2": score(4, "Per-subtree plots come from the tree; the six signal tabs are fixed, so a camera-only "
                            "or IMU subtree has no natural place.", "medium"),
            "fr3": score(4, "One plot per subtree per signal whatever the joint count: dual-arm is two plots per "
                            "tab; more arms just add rows.", "medium"),
            "fr4": score(3, "Cameras live in their own tab, apart from the signals.", "high"),
            "fr5": score(5, "Config tab: the same full per-group tables as V1.", "high"),
        },
    },
    {
        "id": "v5", "name": "Part cards", "accent": "#f0605a",
        "thesis": "Every part is a card (plot + config, or image + config) nested to mirror the tree, rows and "
                  "columns alternating by depth, everything visible at once.",
        "bestWhen": "Small robots (≤ 7 parts) where seeing every part at once beats size.",
        "losesWhen": "Anything bigger than one arm: twelve cards leave each plot ~60 px and the config cut off.",
        "decisions": [
            {"label": "Visibility", "value": "everything at once, no tabs"},
            {"label": "Nesting", "value": "Horizontal at even depth, Vertical at odd: the layout is the tree"},
            {"label": "Card", "value": "plot and config side by side, camera and config side by side"},
        ],
        "keepParts": ["card = signal + config together", "layout drawn as the tree", "no hidden state"],
        "steps": [("v5-dual_arm", "dual-arm cell"), ("v5-ur5e", "UR5e"), ("v5-pendulum", "pendulum")],
        "story": [("Dual-arm cell", "Both arms' twelve cards and three cameras on one screen, each very small."),
                  ("UR5e", "One arm: the cards get room and become readable."),
                  ("Pendulum", "Two cards: the servo and its camera.")],
        "media": lambda: [video("v5", "Plays the cell: everything moves at once, nothing to click."),
                          three_robots("v5"),
                          code("The layout function", "The card recursion: no tabs, the tree's shape is the page's.",
                               "_card", "v5_cards")],
        "scores": {
            "fr1": score(2, "On the dual-arm cell each plot is ~60 px tall: command and state overlap into one line "
                            "except the faulted elbow.", "high"),
            "fr2": score(5, "Pure recursion over the tree; a new part is a new card with no edit.", "high"),
            "fr3": score(2, "The opposite of 'no huge list': every part is on screen, so the dual-arm cell already "
                            "crushes the cards.", "high"),
            "fr4": score(3, "All cameras always visible, but the wrist cameras are thumbnails in a crowded "
                            "column.", "high"),
            "fr5": score(2, "Config is beside every plot but truncated to its first line at this size.", "high"),
        },
    },
]

REQUIREMENTS = [
    {"id": "fr1", "name": "Command vs state for every actuator readable at a glance", "weight": 25,
     "why": "The first question at the rig is 'is it doing what I told it?', for every joint.",
     "passCondition": "For any actuator, command and measured state (and their gap) read without hovering.",
     "anchors": {"1": "cmd and state are not shown together", "3": "shown together but small or overlaid",
                 "5": "each actuator's cmd vs state is large and immediately distinct"}},
    {"id": "fr2", "name": "Composes with the description tree", "weight": 25,
     "why": "Zach: 'as you compose the robotdescription so should their visualization compose'.",
     "passCondition": "Adding a part to the description adds its view with no dashboard edit.",
     "anchors": {"1": "a hand-written layout per robot", "3": "generated, but new kinds or groups need edits",
                 "5": "the layout is a function of the tree; nothing per robot"}},
    {"id": "fr3", "name": "Scales from 1-DOF to a dual-arm cell without a huge list", "weight": 20,
     "why": "Zach: 'break down by right arm etc so you don't have a crazy huge list'.",
     "passCondition": "The dual-arm cell stays as readable as the pendulum.",
     "anchors": {"1": "one long list or crushed views at dual-arm", "3": "works at dual-arm, strains beyond",
                 "5": "grows by subtree with no loss of legibility"}},
    {"id": "fr4", "name": "Sensor measurements (cameras) visible", "weight": 15,
     "why": "'all the sensors we want to see the measurements'.",
     "passCondition": "Every camera's image is visible and legible.",
     "anchors": {"1": "missing or illegible", "3": "legible but apart from the actuators",
                 "5": "legible and in context with what it watches"}},
    {"id": "fr5", "name": "Configuration visible, read-only", "weight": 15,
     "why": "'you can see the configurations etc on the actuators and sensors'.",
     "passCondition": "Every servo's kp/kd/current/limits and each camera's resolution/exposure/fps are readable.",
     "anchors": {"1": "not shown", "3": "shown but truncated or out of sight", "5": "complete and readable"}},
]


def spec() -> dict:
    variants = []
    for v in VARIANTS:
        variants.append({
            "id": v["id"], "name": v["name"], "thesis": v["thesis"], "accent": v["accent"],
            "bestWhen": v["bestWhen"], "losesWhen": v["losesWhen"], "decisions": v["decisions"],
            "keepParts": v["keepParts"],
            "proof": ["Built by build.py into a real .rrd and shot in the stock Rerun 0.40 web viewer, headless "
                      "(0 viewer errors).", "Same recording as the other four; only the blueprint differs."],
            "previewLabel": "real Rerun stills", "preview": preview(v["steps"]),
            "story": story(f"{v['name']}: three states", v["story"]),
            "media": v["media"](), "scores": v["scores"], "gateResults": GATES_OK,
        })
    return {
        "schemaVersion": 3,
        "title": "Five ways to view a robot in Rerun",
        "kicker": "Babble round · 2026-11-28 · robot_dashboards",
        "brief": "One read-only Rerun viewer per robot description: 3D on the left; every actuator's command vs "
                 "state, every sensor's measurement and every part's configuration; composed from the description "
                 "tree so the pendulum, the UR5e and the dual-arm cell come out of the same function.",
        "count": 5,
        "boundary": "Real Rerun 0.40 recordings and blueprints, opened in the stock web viewer. Simulated: commands "
                    "are a real UR5e trajectory, states are it delayed 90 ms with sag and noise (the pendulum runs "
                    "a PD loop); camera frames are a pinhole render of the same geometry; temperatures, the 2.6 s "
                    "elbow fault and all configs are sample numbers. Not registered as dashboards.",
        "invariants": ["Same recording for all five (same entities, samples, colours); only the blueprint differs",
                       "Same three robots: 1-DOF pendulum + side camera, UR5e + wrist camera, dual-arm cell "
                       "(two UR5e subtrees, each with its wrist camera) + overhead camera",
                       "Stills at 1440×900, dark theme, playhead at t = 5.0 s",
                       "Read-only: no command path, no bus"],
        "axes": [
            {"name": "Primary grouping", "values": ["part kind (V1)", "description tree, tabbed (V2)",
                                                    "none: one page (V3)", "signal (V4)",
                                                    "description tree, all at once (V5)"]},
            {"name": "What's visible at once", "values": ["one kind, one subtree (V1)", "one top-level part (V2)",
                                                          "everything, as numbers (V3)",
                                                          "one signal, every subtree (V4)",
                                                          "everything, as views (V5)"]},
            {"name": "Where config lives", "values": ["its own tab (V1, V4)", "beside each part (V2, V5)",
                                                      "under the live table (V3)"]},
        ],
        "requirements": REQUIREMENTS,
        "variants": variants,
        "hardGates": [
            {"id": "g1", "name": "Read-only", "why": "A viewer that can command is not a viewer; it never opens a bus."},
            {"id": "g2", "name": "Stock Rerun from a saved .rrd with its blueprint",
             "why": "No custom viewer or server: the layout travels inside the file."},
            {"id": "g3", "name": "3D view on the left", "why": "Zach: 'definitely we want a 3d view on the left'."},
        ],
        "defaultId": "v1",
        "defaultWhy": "Tabs by part type (80) and Tabs mirror the tree (77) are co-leaders, 3 points apart, so the "
                      "pick is fragile. V1 wins on big per-joint plots and a complete Config tab; V2 wins on "
                      "composition (the layout is a registry + recursion) and on keeping each wrist camera with its "
                      "arm.",
        "decisionHinge": "How much FR2 (composition) outweighs FR5 (config readability). Move 10 points from FR5 to "
                         "FR2 and V2 leads 81 to 78. A splice probably beats both: V2's registry and tree tabs, with "
                         "V1's bigger per-joint plots and a Config tab (or V3's now-table) as the stock view of a "
                         "group.",
        "checks": ["Exactly five structural directions", "Same recording and viewport for every variant",
                   "Criteria and gates fixed before building", "Scores carry evidence and confidence"],
    }


def main() -> int:
    SPEC.write_text(json.dumps(spec(), indent=1, ensure_ascii=False) + "\n")
    # WHY a fresh temp file then a move: gallery.py build refuses to overwrite its output
    with tempfile.TemporaryDirectory() as scratch:
        fresh = Path(scratch) / OUT.name
        build = subprocess.run([sys.executable, str(GALLERY), "build", "--spec", str(SPEC), "--output", str(fresh)],
                               capture_output=True, text=True)  # fmt: skip
        print(build.stdout, build.stderr)
        if build.returncode:
            return build.returncode
        shutil.move(str(fresh), OUT)
    check = subprocess.run([sys.executable, str(GALLERY), "check", "--input", str(OUT), "--strict"],
                           capture_output=True, text=True)  # fmt: skip
    print(check.stdout, check.stderr)
    return check.returncode


if __name__ == "__main__":
    sys.exit(main())
