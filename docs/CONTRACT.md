# robot_dashboards contract (v1, frozen 2026-11-24)

A dashboard is a fixed layout for one robotics use case, fed robot_msgs messages, drawn by a viewer (Rerun today). `tests/test_contract.py` and
robot_msg_viz's `ruler.py` are the ruler: a lane adding a dashboard never edits them.

## The three layers (Zach, 2026-11-24)

```
your data ──adapter──▶ robot_msgs ──dashboard──▶ viewer
```

- **Adapters** (`adapters/`): anything in, robot_msgs out. Built in: `robot_msgs` (identity) and `ros2` (ROS 2
  messages and MCAP / rosbag2 files via robot_msgs.ros). Users convert to robot_msgs or `adapters.register(Adapter(...))`.
- **Dashboards** (`dashboards/<key>.py`): the question, typed robot_msgs inputs, sample data, one renderer per viewer.
- **Viewers** (`viewers/`): Rerun is supported; Foxglove, Viser and static HTML are on the roadmap (docs/ROADMAP.md).

## API (stable)

```python
import robot_dashboards as dash
dash.show(key, ticks=None, viewer="rerun", web=False, adapter=None, **inputs)   # one moment (inputs), ticks, or the sample
dash.save(path, key, ticks=None, viewer="rerun", adapter=None) -> Path          # layout inside
dash.stream(key, viewer="rerun", adapter=None, spawn=True, web=False, save=None) # live: live.log(t=..., **inputs)
dash.get(key) / dash.all_dashboards() / dash.adapters / dash.viewers
```

Inputs that are not robot_msgs messages go through `adapters.to_robot_msgs` (the named adapter, else the first that
accepts the value); a value no adapter accepts raises and names the two ways out.

## One module per dashboard: `src/robot_dashboards/dashboards/<key>.py`

```python
register(Dashboard(
    key="object_detection", title="Object detection", category="perception",   # perception | manipulation | motion | robot state
    summary="One line: what the dashboard answers.",
    consumes=(Image, Detection2DArray),          # robot_msgs types only
    inputs={"image": "the camera frame", "detections": "what the detector found in it"},
    sample=sample,                               # [(t or None, {input: message}), ...] deterministic, looks real
    viewers={"rerun": RerunView(
        blueprint=lambda: rrb.Blueprint(...),    # the fixed layout: ≥2 views, each pointing at data the log writes
        log=log,                                 # log(rec, **inputs): draw one tick; inputs may be partial
    )},
))
```

- **Draw types through robot_msg_viz** (`robot_msg_viz.log(msg, name, recording=rec)`), so a type looks the same in
  a dashboard and in `viz.show(msg)`; add only what the use case needs on top (counts, scores, a highlighted pick).
- Entity layout: robot_msg_viz's (`/world/…`, `/image/…`, `/camera/…`, `/<field>/<column>/<name>`, `/notes/…`), plus
  `/<key>/…` for dashboard-only derived signals (counts, scores, phases).
- **Answer a question.** The summary says what the dashboard answers (e.g. "what did the detector see, and how sure
  was it over time?"); every view earns its place against that question.
- Angles on screen in degrees; class colours through `rr.AnnotationContext`; scores as a colour ramp.
- Sample data: built in code (seeded numpy) or from `robot_dashboards/assets/` (the UR5e model and a real UR5e
  trajectory); no network; the gallery .rrd under 8 MB, ideally under 3 MB.
- Look at it: robot_msg_viz's `tools/shoot.py` shoots the real web viewer (run it from this repo's environment so the
  entry point is installed): `env -u VIRTUAL_ENV uv run --directory /home/bam/robot_dashboards python
  /home/bam/robot_msg_viz/tools/shoot.py <out> --only <key>`, then open the PNG.
