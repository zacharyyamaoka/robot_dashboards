# robot_dashboards contract (v1, frozen 2026-11-24)

A dashboard is a fixed Rerun layout for one robotics use case, fed robot_msgs messages. `tests/test_contract.py` and
robot_msg_viz's `ruler.py` are the ruler: a lane adding a dashboard never edits them.

## API (stable)

```python
import robot_dashboards as dash
dash.show(key, ticks=None, web=False, **inputs)    # one static moment (inputs), ticks, or the sample
dash.save(path, key, ticks=None) -> Path            # blueprint inside
dash.stream(key, spawn=True, web=False, save=None)  # live: live.log(t=..., **inputs)
dash.get(key) / dash.all_dashboards()
```

## One module per dashboard: `src/robot_dashboards/dashboards/<key>.py`

```python
register(Dashboard(
    key="object_detection", title="Object detection", category="perception",   # perception | manipulation | motion | robot state
    summary="One line: what the dashboard answers.",
    consumes=(Image, Detection2DArray),          # robot_msgs types only
    inputs={"image": "the camera frame", "detections": "what the detector found in it"},
    blueprint=lambda: rrb.Blueprint(...),        # the fixed layout: ≥2 views, each pointing at data the log writes
    log=log,                                     # log(rec, **inputs): draw one tick; inputs may be partial
    sample=sample,                               # [(t or None, {input: message}), ...] deterministic, looks real
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
