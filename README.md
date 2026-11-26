# robot_dashboards

**[▶ Live landing page](https://zacharyyamaoka.github.io/robot_dashboards/)** · deployed from `main` by `.github/workflows/site.yml`

**Stock dashboards for common robotics use cases, fed [robot_msgs](https://github.com/zacharyyamaoka/robot_msgs)
messages.** Rerun today; other viewers are on the [roadmap](docs/ROADMAP.md).

```
your data ──adapter──▶ robot_msgs ──dashboard──▶ viewer
             robot_msgs ✓              7 use cases     Rerun ✓
             ROS 2 / MCAP ✓                            Foxglove, Viser, static HTML (roadmap)
             yours: register()
```

```python
import robot_dashboards as dash

dash.show("grasping", cloud=cloud, grasps=candidates)        # one moment, desktop viewer
dash.save("grasping.rrd", "grasping")                        # its sample data, layout inside
with dash.stream("object_detection") as live:                # live
    for t, frame, dets in camera:
        live.log(t=t, image=frame, detections=dets)

from robot_dashboards.adapters import ros2                   # already have ROS 2 bags?
run = ros2.bag_ticks("run_0412/", joint_state="/joint_states", wrench="/ft_sensor/wrench")
dash.show("robot_state", ticks=run)
```

| Use case | The question it answers | Draws |
|---|---|---|
| `object_detection` | What did the detector see in each frame, and how sure was it over time? | camera with class-coloured boxes and track trails, score per track, count per class, this frame's table |
| `detection_3d` | Where are the objects in 3D, and what are they? | cloud + 3D boxes + camera frustum, the boxes projected onto the picture, a detections table |
| `rgbd_camera` | What does the RGB-D camera see, in 2D and in 3D? | back-projected coloured cloud in the frustum, RGB, depth in metres, depth statistics |
| `grasping` | Which grasps did the planner propose, which is best, and is it feasible? | candidates coloured by score, the chosen grasp, collision points, sorted scores, a pass/fail verdict |
| `pick_and_place` | Where is the arm in the cycle, and did each phase go as planned? | the UR5e on robot_msgs' stock URDF viewer, numbered waypoints, phase / gripper / on-plan lanes, phase-vs-plan table |
| `trajectory_generation` | Did the generator turn these waypoints into a smooth trajectory inside the limits? | the stock URDF trajectory viewer, numbered waypoints, a verdict per limit, peak % of each limit |
| `robot_state` | What is the robot doing right now, and is anything unhealthy? | URDF arm driven by JointState, joint angles in degrees, efforts, wrist F/T, IMU, drive temperatures, a WARN/ERROR log |

Each dashboard draws its types through [robot_msg_viz](https://github.com/zacharyyamaoka/robot_msg_viz)'s stock views,
so a type looks the same in a dashboard and in `viz.show(msg)`. The use cases appear on robot_msg_viz's landing page,
with a screenshot and Open in browser / Open on desktop buttons.

## Bring your own data

- **Convert to robot_msgs** and pass the messages; the `robot_msgs` adapter is the identity.
- **Register an adapter** for a format you have many of:
  `adapters.register(Adapter("my_format", "…", convert=to_robot_msgs, accepts=is_mine))`. After that, `show`, `save`
  and `stream` take your objects directly.
- **ROS 2:** messages and MCAP / rosbag2 recordings work out of the box (`pip install "robot-msgs[ros]"`). Decoding
  is robot_msgs' own `robot_msgs.ros`.

## Add a dashboard

One module in `src/robot_dashboards/dashboards/<key>.py`; see [docs/CONTRACT.md](docs/CONTRACT.md). It declares:
- the question it answers;
- its typed robot_msgs inputs;
- deterministic sample data;
- one renderer per viewer: `viewers={"rerun": RerunView(blueprint=..., log=...)}`.

`tests/test_contract.py` checks three things:
- every view in the layout has data in the saved file (read back by `rerun rrd print`);
- the sample uses only the declared types;
- a stream lands every tick on the timeline.

## The landing page

The seven dashboards with a still, a hero clip, the question each answers, its code, and Open in browser (Rerun's web
viewer, served from the page itself, so recordings open without a third-party host). Built with
robot_msg_viz's site builder in its plugin mode; the stills and the clip are committed in `website/media`.

```bash
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_dashboards python -m robot_msg_viz.site build /home/bam/robot_dashboards/website/dist --use-cases-only --title robot_dashboards --tagline "stock dashboards for common robotics use cases" --intro /home/bam/robot_dashboards/website/intro.html --github https://github.com/zacharyyamaoka/robot_dashboards --shots /home/bam/robot_dashboards/website/media/shots --hero /home/bam/robot_dashboards/website/media/hero.mp4 --viewer
python3 -m http.server 4494 --bind 127.0.0.1 --directory /home/bam/robot_dashboards/website/dist
```

**Where it is hosted:** GitHub Pages (the repo is public). CI tests, builds and deploys from `main`; the README's
live-page line is the switch the workflow reads.

## Develop

```bash
env -u VIRTUAL_ENV uv run --directory /home/bam/robot_dashboards --extra test python -m pytest -q
```

MIT licensed. The UR5e model in `assets/ur5e/` is from `ur_description` (BSD-3-Clause, license beside it).
