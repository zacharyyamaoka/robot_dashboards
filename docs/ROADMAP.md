# robot_dashboards roadmap

```
your data ──adapter──▶ robot_msgs ──dashboard──▶ viewer
             robot_msgs ✓            object detection ✓      Rerun ✓
             ROS 2 / MCAP ✓          detection 3D ✓          Foxglove (next)
             yours: register()       RGB-D camera ✓          Viser
                                     grasping ✓              static HTML
                                     pick and place ✓
                                     trajectory generation ✓
                                     robot state ✓
```

Zach, 2026-11-24: *"right now support rerun. and then you can have on the roadmap like foxglove, etc.. so we have
different viewers, and then we should have adapters. and right now we have like an adapter for robot_msgs. that allows
anyone to easily extend by adding their own adapter or by converting their messages to robot_msgs. and then the
consumers are simple you just convert to like rerun foxglove etc."*

## The three layers

- **Adapters** (`robot_dashboards/adapters/`): anything in, robot_msgs out. Built in: `robot_msgs` (the identity) and
  `ros2` (rosbags messages and whole MCAP / rosbag2 recordings, through robot_msgs.ros). Bring your own by converting
  to robot_msgs, or `adapters.register(Adapter(name, summary, convert=..., accepts=...))`.
- **Dashboards** (`robot_dashboards/dashboards/`): one module per use case: the question it answers, its typed
  robot_msgs inputs, deterministic sample data, and one renderer per viewer it supports.
- **Viewers** (`robot_dashboards/viewers/`): where a dashboard is drawn. `show` / `save` / `stream` take `viewer=`.

## Viewers

| Viewer | Status | What it takes |
|---|---|---|
| Rerun 0.40 | supported | blueprint + logging per dashboard (today's renderers) |
| Foxglove | next | an MCAP of the ticks (robot_msgs.ros.write_bag already writes one Foxglove opens) + a layout JSON per dashboard (foxglove-sdk `foxglove.layouts`, MIT); map detections onto `ImageAnnotations` / `SceneUpdate` (the 3D panel has no native vision_msgs) and drive the URDF from `JointStates`. Caveats: a Basic seat cannot open local files; comparison mode and embedding are paid. Lichtblick (the open-source fork) takes `?layoutUrl=` and `lichtblick://`, so it may be the better second target. |
| Viser | later | a scene builder per dashboard on viser's API; static `.viser` recordings embed in the landing page. Isaac Lab moved its visualizer from Rerun to Viser in 3.2, so a viewer-neutral dashboard is worth it. |
| Static HTML | later | robot_msgs' `.plot()` figures plus each dashboard's derived numbers, one self-contained page per run. |

**Viewer-neutral next step:** split each dashboard's `log` into `derive(inputs) -> numbers` (counts, scores, phases,
verdicts; no viewer) and a thin renderer per viewer, so a second viewer reuses the maths and only redraws.

## Adapters

| Adapter | Status |
|---|---|
| robot_msgs | supported (the identity) |
| ROS 2 messages + MCAP / rosbag2 files | supported (`adapters.ros2`, needs `robot-msgs[ros]`) |
| LeRobot datasets | later: observation / action keys to JointState, Image |
| Live ROS 2 topics (rclpy) | later: the `ros2` adapter already converts the messages; needs a subscriber loop into `stream()` |

## Use cases next

From the sweep (vault note *Stock Robot Viewers and Dashboards — Prior Art (2026-11-24)*): SLAM / odometry (path,
frustum, map, error plot, event log), lidar point clouds, camera calibration (rig + residuals), IMU (gyro deg/s,
roll/pitch/yaw), policy dataset playback (task banner over cameras, state, action), force-torque contact.
