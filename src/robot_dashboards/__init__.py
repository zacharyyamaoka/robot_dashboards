"""robot_dashboards: stock Rerun dashboards for common robotics use cases, on robot_msgs types.

    import robot_dashboards as dash

    dash.show("grasping", cloud=cloud, grasps=candidates)   # one moment, desktop viewer
    dash.save("grasping.rrd", "grasping")                   # its sample data, blueprint inside
    with dash.stream("object_detection") as live:           # live
        live.log(t=t, image=frame, detections=dets)

Each dashboard is one module in robot_dashboards/dashboards/ (docs/CONTRACT.md); the types inside are drawn by
robot_msg_viz's stock views, so a dashboard and viz.show(msg) agree.
"""

from robot_dashboards.core import (
    DASHBOARDS,
    Dashboard,
    DashboardStream,
    all_dashboards,
    get,
    register,
    save,
    show,
    stream,
)

__version__ = "0.1.0"

__all__ = [
    "DASHBOARDS",
    "Dashboard",
    "DashboardStream",
    "all_dashboards",
    "get",
    "register",
    "save",
    "show",
    "stream",
]
