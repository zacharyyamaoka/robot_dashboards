"""Build the 2026-11-25 report: comparison types for every stream (robot_msgs), their viewers (robot_msg_viz) and the
trajectory_comparison + sim2real dashboards (here), from the committed stills and the scratch captures.

    python3 /home/bam/robot_dashboards/docs/build_sim2real_report.py <scratch_png_dir>

Writes /home/bam/robot_dashboards/reports/media/sim2real-compare-types-2026-11-25.html (self-contained; images inlined).
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

HOME = Path("/home/bam")
DASH = HOME / "robot_dashboards/website/media/shots"
VIZ = HOME / "robot_msg_viz/website/media/shots"
OUT = HOME / "robot_dashboards/reports/media/sim2real-compare-types-2026-11-25.html"


def img(path: Path, caption: str) -> str:
    kind = "webp" if path.suffix == ".webp" else "png"
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img loading="lazy" src="data:image/{kind};base64,{data}" alt="{caption}">'
            f"<figcaption>{caption}</figcaption></figure>")


def main(scratch: Path) -> None:
    drill = scratch / "drill"
    extra = scratch / "extra/shots"
    sections = [
        ("1 · Sim-to-real dashboards (robot_dashboards @ 80feaa1)", [
            (DASH / "sim2real_pendulum.webp", "sim2real_pendulum: the answer first. sim_backlash 0.37 beats rigid sim 0.64 and the small-angle analytical model 6.33; envelope tol(t) = 0.5° + 1°/s·t"),
            (drill / "sim2real_pendulum-release_60deg-0.png", "Drill-down, release from 60°: |error| per model against the time-growing envelope"),
            (DASH / "sim2real_arm.webp", "sim2real_arm (UR5e): sim 0.31 vs plan 0.38 on the servo encoders; servo state, command, wrist IMU, F/T"),
            (DASH / "sim2real_five_bar.webp", "sim2real_five_bar (BAM's pilot cell shape): ranked on camera tool position in mm"),
            (DASH / "sim2real_diff_drive.webp", "sim2real_diff_drive: open-loop drift; calibrated model 0.19 vs nominal 0.49"),
        ]),
        ("2 · trajectory_comparison (robot_dashboards @ f023a81): traj_compare's Target-vs-run viewer, unchanged", [
            (DASH / "trajectory_comparison.webp", "Same code as traj_compare.rerun_target_run: arm, ×N deviation skeleton, Error tab first"),
        ]),
        ("3 · Comparison viewers per message type (robot_msg_viz @ 3461069)", [
            (extra / "overlay-servo_state.png", "viz.show(sim=…, real=…): two ServoStateTrajectories on one plot, real a lighter shade"),
            (extra / "cmp-camera.png", "Image comparison: error frame as a diverging heat map + mae/rmse/psnr per frame"),
            (extra / "cmp-servo_state.png", "ServoState comparison: per-field error tabs + fault-mismatch lane"),
            (extra / "cmp-imu.png", "IMU comparison: rotation (deg), ω (deg/s), acceleration (m/s²)"),
            (extra / "cmp-wrench.png", "Wrench comparison: force (N), torque (N·m)"),
            (extra / "cmp-scan.png", "PointCloud comparison: measured cloud coloured by distance (mm)"),
            (VIZ / "compare-DatasetComparison.webp", "DatasetComparison: worst of each stream first, then runs × pairs ranked, a tab per run to drill in"),
        ]),
    ]
    body = []
    for title, figures in sections:
        body.append(f"<h2>{title}</h2><div class=grid>")
        body += [img(p, c) for p, c in figures if p.exists()]
        body.append("</div>")
    html = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Sim2real comparison types</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mute:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mute:#999;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}}
figure{{margin:0}} img{{width:100%;border:1px solid var(--line);border-radius:6px}} figcaption{{color:var(--mute);font-size:13px}}
code{{font-size:13px}} td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}} table{{border-collapse:collapse}}
</style></head><body>
<h1>Every sensor and actuator stream is now comparable, and has a viewer and a sim2real dashboard</h1>
<p><b>Landed and pushed:</b> robot_msgs <code>7f08d8a</code> (0.4.0, not on PyPI), robot_msg_viz <code>3461069</code>, robot_dashboards <code>f023a81</code> + <code>80feaa1</code>. Tests: 1352 / 330 / 43 passed. robotmsgs.com shows the new types only after a PyPI release.</p>
<h2>Why robotmsgs.com had no TrajectoryDataset</h2>
<p>It was never in robot_msgs. It lived in traj_compare (<code>dataset.Run</code>) and bam_ws <code>traj_integration_tests</code>. The site is generated from the installed package. The generalised version is now in robot_msgs: <code>Recording</code> → <code>Run</code> → <code>Dataset</code>, compared by <code>compare_dataset</code> into a <code>DatasetComparison</code>.</p>
<h2>What robot_msgs gained</h2>
<table><tr><th>Stream type</th><th>compare() metrics</th></tr>
<tr><td>ServoStateTrajectory</td><td>positions|velocities|effort|temperature|voltage/&lt;j&gt;, fault_mismatch</td></tr>
<tr><td>ServoCommandTrajectory</td><td>positions|velocities|feedforward_effort|stiffness|damping|max_effort/&lt;j&gt;, channels/latency</td></tr>
<tr><td>ImuTrajectory</td><td>rotation, rx ry rz, angular_velocity, linear_acceleration (+ per axis)</td></tr>
<tr><td>WrenchTrajectory</td><td>force, torque, forces/x|y|z, torques/x|y|z</td></tr>
<tr><td>ImageTrajectory</td><td>pixels, mae, rmse, psnr</td></tr>
<tr><td>PointCloud (single)</td><td>distance, distance_reverse, chamfer</td></tr>
<tr><td>JointState / ServoState / ServoCommand / Imu / Wrench / Image (single)</td><td>as their stream, as one sample, stamps ignored</td></tr></table>
{''.join(body)}
<p style="color:var(--mute)">Stills only, no hero video: these are recorded viewers; every image is a capture of the real Rerun 0.40 web viewer.</p>
</body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
