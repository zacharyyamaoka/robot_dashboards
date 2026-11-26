"""The landing page: every dashboard with its still, built by robot_msg_viz's builder.

WHY (Zach, 2026-11-25): "for public github repos use github io, but for private do cloud flare". This repo went public
the same day, so its README's live-page line (the deploy's one switch) names the github.io page.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest

from robot_dashboards.core import all_dashboards

REPO = Path(__file__).resolve().parents[1]
MEDIA = REPO / "website" / "media"
LIVE = re.compile(r"\[▶ Live landing page\]\((https://[^)\s]+)\)")


def test_the_live_page_is_on_github_pages_because_the_repo_is_public():
    lines = LIVE.findall((REPO / "README.md").read_text())
    assert lines == ["https://zacharyyamaoka.github.io/robot_dashboards/"], lines
    workflow = (REPO / ".github" / "workflows" / "site.yml").read_text()
    assert "actions/deploy-pages" in workflow, "the deploy publishes to GitHub Pages"


def test_every_dashboard_has_its_committed_still_and_the_hero_is_there():
    missing = [d.key for d in all_dashboards()
               if not (MEDIA / "shots" / f"{d.key}.webp").is_file() or not (MEDIA / "shots" / f"{d.key}-480.webp").is_file()]  # fmt: skip
    assert not missing, f"no still in website/media/shots for {missing}"
    assert (MEDIA / "hero.mp4").is_file() and (MEDIA / "hero.webp").is_file()
    head = (MEDIA / "hero.mp4").read_bytes()[:400_000]
    assert head[4:8] == b"ftyp" and b"avc1" in head, "H.264 in an mp4, moov first"


def fake_viewer(folder: Path) -> Path:
    """Rerun's viewer files in shape only (the real ones come from rerun-sdk in CI): the loader line the build patches."""
    folder.mkdir(parents=True)
    (folder / "index.html").write_text('<script>const response = await fetch("./re_viewer_bg.wasm");</script>')
    for name in ("re_viewer_bg.wasm", "re_viewer.js", "favicon.ico", "apple-touch-icon.png"):
        (folder / name).write_bytes(b"x" * 64)
    return folder


def test_the_page_builds_from_the_plugin_alone(tmp_path):
    site_build = pytest.importorskip("robot_msg_viz.site.build")
    from robot_msg_viz.gallery import use_case_entries

    gallery = [e for e in use_case_entries() if e.source == "robot_dashboards"]
    site = site_build.Site(title="robot_dashboards", tagline="stock dashboards", code="", use_cases_only=True,
                           lead_html=(REPO / "website" / "intro.html").read_text(), private=True,
                           github="https://github.com/zacharyyamaoka/robot_dashboards",
                           types_url="https://zacharyyamaoka.github.io/robot_msg_viz/")  # fmt: skip
    out = tmp_path / "site"
    site_build.build(out, shots=MEDIA / "shots", gallery=gallery, site=site, hero=MEDIA / "hero.mp4",
                     viewer=fake_viewer(tmp_path / "viewer"))  # fmt: skip
    index = (out / "index.html").read_text()
    assert 'class="layers"' in index, "the adapters -> robot_msgs -> viewers diagram"
    assert "no screenshot yet" not in index and 'src="media/hero.mp4"' in index
    for dashboard in all_dashboards():
        page = (out / "use-cases" / f"{dashboard.key}.html").read_text()
        assert html.escape(dashboard.summary) in page, "the question it answers"
        assert f"dashboards/{dashboard.key}.py" in page, "its code"
        assert (out / "rrd" / f"{dashboard.key}.rrd").stat().st_size > 0
    assert (out / "viewer.json").is_file(), "Open in browser uses the viewer served from the page itself"
