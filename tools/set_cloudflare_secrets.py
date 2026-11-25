#!/usr/bin/env python3
"""Turn on the Cloudflare deploy of this repo's landing page, once, after the Cloudflare setup.

    python3 /home/bam/robot_dashboards/tools/set_cloudflare_secrets.py

Reads the token and account that blackbox_agent's `tools/cloudflare_pages.py setup` saved
(~/.config/blackbox_agent/cloudflare.json, mode 600), sets this repo's CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID
secrets with `gh secret set` (the token goes in on stdin, never on a command line or in a log) and starts the site
workflow, which then deploys to the host the README's live-page line names, behind Cloudflare Access.

WHY its own script: `cloudflare_pages.py add` relinks a black-box box (its pages.yml calls blackbox_agent's site.yml);
this repo has its own workflow (.github/workflows/site.yml) and its README line already names the Cloudflare host, so
only the two secrets are missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

CONFIG = Path.home() / ".config" / "blackbox_agent" / "cloudflare.json"
REPO = "zacharyyamaoka/robot_dashboards"


def main() -> int:
    if not CONFIG.exists():
        print(f"{CONFIG} is missing: run `python3 ~/blackbox_agent/tools/cloudflare_pages.py setup --allow-email <you>` "
              "first", file=sys.stderr)
        return 1
    cfg = json.loads(CONFIG.read_text())
    for name, key in (("CLOUDFLARE_API_TOKEN", "token"), ("CLOUDFLARE_ACCOUNT_ID", "account")):
        if not cfg.get(key):
            print(f"{CONFIG} has no {key!r}", file=sys.stderr)
            return 1
        subprocess.run(["gh", "secret", "set", name, "-R", REPO], input=cfg[key], text=True, check=True)
    subprocess.run(["gh", "workflow", "run", "site.yml", "-R", REPO], check=False)
    print(f"{REPO}: secrets set; the site workflow deploys it behind Access (gh run list -R {REPO} -w site)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
