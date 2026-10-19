"""Run `rdq check` on Hub datasets and print the terminal report (one annotation per dataset), for the README demo.

    python experiments/cli_demo.py owner/name[@rev] [--video-full] ...
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    for spec in [a for a in args if not a.startswith("--")]:
        repo, _, rev = spec.partition("@")
        cmd = ["rdq", "check", repo, "--no-color", *flags] + (["--revision", rev] if rev else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = (out.stdout + out.stderr).replace("\r", "")
        print(text)
        if os.environ.get("GITHUB_ACTIONS"):
            esc = text.replace("%", "%25").replace("\n", "%0A")
            for i in range(0, len(esc), 3500):
                print(f"::notice title=demo-{repo.split('/')[-1][:30]}-{i // 3500}::" + esc[i:i + 3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
