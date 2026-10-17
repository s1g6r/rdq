"""Run `rdq check` on Hub datasets and print the full error trace as an annotation (for crash triage).

    python experiments/trace_check.py owner/name [--video-full]
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    args = sys.argv[1:]
    flags = [a for a in args if a.startswith("--")]
    for ds in (a for a in args if not a.startswith("--")):
        p = subprocess.run([sys.executable, "-m", "rdq.cli", "check", ds, "--no-color", "--max-mb", "1500", *flags],
                           capture_output=True, text=True)
        tail = (p.stderr or "")[-3500:]
        print(f"== {ds} exit {p.returncode}\n{tail}")
        if os.environ.get("GITHUB_ACTIONS"):
            msg = f"exit {p.returncode} " + tail.replace("%", "%25").replace("\n", "%0A")
            print(f"::notice title=trace-{ds.split('/')[-1][:30]}::{msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
