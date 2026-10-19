"""Install rdq from PyPI into a fresh venv and run `rdq check` on Hub datasets: the path a new user takes.

    python experiments/pypi_smoke.py owner/name ...
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time


def sh(cmd: list[str]) -> tuple[int, str, float]:
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr), time.time() - t


def main() -> int:
    venv = tempfile.mkdtemp()
    lines = []
    code, out, dt = sh([sys.executable, "-m", "venv", venv])
    pip, rdq = os.path.join(venv, "bin", "pip"), os.path.join(venv, "bin", "rdq")
    code, out, dt = sh([pip, "install", "-q", "rdq"])
    v = sh([os.path.join(venv, "bin", "python"), "-c", "import rdq;print(rdq.__version__)"])[1].strip()
    lines.append(f"pip install rdq: exit {code} in {dt:.0f}s, version {v}")
    for repo in sys.argv[1:]:
        code, out, dt = sh([rdq, "check", repo, "--no-color"])
        head = "\n".join(l for l in out.splitlines() if l.strip() and "Fetching" not in l and "HF_TOKEN" not in l)[:900]
        lines.append(f"--- rdq check {repo}: exit {code} in {dt:.0f}s\n{head}")
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_ACTIONS"):
        esc = text.replace("%", "%25").replace("\n", "%0A")
        for i in range(0, len(esc), 3500):
            print(f"::notice title=smoke-{i // 3500}::" + esc[i:i + 3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
