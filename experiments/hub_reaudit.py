"""Re-audit a fixed list of Hub datasets (tables only) with the current rdq, one shard per CI job.

    python experiments/hub_reaudit.py experiments/results/hub_audit_210.txt --shard 0/4 --out reaudit

Prints one GitHub annotation per dataset with its severity counts and the checks that fired, so the
results can be read without downloading the workflow artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq.hub import audit_repos  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="reaudit")
    ap.add_argument("--max-mb", type=float, default=250)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    ids = Path(a.ids).read_text().split()[k::n]
    index = audit_repos(ids, a.out, max_mb=a.max_mb, log=lambda *x: print(*x, flush=True))
    for i, line in enumerate(Path(index).read_text().splitlines(), 1):
        r = json.loads(line)
        row = {"id": r["id"], "sev": r.get("severity_counts"), "checks": r.get("checks"),
               "eps": r.get("episodes"), "usable": r.get("usable_fraction"),
               "skip": r.get("skipped"), "err": (r.get("error") or "")[:200] or None}
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::notice title=reaudit-{k}-{i}::" + json.dumps(row, separators=(",", ":"))[:3800].replace("%", "%25"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
