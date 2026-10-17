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
    rows = []
    for line in Path(index).read_text().splitlines():
        r = json.loads(line)
        sev = r.get("severity_counts") or {}
        status = "ok" if "score" in r else ("skip" if r.get("skipped") else "err")
        rows.append("|".join([r["id"], status, str(sev.get("critical", "")), str(sev.get("warning", "")),
                              str(r.get("episodes", "")), str(r.get("usable_fraction", "")),
                              ",".join(r.get("checks") or [])]))
    if os.environ.get("GITHUB_ACTIONS"):   # GitHub keeps 10 annotations per job: pack the rows
        chunk, n = [], 0
        for row in rows + [None]:
            if row is None or sum(len(x) + 1 for x in chunk) + len(row) > 3700:
                if chunk:
                    n += 1
                    print(f"::notice title=reaudit-{k}-{n}::" + "~".join(chunk).replace("%", "%25"))
                chunk = []
            if row is not None:
                chunk.append(row)
    else:
        print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
