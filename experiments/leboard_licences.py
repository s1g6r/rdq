"""Licence check for every dataset on LeBoard (leboard.xyz), at the commit LeBoard pins.

Reads experiments/leboard_datasets.txt (owner/name@sha per line), fetches each dataset card at
that commit, runs rdq's licence and provenance check, and writes a CSV. In CI it also prints the
rows as GitHub annotations.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdq.provenance import analyse_card  # noqa: E402


def main() -> int:
    from huggingface_hub import hf_hub_download

    rows = []
    for line in Path(__file__).with_name("leboard_datasets.txt").read_text().split():
        repo, sha = line.split("@")
        try:
            readme = Path(hf_hub_download(repo, "README.md", repo_type="dataset", revision=sha)).read_text(errors="replace")
        except Exception:  # noqa: BLE001 - no card at that commit
            readme = ""
        p = analyse_card(readme, repo_id=repo)
        up = p.upstream or {}
        u = [x for x in (up.get("upstream") or []) if x]
        rows.append({
            "dataset": repo, "commit": sha[:12], "declared_licence": p.license or "none",
            "result": {"go": "OK", "check": "Check first", "stop": "Stop"}[p.bottom_level],
            "status": up.get("status"), "original_dataset": "; ".join(x["name"] for x in u),
            "original_licence": "; ".join(x.get("license") or "" for x in u),
            "matched_by": "; ".join(x.get("evidence", "") for x in u),
            "original_terms_confidence": "; ".join(x.get("confidence", "") for x in u),
            "original_statement_url": "; ".join(x.get("statement_url", "") for x in u),
            "bottom_line": p.bottom_line, "why": " | ".join(p.notes),
            "suggested_card_fix": (p.card_fix or "").replace("\n", " | "),
        })
    out = Path(os.environ.get("OUT", "leboard_licence_check.csv"))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if os.environ.get("GITHUB_ACTIONS"):
        buf = io.StringIO()
        w = csv.writer(buf)
        for r in rows:  # compact rows for annotations
            w.writerow([r["dataset"], r["declared_licence"], r["result"], r["status"], r["original_dataset"],
                        r["original_licence"], r["matched_by"]])
        lines = buf.getvalue().splitlines()
        chunk, k = [], 0
        for ln in lines:
            chunk.append(ln)
            if sum(len(x) + 3 for x in chunk) > 2800:
                print(f"::notice title=leboard-{k}::" + " %0A".join(chunk))
                chunk, k = [], k + 1
        if chunk:
            print(f"::notice title=leboard-{k}::" + " %0A".join(chunk))
    print(f"{len(rows)} datasets -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
