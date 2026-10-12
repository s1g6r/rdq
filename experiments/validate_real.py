"""Summarise an `rdq audit` run on a hand-picked set of real Hub datasets.

The set (see .github/workflows/validate.yml) covers the cases the synthetic tests can't:
channels that "glitch" in most episodes, depth and other non-signal features, missing
columns, NaNs, desync and idle flags found by earlier versions. Compare the table with
the previous audit to see what each change did on real data.

    rdq audit --repos A B C --out val && python experiments/validate_real.py val
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(out: Path) -> str:
    last = {}
    for line in open(out / "audit_index.jsonl"):
        r = json.loads(line)
        last[r["id"]] = r
    lines = ["| Dataset | Episodes | Usable | Score | Findings (episodes) | Not checked |", "|---|---|---|---|---|---|"]
    totals: Counter = Counter()
    for rid, r in last.items():
        if "report" not in r:
            lines.append(f"| {rid} | | | | {r.get('error') or r.get('skipped') or 'not audited'} | |")
            continue
        rep = json.loads((out / r["report"]).read_text())
        n = rep["dataset"]["episodes"]
        found = []
        totals.update({f["check"] for f in rep["findings"]})
        for f in rep["findings"]:
            found.append(f"{f['check']} ({len(f['episodes'])})" if f["episodes"] else f["check"])
        lines.append(
            f"| {rid} | {n} | {len(rep['score']['usable_episodes'])} | {rep['score']['dataset_score']} | "
            f"{'; '.join(found) or 'none'} | {'; '.join(rep.get('not_checked') or []) or '-'} |"
        )
    lines += ["", "Datasets per check: " + ", ".join(f"{k} {v}" for k, v in totals.most_common())]
    return "\n".join(lines)


if __name__ == "__main__":
    text = main(Path(sys.argv[1]))
    print(text)
    (Path(sys.argv[1]) / "validation.md").write_text(text)
