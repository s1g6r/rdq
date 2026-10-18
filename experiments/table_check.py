"""rdq check without video (tables, metadata and card only) on Hub datasets; one annotation per dataset.

    python experiments/table_check.py owner/name[@rev] ...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq.cli import _resolve_path  # noqa: E402
from rdq.report import audit  # noqa: E402


def main() -> int:
    for spec in sys.argv[1:]:
        repo, _, rev = spec.partition("@")
        try:
            path, _ = _resolve_path(repo, None, False, None, rev or None)
            r = audit(path, repo_id=repo)
            out = {"dataset": spec, "episodes": r["dataset"]["episodes"],
                   "findings": [{"check": f["check"], "sev": f["severity"], "n": len(f["episodes"]),
                                 "eps": f["episodes"][:40], "msg": f["message"][:160]} for f in r["findings"]]}
        except Exception as e:
            out = {"dataset": spec, "error": repr(e)}
        line = json.dumps(out, separators=(",", ":"), default=str)
        print(line)
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::notice title=table-{repo.split('/')[-1][:40]}::" + line[:3900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
