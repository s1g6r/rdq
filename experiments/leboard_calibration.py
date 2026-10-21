"""Score rdq against LeBoard's calibration datasets (rogation/libero10-*): copies of lerobot/libero (libero_10)
with one defect injected on purpose and the ground truth in calibration/injection_log.json.

    python experiments/leboard_calibration.py --out calib rogation/libero10-freeze-r10-c0 ...

For every finding rdq reports, prints how many of its episodes are in the injected set (hit) and how many are not
(extra), next to the number of injected episodes. Comparing with libero10-clean-c0 separates rdq's reaction to the
defect from what it already says about the base data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def truth(ds: str) -> dict:
    from huggingface_hub import hf_hub_download

    try:
        p = hf_hub_download(ds, "calibration/injection_log.json", repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}
    j = json.loads(Path(p).read_text())
    aff = j.get("affected") or []
    eps = sorted({a["episode"] for a in aff if isinstance(a, dict) and "episode" in a})
    return {"defect": j.get("defect"), "params": j.get("params"), "episodes": eps,
            "output_episodes": (j.get("output") or {}).get("episodes"), "sample": aff[:2]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--out", default="calib")
    ap.add_argument("--max-mb", type=float, default=3000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        j = out / (ds.replace("/", "__") + ".json")
        t0 = time.time()
        for attempt in range(3):
            p = subprocess.run([sys.executable, "-m", "rdq.cli", "check", ds, "--video-full", "--json", str(j),
                                "--max-mb", str(args.max_mb), "--no-color"], capture_output=True, text=True)
            if j.exists() or attempt == 2:
                break
            time.sleep(60 * (attempt + 1))
        t = truth(ds)
        g = set(t.get("episodes") or [])
        row = {"dataset": ds.split("/")[-1], "s": round(time.time() - t0), "defect": t.get("defect"),
               "params": t.get("params"), "injected": len(g), "of": t.get("output_episodes")}
        if t.get("sample"):
            row["sample"] = t["sample"]
        if j.exists():
            r = json.loads(j.read_text())
            checks = {}
            for f in r.get("findings", []):
                eps = set(f.get("episodes") or [])
                d = f.get("details") or {}
                cam = d.get("video_key") or d.get("camera") or d.get("feature") or ""
                key = f["check"] + (":" + str(cam).split(".")[-1] if cam else "")
                hit = len(eps & g)
                checks[key] = [f["severity"][0], len(eps), hit, len(eps) - hit]
            row["checks"] = checks          # check: [severity, flagged, hit, extra]
            flagged = set()
            for f in r.get("findings", []):
                if f["severity"] != "info":
                    flagged |= set(f.get("episodes") or [])
            row["caught_any_warning"] = len(g & flagged)
            row["not_checked"] = r.get("not_checked")
        else:
            row["error"] = (p.stderr or p.stdout)[-400:]
        rows.append(row)
        line = json.dumps(row, separators=(",", ":"))
        print(line, flush=True)
        if os.environ.get("GITHUB_ACTIONS"):
            for i in range(0, len(line), 3800):
                print(f"::notice title=calib-{row['dataset']}-{i // 3800}::" + line[i:i + 3800].replace("%", "%25"),
                      flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
