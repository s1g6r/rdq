"""rdq check --video-full on Hub datasets: defects and duplicated episodes in real LeRobot data.

    python experiments/video_full_hub.py --out vfull lerobot/svla_so101_pickplace ...
    python experiments/video_full_hub.py --out vfull --auto 20 --max-mb 600   # popular community datasets

--auto picks the most-downloaded datasets tagged LeRobot outside the lerobot org whose data and videos
fit under --max-mb. Results go to OUT/summary.json and, in CI, to annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def pick_auto(n: int, max_mb: float, scan: int = 800, skip: int = 0, exclude: set[str] | None = None) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    for d in api.list_datasets(filter="LeRobot", sort="downloads", limit=scan):
        if d.id.startswith("lerobot/") or d.id in (exclude or set()):
            continue
        try:
            info = api.dataset_info(d.id, files_metadata=True)
        except Exception:  # noqa: BLE001
            continue
        files = [f for f in info.siblings if f.rfilename.startswith(("data/", "meta/", "videos/"))]
        if not any(f.rfilename.startswith("videos/") and f.rfilename.endswith(".mp4") for f in files):
            continue
        if not any(f.rfilename == "meta/info.json" for f in files):
            continue
        size = sum((f.size or 0) for f in files) / 1e6
        if 5 < size <= max_mb:
            out.append(d.id)
        if len(out) >= n + skip:
            break
    return out[skip:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--out", default="vfull")
    ap.add_argument("--auto", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N --auto picks (already checked)")
    ap.add_argument("--exclude-file", help="text file of dataset ids already checked (one per line)")
    ap.add_argument("--max-mb", type=float, default=600)
    ap.add_argument("--shard", default="0/1", help="k/n: check every n-th dataset starting at k (parallel jobs)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = set(Path(args.exclude_file).read_text().split()) if args.exclude_file else set()
    ids = list(args.datasets) + (pick_auto(args.auto, args.max_mb, skip=args.skip, exclude=exclude) if args.auto else [])
    k, n = (int(x) for x in args.shard.split("/"))
    ids = ids[k::n]
    time.sleep(45 * k)  # stagger the parallel jobs
    print("::notice title=datasets::" + " ".join(ids))
    rows = []
    for ds in ids:
        j = out / (ds.replace("/", "__").replace("@", "__") + ".json")
        t0 = time.time()
        for attempt in range(4):  # the Hub rate-limits parallel anonymous downloads: back off and retry
            name, _, rev = ds.partition("@")       # owner/name@commit pins a revision (LeBoard's pinned SHAs)
            p = subprocess.run([sys.executable, "-m", "rdq.cli", "check", name, "--video-full", "--json", str(j),
                                "--max-mb", str(args.max_mb), "--no-color"] + (["--revision", rev] if rev else []),
                               capture_output=True, text=True)
            if j.exists() or "Hub" not in (p.stderr or "") or attempt == 3:
                break
            time.sleep(90 * (attempt + 1))
        el = time.time() - t0
        row = {"dataset": ds, "seconds": round(el), "exit": p.returncode}
        if j.exists():
            r = json.loads(j.read_text())
            row["episodes"] = (r.get("dataset") or {}).get("episodes") or (r.get("counts") or {}).get("episodes")
            row["video"] = {f["check"] + (":" + f["details"]["video_key"].split(".")[-1]
                                          if (f.get("details") or {}).get("video_key") else ""): f.get("episodes", [])[:30]
                            for f in r.get("findings", []) if f["check"].startswith("video.")}
            vp = [f for f in r.get("findings", []) if f["check"] == "video.viewpoint_change"]
            if vp:
                row["viewpoints"] = [(f["details"]["video_key"], f["details"]["groups"]) for f in vp]
            dup = next((f for f in r.get("findings", []) if f["check"] == "video.duplicate_episode"), None)
            if dup:
                row["dup_pairs"] = (dup.get("details") or {}).get("pairs", [])[:20]
        else:
            row["error"] = (p.stderr or p.stdout)[-300:]
        rows.append(row)
        print(json.dumps(row), flush=True)
        if os.environ.get("GITHUB_ACTIONS"):      # one annotation per dataset, as it finishes (a timeout keeps the rest)
            print(f"::notice title=vfull-{k}-{len(rows)}::" + json.dumps(row, separators=(",", ":"))[:3800].replace("%", "%25"),
                  flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
