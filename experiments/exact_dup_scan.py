"""Episodes whose robot data (state and action) is byte-for-byte identical to another episode's, across an org's
LeRobot datasets. Real recordings never repeat sensor values exactly, so an identical episode is a stored copy.
Reads only the episode_index / state / action columns of the data files.

    python experiments/exact_dup_scan.py RoboCOIN --shard 0/5
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


def scan(repo, fs):
    base = f"datasets/{repo}"
    info = json.loads(fs.read_text(f"{base}/meta/info.json"))
    feats = info.get("features") or {}
    keys = [k for k in ("observation.state", "action") if k in feats]
    if not keys:
        return {"skip": "no state/action"}
    per_ep = {}
    for p in sorted(fs.glob(f"{base}/data/**/*.parquet")):
        with fs.open(p, "rb") as f:
            t = pq.read_table(f, columns=["episode_index", *keys]).to_pandas()
        for e, g in t.groupby("episode_index", sort=False):
            h = per_ep.setdefault(int(e), hashlib.sha1())
            for k in keys:
                h.update(np.ascontiguousarray(np.stack(g[k].to_numpy()).astype(np.float64)).tobytes())
    groups = {}
    for e, h in per_ep.items():
        groups.setdefault(h.hexdigest(), []).append(e)
    dup_groups = sorted(sorted(v) for v in groups.values() if len(v) > 1)
    copies = sum(len(g) - 1 for g in dup_groups)
    adjacent = sum(1 for g in dup_groups if len(g) == 2 and g[1] - g[0] == 1)
    return {"episodes": len(per_ep), "copies": copies, "groups": len(dup_groups), "adjacent_pairs": adjacent,
            "examples": dup_groups[:6]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("org")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    api, fs = HfApi(), HfFileSystem()
    ids = sorted(d.id for d in api.list_datasets(author=a.org, limit=5000))[k::n]
    rows = []
    for repo in ids:
        try:
            r = scan(repo, fs)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {str(e)[:100]}"}
        r["repo"] = repo.split("/")[-1]
        rows.append(r)
    checked = [r for r in rows if "copies" in r]
    hit = [r for r in checked if r["copies"]]
    summary = {"shard": a.shard, "datasets": len(ids), "checked": len(checked), "with_copies": len(hit),
               "episodes": sum(r["episodes"] for r in checked), "copies": sum(r["copies"] for r in checked),
               "errors": sum(1 for r in rows if "error" in r)}
    print(json.dumps(summary))
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=eds-summary-{k}::" + json.dumps(summary))
        compact = [[r["repo"], r["episodes"], r["copies"], r["adjacent_pairs"]] for r in hit]
        s = json.dumps(compact, separators=(",", ":"))
        for j in range(0, len(s), 3800):
            print(f"::notice title=eds-hits-{k}-{j // 3800}::" + s[j:j + 3800])
        ex = json.dumps([[r["repo"], r["examples"]] for r in hit[:15]], separators=(",", ":"))
        print(f"::notice title=eds-ex-{k}::" + ex[:3800])
        errs = json.dumps([[r["repo"], r["error"]] for r in rows if "error" in r][:10])
        print(f"::notice title=eds-err-{k}::" + errs[:3800])


if __name__ == "__main__":
    main()
