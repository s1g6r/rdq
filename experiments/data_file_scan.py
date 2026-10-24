"""How many v3.0 LeRobot datasets on the Hub carry the lerobot#2401 file-boundary bug: an episode whose metadata
(data/chunk_index, data/file_index) names a data file it isn't in. Reads only meta/episodes and the episode_index
column of each data file (HfFileSystem + pyarrow fetch just those bytes).

    python experiments/data_file_scan.py --limit 600 --shard 0/4
"""
import argparse
import json
import os
import time

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


def scan(repo: str, fs: HfFileSystem) -> dict:
    base = f"datasets/{repo}"
    info = json.loads(fs.read_text(f"{base}/meta/info.json"))
    if not str(info.get("codebase_version", "")).startswith("v3"):
        return {"skip": "not v3"}
    data_files = sorted(p for p in fs.glob(f"{base}/data/**/*.parquet"))
    if len(data_files) < 2:
        return {"skip": "one data file", "files": len(data_files)}
    if len(data_files) > 40:
        return {"skip": "too many files", "files": len(data_files)}
    tmpl = info.get("data_path")
    metas = []
    for p in sorted(fs.glob(f"{base}/meta/episodes/**/*.parquet")):
        with fs.open(p, "rb") as f:
            metas.append(pq.read_table(f, columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pandas())
    import pandas as pd
    meta = pd.concat(metas)
    where = {}
    for p in data_files:
        with fs.open(p, "rb") as f:
            eps = pq.read_table(f, columns=["episode_index"]).column(0).to_pylist()
        rel = p[len(base) + 1:]
        for e in set(eps):
            where.setdefault(int(e), set()).add(rel)
    wrong = []
    for r in meta.itertuples(index=False):
        named = tmpl.format(chunk_index=int(r[1]), file_index=int(r[2]))
        e = int(r[0])
        if e in where and named not in where[e]:
            wrong.append(e)
    return {"episodes": len(meta), "files": len(data_files), "wrong": len(wrong), "wrong_eps": wrong[:20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    api, fs = HfApi(), HfFileSystem()
    ids = [d.id for d in api.list_datasets(filter="LeRobot", sort="downloads", limit=a.limit)][k::n]
    rows = []
    for i, repo in enumerate(ids):
        t0 = time.time()
        try:
            r = scan(repo, fs)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        r["repo"] = repo
        rows.append(r)
        if r.get("wrong") or (i % 25 == 0):
            print(json.dumps(r), flush=True)
        if r.get("wrong") and os.environ.get("GITHUB_ACTIONS"):   # report as found, so a timeout keeps the hits
            print(f"::notice title=dfs-live-{k}-{i}::" + json.dumps(r)[:3500], flush=True)
    hit = [r for r in rows if r.get("wrong")]
    checked = [r for r in rows if "wrong" in r]
    summary = {"shard": a.shard, "listed": len(ids), "multi_file_v3_checked": len(checked), "with_bug": len(hit),
               "skips": {s: sum(1 for r in rows if r.get("skip") == s) for s in ("not v3", "one data file")},
               "errors": sum(1 for r in rows if "error" in r)}
    if os.environ.get("GITHUB_ACTIONS"):
        print("::notice title=dfs-summary-%d::%s" % (k, json.dumps(summary)))
        for j, r in enumerate(hit):
            print("::notice title=dfs-hit-%d-%d::%s" % (k, j, json.dumps(r)[:3500]))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
