"""Diagnose the camera-swap and duplicate checks on real datasets.

    python experiments/swapdup_diag.py swap:owner/name[@rev]:camA,camB  dup:owner/name[@rev]:3-4,10-12 ...

swap: background correlations behind camera_swaps (reference picks, each camera's own match, the cross match).
dup:  data_agreement for episode pairs, with the raw pair and typical distances (tables only, no video).
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq import rawvideo as rv  # noqa: E402
from rdq.cli import _resolve_path  # noqa: E402
from rdq.dataset import load_dataset  # noqa: E402
from rdq.video import (SWAP_CORR, _traj_distance, _video_file, camera_swaps, data_agreement,  # noqa: E402
                       episode_backgrounds)


def emit(tag: str, obj) -> None:
    line = json.dumps(obj, separators=(",", ":"), default=str)
    print(line)
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=diag-{tag[:40]}::" + line[:3900])


def load(spec: str, video: bool):
    repo, _, rev = spec.partition("@")
    path, _ = _resolve_path(repo, None, video, 3000, rev or None)
    return load_dataset(path)


def swap(spec: str, cams: str) -> None:
    ds = load(spec, True)
    a, b = cams.split(",")
    bgs = {}
    for cam in (a, b):
        measured, records = {}, []
        for ep in sorted(ds.episodes.values(), key=lambda e: e.index):
            path, start, end = _video_file(ds, f"observation.images.{cam}", ep)
            if path is None or not path.exists():
                continue
            if str(path) not in measured:
                measured[str(path)] = rv.measure_file(str(path))
            r = rv.segment_record(measured[str(path)], start, end, f"episode {ep.index}")
            r["episode"] = ep.index
            records.append(r)
        bgs[cam] = episode_backgrounds(records)
    A, B = bgs[a], bgs[b]
    c = lambda x, y: round(float(np.mean(x * y)), 3)  # noqa: E731
    eps = sorted(set(A) & set(B))
    same = {e: c(A[e], B[e]) for e in eps}
    rows = {}
    for r in eps[:3] + [e for e in (8, 11, 12) if e in A]:
        rows[r] = {e: [c(A[e], A[r]), c(B[e], B[r]), c(A[e], B[r]), c(B[e], A[r])] for e in eps}
    emit(f"swap-{spec.split('/')[-1]}", {"spec": spec, "swap_corr": SWAP_CORR, "now": camera_swaps(bgs),
                                         "A_vs_B_same_episode": same})
    for r, d in rows.items():
        emit(f"swap-ref{r}", {"ref": r, "own_a,own_b,a_vs_bref,b_vs_aref": d})


def dup(spec: str, pairs: str) -> None:
    ds = load(spec, False)
    key = ds.state_key or ds.action_key
    out = {"spec": spec, "key": key, "eps": len(ds.episodes)}
    ps = [tuple(int(x) for x in p.split("-")) for p in pairs.split(",")]
    if len(ps) == 1 and ps[0][0] > ps[0][1]:
        ps = list(combinations(range(ps[0][1], ps[0][0] + 1), 2))
    res = {}
    for a, b in ps:
        if a in ds.episodes and b in ds.episodes:
            fa, fb = ds.episodes[a].features.get(key), ds.episodes[b].features.get(key)
            res[f"{a}-{b}"] = {"agree": data_agreement(ds, a, b),
                               "len": [len(fa) if fa is not None else None, len(fb) if fb is not None else None],
                               "raw": _traj_distance(np.asarray(fa, float), np.asarray(fb, float), 1) if fa is not None else None}
    out["pairs"] = res
    emit(f"dup-{spec.split('/')[-1]}", out)


def main() -> int:
    for arg in sys.argv[1:]:
        kind, spec, rest = arg.split(":", 2)
        try:
            (swap if kind == "swap" else dup)(spec, rest)
        except Exception as e:  # keep going for the other specs
            emit(f"err-{spec.split('/')[-1]}", {"spec": spec, "error": repr(e)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
