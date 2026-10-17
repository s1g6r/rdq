"""Diagnose the viewpoint test on real datasets: per-episode share of 'background' thumbnail pixels at a few
stability thresholds, and background correlations between episodes (all pixels vs. stable pixels only).

    python experiments/viewpoint_diag.py EPITECH-LILLE/eval_training:observation.images.front ...
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq import rawvideo as rv  # noqa: E402
from rdq.cli import _resolve_path  # noqa: E402
from rdq.dataset import load_dataset  # noqa: E402
from rdq.video import _video_file, viewpoints  # noqa: E402


def thumbs(ds, cam, max_eps=60):
    out, measured = {}, {}
    for ep in sorted(ds.episodes.values(), key=lambda e: e.index)[:max_eps]:
        path, start, end = _video_file(ds, cam, ep)
        if path is None or not path.exists():
            continue
        if str(path) not in measured:
            measured[str(path)] = rv.measure_file(str(path))
        r = rv.segment_record(measured[str(path)], start, end, f"episode {ep.index}")
        th = ((r.get("dec") or {}).get("samples") or {}).get("thumb")
        if th is not None and len(th) >= 4:
            out[ep.index] = (th.astype(np.float32), r)
    return out


def main() -> int:
    for spec in sys.argv[1:]:
        repo, cam = spec.split(":")
        path, _ = _resolve_path(repo, None, True, 1500)
        ds = load_dataset(path)
        T = thumbs(ds, cam)
        eps = sorted(T)
        rows = {}
        for e in eps:
            a = T[e][0]
            sd = a.std(axis=0)
            rows[e] = {f"s{t}": round(float((sd < t).mean()), 2) for t in (4, 8, 12, 16, 24)}
            rows[e]["n"] = len(a)
        med = np.median
        bg = {e: med(T[e][0], axis=0).ravel() for e in eps}

        def corr(x, y, m=None):
            if m is not None:
                x, y = x[m], y[m]
            x = (x - x.mean()) / (x.std() + 1e-6)
            y = (y - y.mean()) / (y.std() + 1e-6)
            return round(float(np.mean(x * y)), 2)

        e0 = eps[0]
        pair = {}
        for e in eps[:: max(1, len(eps) // 12)]:
            m8 = (T[e0][0].std(0).ravel() < 8) & (T[e][0].std(0).ravel() < 8)
            m16 = (T[e0][0].std(0).ravel() < 16) & (T[e][0].std(0).ravel() < 16)
            pair[e] = {"all": corr(bg[e0], bg[e]), "m8": corr(bg[e0], bg[e], m8) if m8.mean() > 0.05 else None,
                       "m16": corr(bg[e0], bg[e], m16) if m16.mean() > 0.05 else None,
                       "share8": round(float(m8.mean()), 2), "share16": round(float(m16.mean()), 2)}
        v = viewpoints([{"episode": e, "dec": {"samples": {"thumb": T[e][0]}}} for e in eps])
        summary = {"repo": repo, "cam": cam, "eps": len(eps),
                   "stable_median": {k: round(float(med([rows[e][k] for e in eps])), 2) for k in ("s4", "s8", "s12", "s16", "s24")},
                   "vs_first": pair, "viewpoints_now": v and v["ranges"]}
        line = json.dumps(summary, separators=(",", ":"))
        print(line)
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::notice title=diag-{repo.split('/')[-1][:30]}::" + line[:3800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
