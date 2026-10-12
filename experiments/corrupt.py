"""Make a corrupted copy of a real LeRobot v3.0 dataset, for controlled experiments.

    python experiments/corrupt.py SRC_DIR DST_DIR --rate 0.3 --seed 0

A fraction of episodes get one of three realistic recording faults, applied to
the action/state columns only (videos are linked, not copied):

  idle    the teleop log froze: action and state stay at the first frame
  desync  the action stream was written 8 frames late
  glitch  a 3-frame dropout where state and action jump to garbage

The ground truth goes to DST_DIR/corruption.json so you can measure what
rdq catches. Episode lengths, metadata and stats.json are left untouched, as
they would be in a real broken recording.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FAULTS = ("idle", "desync", "glitch")


def _apply(values: np.ndarray, state: np.ndarray | None, kind: str, rng: np.random.Generator):
    n = len(values)
    if kind == "idle":
        values = np.repeat(values[:1], n, axis=0)
        if state is not None:
            state = np.repeat(state[:1], n, axis=0)
    elif kind == "desync":
        k = min(8, n - 1)
        values = np.vstack([np.repeat(values[:1], k, axis=0), values[:-k]])
    elif kind == "glitch":
        t = int(rng.integers(1, max(2, n - 4)))
        span = np.ptp(values, axis=0) + 1e-6
        jump = rng.uniform(2, 4, values.shape[1]) * span * rng.choice([-1, 1], values.shape[1])
        values = values.copy()
        values[t : t + 3] += jump
        if state is not None:
            sspan = np.ptp(state, axis=0) + 1e-6
            state = state.copy()
            state[t : t + 3] += rng.uniform(2, 4, state.shape[1]) * sspan
    return values, state


def corrupt_dataset(src: Path, dst: Path, rate: float, seed: int, state_key: str = "observation.state") -> dict:
    rng = np.random.default_rng(seed)
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    if (src / "videos").exists():
        (dst / "videos").symlink_to((src / "videos").resolve())
    if (src / "README.md").exists():
        shutil.copy(src / "README.md", dst / "README.md")

    info = json.loads((src / "meta" / "info.json").read_text())
    n_eps = info["total_episodes"]
    chosen = rng.choice(n_eps, int(round(rate * n_eps)), replace=False)
    kinds = {int(e): FAULTS[i % len(FAULTS)] for i, e in enumerate(rng.permutation(chosen))}

    for path in sorted((src / "data").glob("**/*.parquet")):
        out_path = dst / path.relative_to(src)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(path)
        for ep, kind in kinds.items():
            mask = (df["episode_index"] == ep).to_numpy()
            if not mask.any():
                continue
            action = np.stack(df.loc[mask, "action"].to_numpy())
            state = np.stack(df.loc[mask, state_key].to_numpy()) if state_key in df else None
            action, state = _apply(action, state, kind, rng)
            dtype = np.asarray(df.loc[mask, "action"].iloc[0]).dtype
            df.loc[mask, "action"] = pd.Series(list(action.astype(dtype)), index=df.index[mask])
            if state is not None:
                sdtype = np.asarray(df.loc[mask, state_key].iloc[0]).dtype
                df.loc[mask, state_key] = pd.Series(list(state.astype(sdtype)), index=df.index[mask])
        df.to_parquet(out_path)

    truth = {"source": str(src), "rate": rate, "seed": seed, "corrupted": {str(k): v for k, v in sorted(kinds.items())}}
    (dst / "corruption.json").write_text(json.dumps(truth, indent=1))
    return truth


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--rate", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t = corrupt_dataset(a.src, a.dst, a.rate, a.seed)
    print(f"corrupted {len(t['corrupted'])} of the episodes; ground truth in {a.dst / 'corruption.json'}")
