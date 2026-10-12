"""Natural-noise benchmark on RoboMimic multi-human (MH) data: no injected faults.

RoboMimic's MH datasets contain 300 demonstrations per task from six operators,
two each labelled "better", "okay" and "worse" (Mandlekar et al., 2021). The labels
are ground truth for demonstration quality that nobody injected, which is exactly what
a curation method should recover. Following Sojib et al. (2026, arXiv 2605.01544,
Table I) we keep 50% of the demonstrations with each method and report the mean
operator tier of what is kept (better = 3, okay = 2, worse = 1). Random keeps ~2.0,
the oracle keeps 2.67.

Methods compared, all training-free:
  random, keep-shortest (length), PSD (Sojib et al., replicated on end-effector position),
  rdq DemInf-lite, rdq consistency, rdq HFR / LDLJ smoothness, rdq composite,
  rdq rules (structural checks; not designed for skill, reported for completeness).

    pip install h5py huggingface_hub
    python experiments/robomimic_mh.py --tasks can square --out experiments/results

Runs on a laptop CPU in a few minutes; downloads ~113 MB per task from the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq.checks import run_checks  # noqa: E402
from rdq.dataset import load_dataset  # noqa: E402
from rdq.scores import score_episodes  # noqa: E402
from rdq.scoring import score  # noqa: E402
from rdq.synthetic import write_dataset  # noqa: E402

FPS = 20  # RoboMimic control rate
TIER = {"better": 3, "okay": 2, "worse": 1}
# Published kept-set operator tier at 50% on Can-MH (Sojib et al. 2026, Table I).
PUBLISHED_CAN = {"random": 1.99, "DemInf (Hejna et al.)": 2.37, "CUPID (Agia et al.)": 2.32, "PSD (Sojib et al.)": 2.53, "oracle": 2.67}


def fetch(task: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("robomimic/robomimic_datasets", f"v1.5/{task}/mh/low_dim_v15.hdf5",
                                repo_type="dataset", cache_dir=str(cache)))


def load_mh(path: Path):
    import h5py

    demos, tiers = [], []
    with h5py.File(path, "r") as f:
        masks = {k: {d.decode() if isinstance(d, bytes) else str(d) for d in f["mask"][k][()]} for k in f["mask"].keys()}
        tier_of = {}
        for name, members in masks.items():
            if name in TIER:
                for d in members:
                    tier_of[d] = TIER[name]
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for d in names:
            if d not in tier_of:
                continue
            g = f["data"][d]
            obs = g["obs"]
            eef = obs["robot0_eef_pos"][()]
            state = np.hstack([eef, obs["robot0_eef_quat"][()], obs["robot0_gripper_qpos"][()]])
            demos.append({"action": g["actions"][()], "state": state, "object": obs["object"][()], "eef": eef})
            tiers.append(tier_of[d])
    return demos, np.array(tiers), sorted(masks)


def psd_power(eef: np.ndarray) -> float:
    """Sojib et al.: total spectral power of the 3D end-effector trajectory (no detrending)."""
    return float(sum((np.abs(np.fft.fft(eef[:, d])) ** 2).sum() for d in range(eef.shape[1])))


def auroc(bad: np.ndarray, s: np.ndarray) -> float:
    s = np.where(np.isfinite(s), s, -np.inf)
    pos, neg = s[bad], s[~bad]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def run_task(task: str, cache: Path) -> dict:
    t0 = time.time()
    path = fetch(task, cache)
    demos, tiers, mask_names = load_mh(path)
    n = len(demos)
    lengths = np.array([len(d["action"]) for d in demos], float)
    with tempfile.TemporaryDirectory() as tmp:
        eps = [{"action": d["action"], "state": d["state"], "timestamp": np.arange(len(d["action"])) / FPS, "task_index": 0,
                "extra": {"observation.environment_state": d["object"]}} for d in demos]
        names = ["eef_x", "eef_y", "eef_z", "qx", "qy", "qz", "qw", "gripper_l", "gripper_r"]
        ds = load_dataset(write_dataset(tmp, eps, fps=FPS, names=names, with_video=False, robot_type="panda"))
        # actions are 7-D OSC deltas; give them their own names
        findings = run_checks(ds)
        sc = score(findings, sorted(ds.episodes))
        t1 = time.time()
        S = score_episodes(ds)
        score_time = time.time() - t1
    E = S["episodes"]

    def col(k):
        return np.array([E[i][k] if E[i][k] is not None else np.nan for i in range(n)], float)

    quality = {  # higher = better demo
        "keep shortest (length)": -lengths,
        "PSD (Sojib et al., replicated)": -np.array([psd_power(d["eef"]) for d in demos]),
        "rdq deminf-lite": col("deminf"),
        "rdq consistency": -col("consistency"),
        "rdq hfr": -col("hfr"),
        "rdq ldlj": col("ldlj"),
        "rdq dither": -col("dither"),
        "rdq composite": col("composite"),
    }
    usable = set(sc.usable_episodes)
    keep = n // 2
    rng = np.random.default_rng(0)
    kept_tier = {"random (mean of 1000 draws)": float(np.mean([tiers[rng.choice(n, keep, replace=False)].mean() for _ in range(1000)])),
                 "oracle": float(np.sort(tiers)[::-1][:keep].mean())}
    for name, q in quality.items():
        qq = np.where(np.isfinite(q), q, -np.inf)
        kept_tier[name] = float(tiers[np.argsort(-qq, kind="stable")[:keep]].mean())
    worse, better = tiers == 1, tiers == 3
    sel = worse | better
    res = {
        "task": task,
        "demos": n,
        "tiers": {k: int((tiers == v).sum()) for k, v in TIER.items()},
        "mask_names": mask_names,
        "median_length_by_tier": {k: float(np.median(lengths[tiers == v])) for k, v in TIER.items()},
        "kept_fraction": 0.5,
        "kept_set_mean_tier": {k: round(v, 3) for k, v in kept_tier.items()},
        "auroc_worse_vs_better": {k: round(auroc(worse[sel], -q[sel]), 3) for k, q in quality.items()},
        "spearman_with_tier": {k: round(spearman(q, tiers.astype(float)), 3) for k, q in quality.items()},
        "spearman_with_length": {k: round(spearman(q, -lengths), 3) for k, q in quality.items()},
        "rules": {"usable": len(usable), "findings": sorted({(f.check, f.severity, len(f.episodes)) for f in findings})},
        "rules_kept_mean_tier": round(float(tiers[sorted(usable)].mean()), 3) if usable else None,
        "score_summary": S["summary"],
        "score_seconds": round(score_time, 2),
        "seconds": round(time.time() - t0, 1),
    }
    if task == "can":
        res["published_kept_set_mean_tier_can"] = PUBLISHED_CAN
    return res


def to_markdown(results: list[dict]) -> str:
    L = ["# RoboMimic MH natural-noise benchmark", "",
         "Keep 50% of demonstrations with each method; report the mean operator tier kept (better = 3, okay = 2, worse = 1).", ""]
    for r in results:
        L += [f"## {r['task']} ({r['demos']} demos; median length by tier: " +
              ", ".join(f"{k} {v:.0f}" for k, v in r["median_length_by_tier"].items()) + ")", "",
              "| Method | Kept-set mean tier | AUROC worse vs better | Spearman with tier | Spearman with length |", "|---|---|---|---|---|"]
        for k, v in sorted(r["kept_set_mean_tier"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {k} | {v:.2f} | {r['auroc_worse_vs_better'].get(k, float('nan')):.2f} | "
                     f"{r['spearman_with_tier'].get(k, float('nan')):.2f} | {r['spearman_with_length'].get(k, float('nan')):.2f} |")
        if r.get("published_kept_set_mean_tier_can"):
            L += ["", "Published on Can-MH at 50% (Sojib et al. 2026, Table I): " +
                  ", ".join(f"{k} {v:.2f}" for k, v in r["published_kept_set_mean_tier_can"].items())]
        L += ["", f"rdq rules: {r['rules']['usable']}/{r['demos']} demos pass (kept-set tier {r['rules_kept_mean_tier']}); "
              f"scoring took {r['score_seconds']} s.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["can", "square"])
    p.add_argument("--cache", default=str(Path.home() / ".cache" / "rdq_robomimic"))
    p.add_argument("--out", default="experiments/results")
    a = p.parse_args()
    results = [run_task(t, Path(a.cache)) for t in a.tasks]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robomimic_mh.json").write_text(json.dumps(results, indent=1, default=str))
    (out / "robomimic_mh.md").write_text(to_markdown(results))
    print(to_markdown(results))
