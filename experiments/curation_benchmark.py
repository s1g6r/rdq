"""Curation benchmark: which episode-selection method trains the best policy?

A controlled, CPU-only benchmark that goes beyond the first toy experiment in three ways:

1. **Natural skill variation, not only injected faults.** Besides recording faults that
   rdq's rules are designed to catch (frozen log, 8-frame desync, teleop glitch), the
   data contains demonstrations from imperfect operators: shaky hands, hesitation with
   pauses and back-and-forth, detours, and failed attempts that stop short. The rules
   were not written for these, so this tests the literature-style scores honestly.
2. **The baselines reviewers expect** (see research notes R2): all data, an oracle that
   drops exactly the bad episodes, random subsets, keep-the-shortest (length is a strong
   confound), the published spectral (PSD) metric, DemInf-lite, kNN consistency,
   smoothness, a loss-based filter, rdq's rules, and rules followed by the composite.
3. **Detection metrics** (AUROC of each score against the ground-truth labels) as well as
   policy success, so it is clear what each method actually sees.

Every subset-selecting method keeps the same number of episodes (the oracle's count)
unless it decides its own size (rules), in which case a same-size random subset is
reported next to it.

    python experiments/curation_benchmark.py --episodes 200 --seeds 3 --rollouts 400
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from toy_reach import MLP, wilson  # noqa: E402

from rdq.checks import run_checks  # noqa: E402
from rdq.dataset import load_dataset  # noqa: E402
from rdq.scores import score_episodes  # noqa: E402
from rdq.scoring import score  # noqa: E402
from rdq.synthetic import write_dataset  # noqa: E402

FPS = 30  # realistic control rate; episodes are 2-6 s long, as in typical LeRobot data
F = FPS / 10  # time scale relative to a 10 Hz design
STEP = 0.06 / F
RADIUS = 0.05
EVAL_STEPS = int(80 * F)
STRUCTURAL = ("frozen_log", "desync", "glitch")
SKILL = ("shaky", "hesitant", "detour", "failed")


# --------------------------------------------------------------------------- operators


def demo(rng: np.random.Generator, style: str = "expert") -> dict:
    start, goal = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
    while np.linalg.norm(goal - start) < 0.4:
        goal = rng.uniform(-1, 1, 2)
    pos = start.copy()
    curve = rng.normal(0, 0.3)
    waypoint = rng.uniform(-1, 1, 2) if style == "detour" else None
    stop_at = rng.uniform(0.35, 0.7) if style == "failed" else None
    total = np.linalg.norm(goal - start)
    pause_at = rng.uniform(0.3, 0.6) if style == "hesitant" else None
    paused = 0
    actions, states = [], []
    reached_at = None
    for t in range(int(200 * F)):
        states.append(pos.copy())
        target_pt = goal
        if waypoint is not None and np.linalg.norm(pos - waypoint) > 0.1 and t < 40 * F:
            target_pt = waypoint
        progress = 1 - np.linalg.norm(goal - pos) / total
        delta = target_pt - pos
        dist = np.linalg.norm(delta)
        if stop_at is not None and progress >= stop_at:
            target = pos.copy()  # gave up: stays put
        elif pause_at is not None and progress >= pause_at and paused < 25 * F:
            # hesitation: pause, then wiggle back and forth without progress
            paused += 1
            wiggle = np.array([np.sin(paused / F), np.cos(paused * 1.3 / F)]) * 0.02 / F if paused > 8 * F else 0.0
            target = pos + wiggle
        elif dist < RADIUS * 0.5:
            target = pos.copy()
        else:
            d = delta / dist
            perp = np.array([-d[1], d[0]])
            d = d + curve * perp * min(1.0, dist)
            d /= np.linalg.norm(d)
            target = pos + d * min(STEP, dist)
        if style == "shaky":
            target = target + rng.normal(0, 0.025 / np.sqrt(F), 2)
        actions.append(target)
        pos = target + rng.normal(0, 0.003, 2)
        if reached_at is None and np.linalg.norm(pos - goal) < RADIUS:
            reached_at = t
        # operators stop recording a few frames after finishing (or after giving up)
        if reached_at is not None and t >= reached_at + int(rng.integers(2, 6) * F):
            break
        if stop_at is not None and progress >= stop_at and t > 15 * F and rng.random() < 0.08 / F:
            break
    n = len(actions)
    return {
        "action": np.array(actions),
        "state": np.array(states),
        "goal": np.repeat(goal[None], n, axis=0),
        "timestamp": np.arange(n) / FPS,
    }


def structural_fault(ep: dict, kind: str, rng: np.random.Generator) -> None:
    n = len(ep["action"])
    if kind == "frozen_log":
        ep["action"] = np.repeat(ep["state"][:1], n, axis=0) + rng.normal(0, 0.001, (n, 2))
        ep["state"] = ep["action"] + rng.normal(0, 0.003, (n, 2))
    elif kind == "desync":
        k = 8
        ep["action"] = np.vstack([np.repeat(ep["action"][:1], k, axis=0), ep["action"][:-k]])
    elif kind == "glitch":
        t = int(rng.integers(3, max(4, n - 5)))
        ep["state"] = ep["state"].copy()
        ep["state"][t : t + 3] += rng.uniform(1.5, 3.0, 2) * rng.choice([-1, 1], 2)
        ep["action"] = ep["action"].copy()
        ep["action"][t : t + 3] = ep["state"][t : t + 3]


def make_dataset(n: int, scenario: str, rng: np.random.Generator):
    labels = ["expert"] * n
    if scenario in ("skill", "mixed"):
        k = int(round((0.4 if scenario == "skill" else 0.2) * n))
        for j, i in enumerate(rng.choice(n, k, replace=False)):
            labels[i] = SKILL[j % len(SKILL)]
    eps = [demo(rng, lab) for lab in labels]
    if scenario in ("structural", "mixed"):
        free = [i for i in range(n) if labels[i] == "expert"]
        k = int(round((0.3 if scenario == "structural" else 0.15) * n))
        for j, i in enumerate(rng.choice(free, k, replace=False)):
            labels[i] = STRUCTURAL[j % len(STRUCTURAL)]
            structural_fault(eps[i], labels[i], rng)
    return eps, labels


# --------------------------------------------------------------------------- policy


def train(eps: list[dict], seed: int, epochs: int = 40, return_losses: bool = False):
    rng = np.random.default_rng(seed)
    X = np.concatenate([np.hstack([e["state"], e["goal"]]) for e in eps])
    Y = np.concatenate([e["action"] - e["state"] for e in eps])
    owner = np.concatenate([np.full(len(e["action"]), j) for j, e in enumerate(eps)])
    ok = np.all(np.isfinite(X), 1) & np.all(np.isfinite(Y), 1)
    X, Y, owner = X[ok], Y[ok], owner[ok]
    norm = {"x_mean": X.mean(0), "x_std": X.std(0) + 1e-6, "y_scale": STEP}
    Xn, Yn = (X - norm["x_mean"]) / norm["x_std"], Y / norm["y_scale"]
    net = MLP([4, 128, 128, 2], rng)
    for _ in range(epochs):
        order = rng.permutation(len(Xn))
        for i in range(0, len(order), 256):
            idx = order[i : i + 256]
            net.step(Xn[idx], Yn[idx])
    if return_losses:
        err = ((net.forward(Xn)[0] - Yn) ** 2).mean(1)
        return np.array([err[owner == j].mean() if (owner == j).any() else np.inf for j in range(len(eps))])
    return net, norm


def evaluate(net, norm, n: int, seed: int) -> float:
    rng = np.random.default_rng(50_000 + seed)
    starts, goals = rng.uniform(-1, 1, (n, 2)), rng.uniform(-1, 1, (n, 2))
    far = np.linalg.norm(goals - starts, axis=1) < 0.4
    goals[far] = -starts[far]
    pos, done = starts.copy(), np.zeros(n, bool)
    for _ in range(EVAL_STEPS):
        x = (np.hstack([pos, goals]) - norm["x_mean"]) / norm["x_std"]
        d = net.forward(x)[0] * norm["y_scale"]
        L = np.linalg.norm(d, axis=1, keepdims=True)
        d = np.where(L > STEP, d / np.maximum(L, 1e-9) * STEP, d)
        pos = np.where(done[:, None], pos, pos + d + rng.normal(0, 0.003, pos.shape))
        done |= np.linalg.norm(pos - goals, axis=1) < RADIUS
    return float(done.mean())


# --------------------------------------------------------------------------- selection


def auroc(bad: np.ndarray, score: np.ndarray) -> float:
    """AUROC of `score` (higher = more suspicious) for the bad label; NaNs rank as least suspicious."""
    s = np.where(np.isfinite(score), score, -np.inf)
    pos, neg = s[bad], s[~bad]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(n_eps: int, seeds: int, rollouts: int, scenarios: list[str], out: Path) -> dict:
    rows = []
    for scenario in scenarios:
        for seed in range(seeds):
            t0 = time.time()
            rng = np.random.default_rng(1000 * seed + {"structural": 1, "skill": 2, "mixed": 3}.get(scenario, 0))
            eps, labels = make_dataset(n_eps, scenario, rng)
            bad = np.array([lab != "expert" for lab in labels])
            with tempfile.TemporaryDirectory() as tmp:
                lr = [{"action": e["action"], "state": e["state"], "timestamp": e["timestamp"], "task_index": 0,
                       "extra": {"observation.environment_state": e["goal"]}} for e in eps]
                root = write_dataset(tmp, lr, fps=FPS, names=["x", "y"], with_video=False)
                ds = load_dataset(root)
                findings = run_checks(ds)
                sc = score(findings, sorted(ds.episodes))
                t1 = time.time()
                S = score_episodes(ds)["episodes"]
                score_time = time.time() - t1
            usable = set(sc.usable_episodes)

            def col(name):
                return np.array([S[i][name] if S[i][name] is not None else np.nan for i in range(n_eps)], float)

            losses = train(eps, seed + 7, epochs=5, return_losses=True)
            lengths = np.array([len(e["action"]) for e in eps], float)
            suspicion = {  # higher = drop first
                "length (longest first)": lengths,
                "psd (Sojib et al.)": col("psd_power"),
                "deminf-lite": -col("deminf"),
                "consistency": col("consistency"),
                "hfr": col("hfr"),
                "dither": col("dither"),
                "final-state outlier": col("final_outlier"),
                "composite": -col("composite"),
                "loss-based": losses,
                "rules (rdq check)": np.array([0.0 if i in usable else 1.0 for i in range(n_eps)]),
            }
            keep_n = int((~bad).sum())

            def keep_lowest(s):
                s = np.where(np.isfinite(s), s, np.inf)
                return sorted(np.argsort(s, kind="stable")[:keep_n].tolist())

            subsets = {
                "all data": list(range(n_eps)),
                "oracle": sorted(np.where(~bad)[0].tolist()),
                "random": sorted(rng.choice(n_eps, keep_n, replace=False).tolist()),
                "keep shortest": keep_lowest(lengths),
                "psd (Sojib et al.)": keep_lowest(suspicion["psd (Sojib et al.)"]),
                "deminf-lite": keep_lowest(suspicion["deminf-lite"]),
                "consistency": keep_lowest(suspicion["consistency"]),
                "hfr": keep_lowest(suspicion["hfr"]),
                "composite": keep_lowest(suspicion["composite"]),
                "loss-based": keep_lowest(losses),
                "rules (rdq check)": sorted(usable),
                "random, rules size": sorted(rng.choice(n_eps, len(usable), replace=False).tolist()),
            }
            # rules first, then fill/trim to keep_n by composite among the rule-passing episodes
            comp = suspicion["composite"]
            ranked_usable = sorted(usable, key=lambda i: (comp[i] if np.isfinite(comp[i]) else np.inf))
            subsets["rules + composite"] = sorted(ranked_usable[:keep_n])

            row = {"scenario": scenario, "seed": seed, "n": n_eps, "keep_n": keep_n, "rules_kept": len(usable),
                   "score_seconds": round(score_time, 2), "labels": {k: labels.count(k) for k in set(labels)}}
            row["auroc"] = {k: round(auroc(bad, v), 3) for k, v in suspicion.items()}
            row["auroc_by_type"] = {
                k: {t: round(auroc(np.array([lab == t for lab in labels])[np.array([lab in (t, "expert") for lab in labels])],
                                   v[np.array([lab in (t, "expert") for lab in labels])]), 3)
                    for t in sorted(set(labels) - {"expert"})}
                for k, v in suspicion.items()
            }
            row["kept_bad_fraction"] = {k: round(float(bad[idx].mean()), 3) if idx else None for k, idx in subsets.items()}
            row["success"] = {}
            for name, idx in subsets.items():
                net, norm = train([eps[i] for i in idx], seed)
                row["success"][name] = evaluate(net, norm, rollouts, seed)
            rows.append(row)
            print(f"[{scenario} seed {seed}] {time.time() - t0:.0f}s  " +
                  "  ".join(f"{k}={v:.2f}" for k, v in row["success"].items()), flush=True)
    summary = summarise(rows, rollouts)
    out.mkdir(parents=True, exist_ok=True)
    (out / "curation_benchmark.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    (out / "curation_benchmark.md").write_text(to_markdown(summary))
    return summary


def summarise(rows: list[dict], rollouts: int) -> dict:
    out = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        rs = [r for r in rows if r["scenario"] == scenario]
        methods = list(rs[0]["success"])
        table = {}
        for m in methods:
            v = np.array([r["success"][m] for r in rs])
            lo, hi = wilson(v.mean(), rollouts * len(rs))
            kept_bad = [r["kept_bad_fraction"][m] for r in rs if r["kept_bad_fraction"][m] is not None]
            table[m] = {"success": round(float(v.mean()), 3), "ci95": [round(lo, 3), round(hi, 3)],
                        "per_seed": [round(x, 3) for x in v], "kept_bad_fraction": round(float(np.mean(kept_bad)), 3)}
        aur = {k: round(float(np.nanmean([r["auroc"][k] for r in rs])), 3) for k in rs[0]["auroc"]}
        types = sorted({t for r in rs for t in r["auroc_by_type"]["composite"]})
        by_type = {k: {t: round(float(np.nanmean([r["auroc_by_type"][k].get(t, np.nan) for r in rs])), 3) for t in types}
                   for k in rs[0]["auroc_by_type"]}
        out[scenario] = {"seeds": len(rs), "episodes": rs[0]["n"], "keep_n": rs[0]["keep_n"],
                         "rules_kept_mean": round(float(np.mean([r["rules_kept"] for r in rs])), 1),
                         "score_seconds_mean": round(float(np.mean([r["score_seconds"] for r in rs])), 2),
                         "success": table, "auroc": aur, "auroc_by_type": by_type}
    return out


def to_markdown(summary: dict) -> str:
    L = ["# Curation benchmark results", ""]
    for sc, s in summary.items():
        L += [f"## Scenario: {sc}", "",
              f"{s['episodes']} episodes, {s['seeds']} seeds; subset methods keep {s['keep_n']} episodes (the number of good ones). "
              f"rdq's rules kept {s['rules_kept_mean']} on average. Scoring took {s['score_seconds_mean']} s per dataset.", "",
              "| Method | Success | 95% CI | Bad episodes kept |", "|---|---|---|---|"]
        for m, v in sorted(s["success"].items(), key=lambda kv: -kv[1]["success"]):
            L.append(f"| {m} | {v['success']:.3f} | {v['ci95'][0]:.3f}–{v['ci95'][1]:.3f} | {v['kept_bad_fraction']:.0%} |")
        L += ["", "Detection AUROC (bad vs expert):", "", "| Score | All | " + " | ".join(next(iter(s["auroc_by_type"].values()))) + " |",
              "|---|---|" + "---|" * len(next(iter(s["auroc_by_type"].values())))]
        for k, v in s["auroc"].items():
            L.append(f"| {k} | {v:.2f} | " + " | ".join(f"{x:.2f}" for x in s["auroc_by_type"][k].values()) + " |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=400)
    p.add_argument("--scenarios", nargs="+", default=["structural", "skill", "mixed"])
    p.add_argument("--out", default="experiments/results")
    a = p.parse_args()
    s = run(a.episodes, a.seeds, a.rollouts, a.scenarios, Path(a.out))
    print(to_markdown(s))
