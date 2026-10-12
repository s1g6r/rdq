"""Does filtering by rdq's verdict train better policies? A small, fully controlled test.

Task: a 2D position-controlled point robot must reach a random goal. An expert
records demonstrations in LeRobot v3.0 format; a fraction of episodes are
corrupted with realistic recording faults. We then train behaviour-cloning
policies three ways and evaluate each with many rollouts:

  raw      : all episodes
  filtered : only the episodes rdq marks usable
  random   : a random subset the same size as `filtered` (controls for the
             fact that filtering also shrinks the dataset)

If filtered beats random, the improvement comes from *which* episodes rdq
removes, not from having less data. This is a toy: it shows the mechanism and
exercises the full pipeline. The real test is SmolVLA on LIBERO (see
experiments/README.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq.checks import run_checks  # noqa: E402
from rdq.dataset import load_dataset  # noqa: E402
from rdq.scoring import score  # noqa: E402
from rdq.synthetic import write_dataset  # noqa: E402

FPS = 10
MAX_STEPS = 60
STEP = 0.06  # max distance the robot can move per frame
SUCCESS_RADIUS = 0.05
FAULTS = ("idle", "desync", "glitch", "wrong_goal")


# --------------------------------------------------------------------------- environment


def expert_episode(rng: np.random.Generator) -> dict:
    start = rng.uniform(-1, 1, 2)
    goal = rng.uniform(-1, 1, 2)
    pos = start.copy()
    actions, states = [], []
    curve = rng.normal(0, 0.4)  # human demos are not perfectly straight
    for _ in range(MAX_STEPS):
        states.append(pos.copy())
        delta = goal - pos
        dist = np.linalg.norm(delta)
        if dist < SUCCESS_RADIUS * 0.5:
            target = pos.copy()
        else:
            direction = delta / dist
            perp = np.array([-direction[1], direction[0]])
            direction = direction + curve * perp * min(1.0, dist)
            direction /= np.linalg.norm(direction)
            target = pos + direction * min(STEP, dist)
        actions.append(target)
        pos = target + rng.normal(0, 0.003, 2)  # position control with sensor noise
    n = len(actions)
    return {
        "action": np.array(actions),
        "state": np.array(states),
        "goal": np.repeat(goal[None], n, axis=0),
        "timestamp": np.arange(n) / FPS,
    }


def corrupt(ep: dict, kind: str, rng: np.random.Generator) -> None:
    n = len(ep["action"])
    if kind == "idle":  # operator never really moved; robot jitters in place
        ep["action"] = np.repeat(ep["state"][:1], n, axis=0) + rng.normal(0, 0.001, (n, 2))
        ep["state"] = ep["action"] + rng.normal(0, 0.003, (n, 2))
    elif kind == "desync":  # action stream recorded 8 frames late relative to observations
        k = 8
        ep["action"] = np.vstack([np.repeat(ep["action"][:1], k, axis=0), ep["action"][:-k]])
    elif kind == "glitch":  # teleop dropout: state jumps to garbage for a few frames
        t = rng.integers(5, n - 10)
        ep["state"] = ep["state"].copy()
        ep["state"][t : t + 3] += rng.uniform(1.5, 3.0, 2) * rng.choice([-1, 1], 2)
        ep["action"] = ep["action"].copy()
        ep["action"][t : t + 3] = ep["state"][t : t + 3]
    elif kind == "wrong_goal":  # failed attempt: robot went somewhere else (rdq cannot see this)
        other = rng.uniform(-1, 1, 2)
        ep["goal"] = np.repeat(other[None], n, axis=0)


def to_lerobot_episodes(episodes: list[dict]) -> list[dict]:
    out = []
    for ep in episodes:
        out.append(
            {
                "action": ep["action"],
                "state": ep["state"],
                "timestamp": ep["timestamp"],
                "task_index": 0,
                "goal": ep["goal"],
            }
        )
    return out


# --------------------------------------------------------------------------- tiny numpy MLP


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
            acts.append(x)
        return x, acts

    def step(self, x, y, lr=1e-3):
        pred, acts = self.forward(x)
        grad = 2 * (pred - y) / len(x)
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.insert(0, acts[i].T @ grad)
            gb.insert(0, grad.sum(0))
            if i:
                grad = (grad @ self.W[i].T) * (acts[i] > 0)
        self.t += 1
        params = self.W + self.b
        grads = gW + gb
        for j, (p, g) in enumerate(zip(params, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * g
            self.v[j] = 0.999 * self.v[j] + 0.001 * g * g
            mh = self.m[j] / (1 - 0.9**self.t)
            vh = self.v[j] / (1 - 0.999**self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean((pred - y) ** 2))


def train_policy(episodes: list[dict], seed: int, epochs: int = 40) -> tuple[MLP, dict]:
    rng = np.random.default_rng(seed)
    X = np.concatenate([np.hstack([e["state"], e["goal"]]) for e in episodes])
    # Predict the commanded displacement, the usual relative-action trick.
    Y = np.concatenate([e["action"] - e["state"] for e in episodes])
    ok = np.all(np.isfinite(X), 1) & np.all(np.isfinite(Y), 1)
    X, Y = X[ok], Y[ok]
    norm = {"x_mean": X.mean(0), "x_std": X.std(0) + 1e-6, "y_scale": STEP}
    Xn, Yn = (X - norm["x_mean"]) / norm["x_std"], Y / norm["y_scale"]
    net = MLP([4, 128, 128, 2], rng)
    for _ in range(epochs):
        order = rng.permutation(len(Xn))
        for i in range(0, len(order), 256):
            idx = order[i : i + 256]
            net.step(Xn[idx], Yn[idx])
    return net, norm


def evaluate(net: MLP, norm: dict, n_rollouts: int, seed: int) -> float:
    rng = np.random.default_rng(10_000 + seed)
    starts = rng.uniform(-1, 1, (n_rollouts, 2))
    goals = rng.uniform(-1, 1, (n_rollouts, 2))
    pos = starts.copy()
    done = np.zeros(n_rollouts, bool)
    for _ in range(MAX_STEPS):
        x = (np.hstack([pos, goals]) - norm["x_mean"]) / norm["x_std"]
        delta = net.forward(x)[0] * norm["y_scale"]
        length = np.linalg.norm(delta, axis=1, keepdims=True)
        delta = np.where(length > STEP, delta / np.maximum(length, 1e-9) * STEP, delta)
        pos = np.where(done[:, None], pos, pos + delta + rng.normal(0, 0.003, pos.shape))
        done |= np.linalg.norm(pos - goals, axis=1) < SUCCESS_RADIUS
    return float(done.mean())


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# --------------------------------------------------------------------------- experiment


def run(n_episodes: int, rates: list[float], seeds: int, rollouts: int, out: Path) -> dict:
    results = []
    for rate in rates:
        for seed in range(seeds):
            rng = np.random.default_rng(seed * 101 + int(rate * 1000))
            eps = [expert_episode(rng) for _ in range(n_episodes)]
            n_bad = int(round(rate * n_episodes))
            bad = rng.choice(n_episodes, n_bad, replace=False)
            kinds = {}
            for j, i in enumerate(bad):
                kinds[int(i)] = FAULTS[j % len(FAULTS)]
                corrupt(eps[i], kinds[int(i)], rng)

            with tempfile.TemporaryDirectory() as tmp:
                root = write_dataset(tmp, to_lerobot_episodes(eps), fps=FPS, names=["x", "y"], with_video=False)
                ds = load_dataset(root)
                findings = run_checks(ds)
                sc = score(findings, sorted(ds.episodes))
            usable = set(sc.usable_episodes)
            caught = {k: sum(1 for i, kk in kinds.items() if kk == k and i not in usable) for k in FAULTS}
            planted = {k: sum(1 for kk in kinds.values() if kk == k) for k in FAULTS}
            clean_dropped = sum(1 for i in range(n_episodes) if i not in kinds and i not in usable)

            subsets = {
                "raw": list(range(n_episodes)),
                "filtered": sorted(usable),
                "random": sorted(rng.choice(n_episodes, len(usable), replace=False).tolist()),
            }
            row = {
                "corruption_rate": rate,
                "seed": seed,
                "dataset_score": sc.dataset_score,
                "usable": len(usable),
                "planted": planted,
                "caught": caught,
                "clean_dropped": clean_dropped,
                "n_episodes": n_episodes,
            }
            for name, idx in subsets.items():
                net, norm = train_policy([eps[i] for i in idx], seed=seed)
                row[f"success_{name}"] = evaluate(net, norm, rollouts, seed)
            results.append(row)
            print(
                f"rate {rate:.0%} seed {seed}: score {sc.dataset_score:3d} usable {len(usable):3d} | "
                f"raw {row['success_raw']:.2f} filtered {row['success_filtered']:.2f} "
                f"random {row['success_random']:.2f} | caught {caught} clean dropped {clean_dropped}",
                flush=True,
            )
    summary = summarise(results, rollouts)
    out.mkdir(parents=True, exist_ok=True)
    (out / "toy_results.json").write_text(json.dumps({"rows": results, "summary": summary}, indent=1))
    return summary


def summarise(rows: list[dict], rollouts: int) -> dict:
    by_rate = {}
    for r in rows:
        by_rate.setdefault(r["corruption_rate"], []).append(r)
    table = []
    for rate, rs in sorted(by_rate.items()):
        entry = {"corruption_rate": rate, "seeds": len(rs)}
        for name in ("raw", "filtered", "random"):
            vals = np.array([r[f"success_{name}"] for r in rs])
            entry[name] = round(float(vals.mean()), 3)
            entry[f"{name}_ci"] = [round(x, 3) for x in wilson(vals.mean(), rollouts * len(rs))]
        entry["dataset_score"] = round(float(np.mean([r["dataset_score"] for r in rs])), 1)
        diffs = np.array([r["success_filtered"] - r["success_random"] for r in rs])
        entry["filtered_minus_random"] = round(float(diffs.mean()), 3)
        table.append(entry)
    planted = {k: sum(r["planted"][k] for r in rows) for k in FAULTS}
    caught = {k: sum(r["caught"][k] for r in rows) for k in FAULTS}
    clean_dropped = sum(r["clean_dropped"] for r in rows)
    clean_total = sum(r["n_episodes"] - sum(r["planted"].values()) for r in rows)
    scores = np.array([r["dataset_score"] for r in rows])
    raw = np.array([r["success_raw"] for r in rows])
    corr = float(np.corrcoef(scores, raw)[0, 1]) if scores.std() > 0 and raw.std() > 0 else float("nan")
    return {
        "by_rate": table,
        "detection": {k: f"{caught[k]}/{planted[k]}" for k in FAULTS},
        "clean_episodes_wrongly_dropped": f"{clean_dropped}/{clean_total}",
        "corr_dataset_score_vs_raw_success": round(corr, 3),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=400)
    p.add_argument("--out", default="experiments/results")
    a = p.parse_args()
    s = run(a.episodes, a.rates, a.seeds, a.rollouts, Path(a.out))
    print(json.dumps(s, indent=1))
