"""Literature-based per-episode scores."""

import numpy as np

from rdq import synthetic as syn
from rdq.cli import main
from rdq.dataset import load_dataset
from rdq.scores import keep_ranges, score_episodes


def _reach(rng, shaky=False, n_extra_idle=0):
    start, goal = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
    pos, acts, states = start.copy(), [], []
    for _ in range(80):
        states.append(pos.copy())
        d = goal - pos
        dist = np.linalg.norm(d)
        tgt = pos + (d / dist) * min(0.05, dist) if dist > 0.02 else pos.copy()
        if shaky:
            tgt = tgt + rng.normal(0, 0.03, 2)
        acts.append(tgt)
        pos = tgt + rng.normal(0, 0.002, 2)
        if dist < 0.02:
            break
    a, s = np.array(acts), np.array(states)
    if n_extra_idle:
        a = np.vstack([a, np.repeat(a[-1:], n_extra_idle, 0)])
        s = np.vstack([s, np.repeat(s[-1:], n_extra_idle, 0)])
    g = np.repeat(goal[None], len(a), 0)
    return {"action": a, "state": s, "timestamp": np.arange(len(a)) / 10, "task_index": 0,
            "extra": {"observation.environment_state": g}}


def _dataset(tmp_path, n=60, shaky=(), dup_of=None):
    rng = np.random.default_rng(0)
    eps = [_reach(rng, shaky=i in shaky) for i in range(n)]
    if dup_of is not None:
        e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in eps[dup_of].items()}
        e["extra"] = {k: v.copy() for k, v in eps[dup_of]["extra"].items()}
        e["action"] = e["action"] + rng.normal(0, 1e-4, e["action"].shape)
        eps.append(e)
    return load_dataset(syn.write_dataset(tmp_path / "ds", eps, fps=10, names=["x", "y"], with_video=False))


def test_scores_rank_shaky_episodes_low(tmp_path):
    shaky = set(range(0, 60, 10))
    res = score_episodes(_dataset(tmp_path, shaky=shaky))
    eps = res["episodes"]
    assert res["summary"]["relative_actions"] is True
    assert "observation.environment_state" in res["summary"]["context_features"]
    hfr = {i: e["hfr"] for i, e in eps.items() if e["hfr"] is not None and np.isfinite(e["hfr"])}
    assert np.median([hfr[i] for i in shaky if i in hfr]) > np.median([hfr[i] for i in hfr if i not in shaky])
    comp = {i: e["composite"] for i, e in eps.items() if e["composite"] is not None}
    assert np.mean([comp[i] for i in shaky if i in comp]) < np.mean([comp[i] for i in comp if i not in shaky])


def test_near_duplicate_found(tmp_path):
    res = score_episodes(_dataset(tmp_path, dup_of=5))
    dups = {i: e["near_dup_of"] for i, e in res["episodes"].items() if e["near_dup_of"] is not None}
    assert dups == {60: 5}


def test_keep_ranges_trim_idle_runs():
    idle = np.zeros(100, bool)
    idle[40:60] = True  # one long pause
    ranges = keep_ranges(idle, fps=15)
    assert ranges[0][0] == 0 and ranges[0][1] <= 40 - 10 + 1
    assert ranges[-1] == (60, 100)


def test_rank_cli(tmp_path, capsys):
    _dataset(tmp_path, shaky={3})
    out = tmp_path / "keep.json"
    assert main(["rank", str(tmp_path / "ds"), "--keep-top", "0.9", "--filter", str(out), "--json", str(tmp_path / "s.json")]) == 0
    import json

    kept = json.loads(out.read_text())["episodes"]
    scores = json.loads((tmp_path / "s.json").read_text())  # strict JSON: no NaN
    scored = [i for i, e in scores["episodes"].items() if e["composite"] is not None and not e["rule_flags"]]
    assert len(kept) == round(0.9 * len(scored))
    assert not any(scores["episodes"][str(i)]["rule_flags"] for i in kept)
    assert "ranked by composite" in capsys.readouterr().out


def test_scorecard_by_collector(tmp_path, capsys):
    import json

    from rdq import synthetic as syn2

    eps = syn2.make_episodes(16)
    for i in (1, 3, 5, 7):  # collector "b" records frozen episodes
        syn2.make_idle(eps[i])
    syn2.write_dataset(tmp_path / "ds", eps)
    (tmp_path / "groups.csv").write_text("episode_index,group\n" + "".join(f"{i},{'b' if i % 2 else 'a'}\n" for i in range(16)))
    assert main(["scorecard", str(tmp_path / "ds"), "--groups", str(tmp_path / "groups.csv"), "--json", str(tmp_path / "s.json")]) == 0
    rows = {r["group"]: r for r in json.loads((tmp_path / "s.json").read_text())}
    assert rows["a"]["usable_fraction"] == 1.0
    assert rows["b"]["usable"] <= 4
    assert "group" in capsys.readouterr().out
