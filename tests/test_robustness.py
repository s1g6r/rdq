"""Inputs real Hub datasets contain that the happy-path tests don't (pre-release review R7)."""

import json

import numpy as np
import pandas as pd

from rdq import synthetic as syn
from rdq.dataset import load_dataset
from rdq.report import audit, format_terminal, render_html
from rdq.scores import score_episodes


def _write(tmp_path, eps=None, **kw):
    root = tmp_path / "ds"
    syn.write_dataset(root, eps if eps is not None else syn.make_episodes(12, 120), with_video=False, **kw)
    return root


def _edit_info(root, fn):
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    fn(info)
    p.write_text(json.dumps(info))


def _edit_data(root, fn):
    for p in sorted((root / "data").glob("**/*.parquet")):
        df = pd.read_parquet(p)
        df = fn(df)
        df.to_parquet(p)


def _checks(result):
    return {f["check"] for f in result["findings"]}


def test_depth_and_point_cloud_features_are_skipped_not_loaded(tmp_path):
    root = _write(tmp_path)

    def add(df):
        df["observation.images.depth"] = [{"bytes": b"x", "path": None}] * len(df)
        df["observation.points"] = [np.zeros((5 + (i % 3)) * 3, np.float32) for i in range(len(df))]
        df["observation.depths.middle"] = [np.zeros(48 * 64, np.uint16) for _ in range(len(df))]
        return df

    _edit_data(root, add)

    def declare(info):
        info["features"]["observation.images.depth"] = {"dtype": "depth", "shape": [48, 64, 1]}
        info["features"]["observation.points"] = {"dtype": "pcd", "shape": [-1, 3]}
        info["features"]["observation.depths.middle"] = {"dtype": "uint16", "shape": [48, 64]}
        info["features"]["observation.depth_ffv1.head"] = {"dtype": "depth_video", "shape": [48, 64, 1]}
        info["features"]["observation.audio"] = {"dtype": "audio", "shape": [1]}

    _edit_info(root, declare)
    ds = load_dataset(root)
    assert set(ds.skipped_features) >= {"observation.images.depth", "observation.points", "observation.depths.middle",
                                       "observation.depth_ffv1.head", "observation.audio"}
    assert not any(k.startswith(("observation.points", "observation.depth")) for k in ds.numeric_keys)
    r = audit(root)
    sev = {f["check"]: f["severity"] for f in r["findings"]}
    assert sev.get("schema.features_not_checked") == "info"
    assert "critical" not in sev.values() and "warning" not in sev.values()
    s = score_episodes(ds)
    assert not any("depth" in k for k in s["summary"]["context_features"])


def test_scores_without_state_do_not_compare_the_action_with_itself(tmp_path):
    root = _write(tmp_path)
    _edit_data(root, lambda df: df.drop(columns=["observation.state"]))
    _edit_info(root, lambda info: info["features"].pop("observation.state"))
    s = score_episodes(load_dataset(root))
    assert s["summary"]["relative_actions"] is False
    assert "action" not in s["summary"]["context_features"]
    assert all(e["deminf"] is None for e in s["episodes"].values())
    assert any("observation.state" in n for n in s["summary"]["skipped"])


def test_a_never_finite_channel_is_one_dataset_finding(tmp_path):
    eps = syn.make_episodes(12, 120)
    for e in eps:
        e["state"][:, 2] = np.nan
    root = _write(tmp_path, eps)
    r = audit(root)
    assert "values.empty_channel" in _checks(r)
    assert "values.non_finite" not in _checks(r)
    assert len(r["score"]["usable_episodes"]) == 12
    s = score_episodes(load_dataset(root))  # used to crash in _robust_standardise
    assert any(e["composite"] is not None for e in s["episodes"].values())


def test_missing_or_null_fps_is_reported_not_a_crash(tmp_path):
    root = _write(tmp_path)
    _edit_info(root, lambda info: info.update(fps=None))
    r = audit(root)
    assert "info.bad_fps" in _checks(r)
    assert "? fps" in format_terminal(r, color=False)
    render_html(r)
    _edit_info(root, lambda info: info.pop("fps"))
    r = audit(root)
    assert "info.missing_key" in _checks(r)
    format_terminal(r, color=False)


def test_very_few_frames_skip_the_scores(tmp_path):
    root = _write(tmp_path, syn.make_episodes(1, 6))
    s = score_episodes(load_dataset(root))
    assert all(e["deminf"] is None for e in s["episodes"].values())


def test_one_episode_with_a_different_width_is_left_out(tmp_path):
    root = _write(tmp_path)

    def widen(df):
        df["action"] = [np.append(a, 0.0) if ep == 3 else a for a, ep in zip(df["action"], df["episode_index"])]
        return df

    _edit_data(root, widen)
    s = score_episodes(load_dataset(root))
    assert s["episodes"][3]["deminf"] is None and "width" in s["episodes"][3]["notes"][0]
    assert any(e["composite"] is not None for i, e in s["episodes"].items() if i != 3)


def test_a_dataset_that_never_moves_is_critical(tmp_path):
    eps = syn.make_episodes(12, 120)
    for e in eps:
        e["action"][:] = e["action"][0]
        e["state"][:] = e["state"][0]
    r = audit(_write(tmp_path, eps))
    assert "episodes.no_motion" in _checks(r)
    assert len(r["score"]["usable_episodes"]) == 0


def test_data_episodes_missing_from_metadata(tmp_path):
    root = _write(tmp_path)
    _edit_data(root, lambda df: df.assign(episode_index=df["episode_index"] + 100))
    r = audit(root)
    assert {"counts.episodes_not_in_metadata", "counts.metadata_episodes_missing"} <= _checks(r)


def test_low_motion_task_is_compared_with_its_own_task(tmp_path):
    eps = syn.make_episodes(24, 120, seed=3)
    for i, e in enumerate(eps):
        if i < 6:  # "press the button": small but real motion
            e["action"] = e["action"][0] + 0.15 * (e["action"] - e["action"][0])
            e["state"] = e["state"][0] + 0.15 * (e["state"] - e["state"][0])
            e["task_index"] = 1
        else:
            e["task_index"] = 0
    r = audit(_write(tmp_path, eps, tasks=["pick", "press"]))
    idle = [f for f in r["findings"] if f["check"] == "episodes.idle"]
    assert not idle or not set(idle[0]["episodes"]) & set(range(6))


def test_systematic_spikes_are_reported_once(tmp_path):
    eps = syn.make_episodes(20, 150)
    for e in eps:  # a channel with a fast 2-frame pulse in every episode, e.g. a gripper trigger
        e["state"][60, 0] += 50 * (e["state"][:, 0].max() - e["state"][:, 0].min() + 1)
    r = audit(_write(tmp_path, eps))
    assert "episodes.spikes_systematic" in _checks(r)
    spikes = [f for f in r["findings"] if f["check"] == "episodes.spikes"]
    assert not spikes or len(spikes[0]["episodes"]) < 5


def test_one_episode_says_what_was_not_checked(tmp_path):
    r = audit(_write(tmp_path, syn.make_episodes(1, 200)))
    assert r["not_checked"]
    assert "Not checked" in format_terminal(r, color=False)


def test_control_characters_are_stripped_from_the_terminal(tmp_path):
    root = _write(tmp_path, tasks=["\x1b[2J\x1b[31mpwned"])
    out = format_terminal(audit(root), color=False)
    assert "\x1b" not in out


# --------------------------------------------------------------------------- #
# Regressions from the second review (R8)


def test_dead_state_channel_keeps_the_sync_check(tmp_path):
    eps = syn.make_episodes(20, 150, seed=4)
    for i in (2, 9, 15):
        syn.delay_action(eps[i], 10)
    for e in eps:
        e["state"][:, 3] = np.nan
    r = audit(_write(tmp_path, eps))
    lag = [f for f in r["findings"] if f["check"] == "sync.action_state_lag"]
    assert lag and set(lag[0]["episodes"]) >= {2, 9, 15}


def test_a_task_whose_recordings_are_all_frozen(tmp_path):
    eps = syn.make_episodes(30, 120, seed=5)
    for i, e in enumerate(eps):
        e["task_index"] = i % 3
        if i % 3 == 2:
            e["action"][:] = e["action"][0]
            e["state"][:] = e["state"][0]
    r = audit(_write(tmp_path, eps, tasks=["a", "b", "c"]))
    idle = [f for f in r["findings"] if f["check"] == "episodes.idle"]
    assert idle and set(idle[0]["episodes"]) == {i for i in range(30) if i % 3 == 2}


def test_glitches_in_a_small_dataset_still_exclude_episodes(tmp_path):
    eps = syn.make_episodes(3, 150)
    for e in eps[:2]:
        syn.add_glitch(e)
    r = audit(_write(tmp_path, eps))
    assert "episodes.spikes" in _checks(r) and "episodes.spikes_systematic" not in _checks(r)


def test_frozen_state_with_sensor_noise_is_critical(tmp_path):
    rng = np.random.default_rng(0)
    eps = syn.make_episodes(20, 150)
    for e in eps:
        e["state"] = e["state"][:1] * 0 + rng.normal(0, 0.05, e["state"].shape)
    r = audit(_write(tmp_path, eps))
    assert "episodes.no_motion" in _checks(r)


def test_all_nan_state_gives_no_fake_ranking(tmp_path):
    eps = syn.make_episodes(20, 150)
    for e in eps:
        e["state"][:] = np.nan
    root = _write(tmp_path, eps)
    r = audit(root)
    assert "values.empty_feature" in _checks(r)
    s = score_episodes(load_dataset(root))
    assert all(e["deminf"] is None for e in s["episodes"].values())


def test_skipped_action_is_named_as_such(tmp_path):
    root = _write(tmp_path)
    _edit_data(root, lambda df: df.assign(action=["key_w"] * len(df)))
    _edit_info(root, lambda info: info["features"]["action"].update(dtype="string", shape=[1]))
    r = audit(root)
    assert "schema.action_not_numeric" in _checks(r) and "schema.no_action" not in _checks(r)


def test_sync_runs_on_named_channels_when_widths_differ(tmp_path):
    eps = syn.make_episodes(20, 150, seed=6)
    for i in (3, 11):
        syn.delay_action(eps[i], 10)
    root = _write(tmp_path, eps)

    def add_velocities(df):
        df["observation.state"] = [np.concatenate([s, np.gradient(np.asarray(s))]) for s in df["observation.state"]]
        return df

    _edit_data(root, add_velocities)

    def declare(info):
        ft = info["features"]["observation.state"]
        names = list(ft["names"]) if isinstance(ft["names"], list) else list(ft["names"].values())[0]
        ft["names"] = names + [f"{n}.vel" for n in names]
        ft["shape"] = [2 * len(names)]

    _edit_info(root, declare)
    r = audit(root)
    lag = [f for f in r["findings"] if f["check"] == "sync.action_state_lag"]
    assert lag and set(lag[0]["episodes"]) >= {3, 11}
    assert not any("sync" in n for n in r["not_checked"])
