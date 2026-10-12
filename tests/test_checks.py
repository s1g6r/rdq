import copy

import numpy as np
import pytest

from rdq import synthetic as syn
from rdq.checks import CRITICAL, INFO, WARNING, run_checks
from rdq.dataset import load_dataset
from rdq.provenance import analyse_card
from rdq.scoring import score


def _audit(tmp_path, episodes, **kwargs):
    root = syn.write_dataset(tmp_path / "ds", episodes, **kwargs)
    ds = load_dataset(root)
    findings = run_checks(ds)
    return ds, findings


def _by_check(findings):
    out = {}
    for f in findings:
        out.setdefault(f.check, []).append(f)
    return out


@pytest.fixture
def clean():
    return syn.make_episodes(n_episodes=30, seed=3)


def test_clean_dataset_has_no_problems(tmp_path, clean):
    ds, findings = _audit(tmp_path, clean)
    serious = [f for f in findings if f.severity in (CRITICAL, WARNING)]
    assert serious == [], [f.message for f in serious]
    s = score(findings, sorted(ds.episodes))
    assert s.usable_fraction == 1.0
    assert s.dataset_score >= 95


def test_loader_reads_v3(tmp_path, clean):
    ds, _ = _audit(tmp_path, clean)
    assert ds.version == "3.0"
    assert len(ds.episodes) == 30
    assert ds.action_key == "action" and ds.state_key == "observation.state"
    assert ds.episodes[0].features["action"].shape[1] == 6
    assert ds.tasks[0].startswith("Pick up")


@pytest.mark.parametrize(
    "fault, check, severity",
    [
        (syn.make_idle, "episodes.idle", CRITICAL),
        (syn.drop_frames, "time.dropped_frames", WARNING),
        (syn.reverse_time, "time.non_monotonic", CRITICAL),
        (syn.add_spike, "episodes.spikes", WARNING),
        (syn.shift_action, "sync.action_state_lag", WARNING),
        (syn.delay_action, "sync.action_state_lag", WARNING),
        (syn.add_nan, "values.non_finite", CRITICAL),
        (syn.truncate, "episodes.too_short", WARNING),
        (syn.pad_idle, "episodes.idle_padding", WARNING),
    ],
)
def test_each_fault_is_caught_on_the_right_episode(tmp_path, clean, fault, check, severity):
    eps = copy.deepcopy(clean)
    fault(eps[7])
    _, findings = _audit(tmp_path, eps)
    hits = _by_check(findings).get(check, [])
    assert hits, f"{check} did not fire; got {[f.check for f in findings]}"
    assert any(7 in f.episodes for f in hits)
    assert all(f.severity == severity for f in hits)
    # The fault should not smear across many clean episodes.
    flagged = {e for f in hits for e in f.episodes}
    assert len(flagged) <= 2, flagged


def test_duplicate_episode(tmp_path, clean):
    eps = copy.deepcopy(clean)
    eps[12] = copy.deepcopy(eps[4])
    _, findings = _audit(tmp_path, eps)
    hits = _by_check(findings)["episodes.duplicate"]
    assert hits[0].episodes == [12]


def test_dead_dimension(tmp_path, clean):
    eps = copy.deepcopy(clean)
    syn.kill_dimension(eps, dim=4)
    _, findings = _audit(tmp_path, eps)
    hit = _by_check(findings)["values.dead_dimension"]
    assert any("wrist_roll" in f.message for f in hit)


def test_placeholder_task(tmp_path, clean):
    _, findings = _audit(tmp_path, clean, tasks=["task"])
    assert "tasks.placeholder" in _by_check(findings)


def test_stale_stats(tmp_path, clean):
    _, findings = _audit(tmp_path, clean, stats_override={"action": {"max": [500.0] * 6}})
    assert "stats.stale" in _by_check(findings)


def test_names_shape_mismatch(tmp_path, clean):
    _, findings = _audit(tmp_path, clean, names=syn.MOTORS[:5])
    assert "schema.names_shape_mismatch" in _by_check(findings)


def test_video_span_mismatch(tmp_path, clean):
    _, findings = _audit(tmp_path, clean, video_span_error={3: 0.5})
    hit = _by_check(findings)["video.span_mismatch"]
    assert hit[0].episodes == [3]


def test_meta_length_mismatch(tmp_path, clean):
    _, findings = _audit(tmp_path, clean, meta_length_error={9: 5})
    hit = _by_check(findings)["counts.length_mismatch"]
    assert hit[0].episodes == [9]


def test_score_drops_with_faults(tmp_path, clean):
    eps = copy.deepcopy(clean)
    for i in (1, 5, 9):
        syn.make_idle(eps[i])
    ds, findings = _audit(tmp_path, eps)
    s = score(findings, sorted(ds.episodes))
    assert set(s.usable_episodes).isdisjoint({1, 5, 9})
    assert s.usable_fraction == pytest.approx(27 / 30)


# ---------------------------------------------------------------- provenance

TEMPLATE_CARD = """---
license: apache-2.0
tags:
- LeRobot
task_categories:
- robotics
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):

## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```
"""


def test_template_card_is_default_license():
    r = analyse_card(TEMPLATE_CARD)
    assert r.template_card and r.default_license_unconfirmed
    assert r.verdict == "default licence, unconfirmed"


def test_documented_card():
    card = TEMPLATE_CARD.replace(
        "## Dataset Description\n\n\n", "## Dataset Description\n\n120 pick-and-place demos recorded at MIT.\n"
    ).replace("- **Homepage:** [More Information Needed]", "- **Homepage:** https://example.org")
    card = card.replace("- **Paper:** [More Information Needed]", "- **Paper:** https://arxiv.org/abs/1234")
    r = analyse_card(card)
    assert not r.default_license_unconfirmed
    assert r.verdict == "documented"


def test_noncommercial_and_restricted():
    assert analyse_card("---\nlicense: cc-by-nc-4.0\n---\nhi").verdict == "non-commercial"
    r = analyse_card("---\nlicense: mit\n---\nConverted from EPIC-KITCHENS videos.")
    assert r.verdict == "restricted source"
    assert analyse_card("no front matter").verdict == "no licence"


def test_names_conventions():
    from rdq.checks import names_status

    six = ["a", "b", "c", "d", "e", "f"]
    assert names_status({"shape": [6], "names": six}) == "ok"
    assert names_status({"shape": [6], "names": [six]}) == "ok"  # wrapped list (v2.0 style)
    assert names_status({"shape": [6], "names": {"motors": six}}) == "ok"
    assert names_status({"shape": [7], "names": ["actions"]}) == "placeholder"
    assert names_status({"shape": [6], "names": six[:5]}) == "mismatch"
    assert names_status({"shape": [6], "names": None}) == "unnamed"
    assert names_status({"shape": [20, 6], "names": [six]}) == "ok"  # action chunk
    assert names_status({"shape": [44], "names": ["height", "width", "channels"]}) == "mismatch"


def test_gripper_steps_are_not_glitches(tmp_path, clean):
    eps = copy.deepcopy(clean)
    syn.add_gripper_steps(eps)
    _, findings = _audit(tmp_path, eps)
    assert "episodes.spikes" not in _by_check(findings)


def test_three_frame_glitch_is_caught(tmp_path, clean):
    eps = copy.deepcopy(clean)
    syn.add_gripper_steps(eps)
    syn.add_glitch(eps[11], frames=3)
    _, findings = _audit(tmp_path, eps)
    hit = _by_check(findings)["episodes.spikes"]
    assert {e for f in hit for e in f.episodes} == {11}


def test_orphaned_files_are_ignored(tmp_path, clean):
    import pandas as pd

    root = syn.write_dataset(tmp_path / "ds", clean)
    data = pd.read_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    # A stale copy of the first episodes, as left behind by an old upload.
    data[data["episode_index"] < 5].to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")
    ds = load_dataset(root)
    findings = _by_check(run_checks(ds))
    assert ds.episodes[0].length == len(clean[0]["action"])
    assert "time.non_monotonic" not in findings and "counts.length_mismatch" not in findings
    assert "files.orphaned" in findings


def test_v21_dataset_with_episode_stats(tmp_path, clean):
    root = syn.write_dataset_v21(tmp_path / "v21", clean)
    ds = load_dataset(root)
    assert ds.version == "2.1" and len(ds.episodes) == 30 and ds.has_episode_stats
    findings = _by_check(run_checks(ds))
    assert "stats.missing" not in findings
    assert "schema.names_shape_mismatch" not in findings  # names wrapped in a list
    serious = [f for f in run_checks(ds) if f.severity in (CRITICAL, WARNING)]
    assert serious == [], [f.message for f in serious]


def test_arrow_backed_list_columns(tmp_path, clean):
    import pandas as pd
    import pyarrow as pa

    root = syn.write_dataset(tmp_path / "ds", clean)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(path)
    df["action"] = pd.array(df["action"].tolist(), dtype=pd.ArrowDtype(pa.list_(pa.float64())))
    df.to_parquet(path)
    ds = load_dataset(root)
    assert ds.episodes[0].features["action"].shape[1] == 6


def test_delta_actions_are_not_glitches(tmp_path, clean):
    """Velocity/delta actions jump frame to frame by design."""
    rng = np.random.default_rng(0)
    eps = copy.deepcopy(clean)
    for ep in eps:
        d = np.diff(ep["state"], axis=0, prepend=ep["state"][:1])
        ep["action"] = d + rng.normal(0, 2.0, d.shape) * (rng.random(d.shape) < 0.05)
    _, findings = _audit(tmp_path, eps)
    assert not any(f.check == "episodes.spikes" and "action" in f.message for f in findings)
