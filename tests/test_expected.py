import json

import pytest

from rdq import synthetic as syn
from rdq.checks import INFO, WARNING, Finding
from rdq.expected import ExpectedError, apply, parse
from rdq.report import audit


def _idle_dataset(tmp_path, idle=(3, 7, 12)):
    eps = syn.make_episodes(n_episodes=24, seed=5)
    for i in idle:
        syn.make_idle(eps[i])
    return syn.write_dataset(tmp_path / "ds", eps)


def _flagged(result, check_prefix=""):
    return {e for f in result["findings"] if f["severity"] != INFO and f["check"].startswith(check_prefix)
            for e in f["episodes"]}


def test_owner_file_in_dataset_marks_findings_expected(tmp_path):
    root = _idle_dataset(tmp_path)
    before = audit(root)
    assert {3, 7, 12} <= _flagged(before)
    (root / "meta" / "rdq_expected.yaml").write_text(
        "- episodes: [3, 7]\n  reason: robot holds still on purpose (no object present)\n")
    after = audit(root)
    assert 3 not in _flagged(after) and 7 not in _flagged(after)
    assert 12 in _flagged(after)  # not listed, still flagged
    assert {3, 7} <= set(after["score"]["usable_episodes"])
    assert 12 not in after["score"]["usable_episodes"]
    marked = [f for f in after["findings"] if f["details"].get("expected")]
    assert marked and all(f["severity"] == INFO for f in marked)
    assert all("no object present" in f["message"] for f in marked)
    assert after["expected"]["file"] == "meta/rdq_expected.yaml"


def test_checks_limit_what_is_expected(tmp_path):
    root = _idle_dataset(tmp_path)
    exp = tmp_path / "exp.yaml"
    exp.write_text("- episodes: '0-23'\n  checks: [video.*]\n  reason: empty-scene clips\n")
    after = audit(root, expected=exp)
    assert {3, 7, 12} <= _flagged(after)  # idle checks are not video.*, so nothing changed
    assert after["expected"]["findings_marked"] == 0


def test_partial_overlap_splits_the_finding():
    f = Finding("video.idle_span", WARNING, "3 episodes are idle", [1, 2, 3])
    out = apply([f], parse([{"episodes": [2], "reason": "pause is part of the task"}]))
    sev = {tuple(x.episodes): x.severity for x in out}
    assert sev == {(1, 3): WARNING, (2,): INFO}


def test_cli_flag_and_bad_file(tmp_path, capsys):
    from rdq.cli import main

    root = _idle_dataset(tmp_path)
    good = tmp_path / "e.json"
    good.write_text(json.dumps([{"episodes": [3, 7, 12], "reason": "calibration holds"}]))
    out = tmp_path / "r.json"
    assert main(["check", str(root), "--expected", str(good), "--json", str(out), "--no-color"]) == 0
    assert "marked as expected" in capsys.readouterr().out
    assert {3, 7, 12} <= set(json.loads(out.read_text())["score"]["usable_episodes"])
    bad = tmp_path / "bad.yaml"
    bad.write_text("- episodes: [3]\n")  # no reason
    assert main(["check", str(root), "--expected", str(bad)]) == 2
    assert "reason" in capsys.readouterr().err


@pytest.mark.parametrize("data", [
    [{"reason": "x"}],                       # no episodes and no checks: would hide everything
    [{"episodes": ["a-b"], "reason": "x"}],
    [{"episodes": [1], "checks": 5, "reason": "x"}],
    "not a list",
])
def test_rejects_malformed_entries(data):
    with pytest.raises(ExpectedError):
        parse(data)


def test_dataset_level_entry_needs_a_check():
    f = Finding("meta.stats_stale", WARNING, "stats differ")
    out = apply([f], parse([{"checks": ["meta.*"], "reason": "stats recomputed downstream"}]))
    assert out[0].severity == INFO and out[0].details["original_severity"] == WARNING


def test_markdown_report_lists_findings_and_licence(tmp_path):
    from rdq.report import format_markdown

    root = _idle_dataset(tmp_path)
    (root / "README.md").write_text("---\nlicense: apache-2.0\n---\nA | dataset with a <pipe>\n")
    md = format_markdown(audit(root))
    assert md.startswith("### rdq:") and "**Score" in md
    assert "| | Check | Finding | Episodes |" in md
    assert "**Licence:** apache-2.0" in md
    assert "<pipe>" not in md            # no raw HTML from the card or messages


def test_json_report_has_documented_top_level_keys(tmp_path):
    root = _idle_dataset(tmp_path)
    r = audit(root)
    assert r["report_version"] == 1
    for k in ("tool", "audited_at", "dataset", "not_checked", "score", "findings", "provenance", "thresholds"):
        assert k in r
    for k in ("dataset_score", "usable_episodes", "usable_fraction", "flagged_episodes", "severity_counts"):
        assert k in r["score"]
    assert set(r["findings"][0]) == {"check", "severity", "message", "episodes", "details"}
