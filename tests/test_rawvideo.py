"""rdq check-video on synthetic footage with known defects and copies."""

import json
import shutil

import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg is needed to make the duplicate variants", allow_module_level=True)

from rdq import synthvideo as sv
from rdq.cli import main
from rdq.rawvideo import Spec, check_video_batch


@pytest.fixture(scope="module")
def delivery(tmp_path_factory):
    d = tmp_path_factory.mktemp("delivery")
    (d / "alice").mkdir()
    (d / "bob").mkdir()
    sv.render_clip(d / "alice" / "clean.mp4", sv.ClipSpec(seconds=20, seed=1))
    sv.render_clip(d / "alice" / "defects.mp4", sv.ClipSpec(seconds=30, seed=2, defects={
        "freeze": (4, 7), "black": (9, 11), "covered": (13, 15), "blur": (17, 23), "drop": (25, 26.5)}))
    sv.render_clip(d / "bob" / "idle.mp4", sv.ClipSpec(seconds=30, seed=3, defects={"idle": (5, 25)}))
    sv.render_clip(d / "bob" / "own.mp4", sv.ClipSpec(seconds=20, seed=4))
    sv.reencode(d / "alice" / "clean.mp4", d / "bob" / "reupload.mp4", width=426, crf=32)
    sv.reencode(d / "alice" / "clean.mp4", d / "bob" / "cut.mp4", mirror=True, trim=(4, 16), brightness=0.05)
    sv.remux(d / "alice" / "clean.mp4", d / "bob" / "remux.mkv")
    sv.render_clip(d / "alice" / "vfr.mp4", sv.ClipSpec(seconds=20, seed=5, vfr_from=10))
    sv.faststart(d / "alice" / "clean.mp4", d / "fs.tmp.mp4")
    sv.truncate(d / "fs.tmp.mp4", d / "alice" / "truncated.mp4")
    (d / "fs.tmp.mp4").unlink()
    return d


@pytest.fixture(scope="module")
def report(delivery):
    return check_video_batch([str(delivery)], by_folder=True, workers=2)


def _file(report, name):
    return next(f for f in report["files"] if f["path"].endswith(name))


def _checks(report, name):
    return {i["check"] for i in _file(report, name)["issues"]}


def test_clean_clip_has_no_findings(report):
    assert _file(report, "clean.mp4")["status"] == "ok"
    assert _file(report, "own.mp4")["status"] == "ok"


def test_content_defects_found_at_the_right_time(report):
    f = _file(report, "defects.mp4")
    by = {i["check"]: i for i in f["issues"]}
    for check, (a, b) in {"video.frozen": (4, 7), "video.black": (9, 11), "video.covered": (13, 15),
                          "video.blur": (17, 23)}.items():
        assert check in by, check
        starts = [s[0] for s in by[check]["spans"]]
        assert any(abs(s - a) <= 1.5 for s in starts), (check, by[check]["spans"])
        assert by[check]["seconds"] >= 0.5 * (b - a)
    assert "video.recording_gap" in by
    assert f["usable_s"] < f["duration_s"] - 8


def test_idle_footage_is_measured(report):
    i = next(i for i in _file(report, "idle.mp4")["issues"] if i["check"] == "video.idle")
    assert 15 <= i["seconds"] <= 21


def test_variable_frame_rate(report):
    assert "video.variable_frame_rate" in _checks(report, "vfr.mp4")


def test_truncated_file(report):
    assert "video.truncated" in _checks(report, "truncated.mp4")


def test_copies_are_reported_once_against_the_original(report):
    assert "video.near_duplicate" in _checks(report, "reupload.mp4")
    assert "video.trimmed_copy" in _checks(report, "cut.mp4")          # mirrored, brighter, cut
    assert "video.same_stream_remuxed" in _checks(report, "remux.mkv")
    for name in ("reupload.mp4", "cut.mp4", "remux.mkv"):
        dup_checks = {"video.near_duplicate", "video.trimmed_copy", "video.same_stream_remuxed",
                      "video.identical_file", "video.partial_overlap"}
        msgs = [i["message"] for i in _file(report, name)["issues"] if i["check"] in dup_checks]
        assert len(msgs) == 1 and "clean.mp4" in msgs[0] and "another contributor" in msgs[0], msgs
    assert _file(report, "clean.mp4")["status"] == "ok"


def test_unrelated_clips_are_not_duplicates(report):
    pairs = {(d["a"].split("/")[-1], d["b"].split("/")[-1]) for d in report["duplicates"]}
    for a, b in pairs:
        assert "own.mp4" not in (a, b) and "idle.mp4" not in (a, b), pairs


def test_scorecard_ranks_the_copying_contributor_last(report):
    names = [r["contributor"] for r in report["scorecard"]]
    assert names[0] == "bob"
    bob = report["scorecard"][0]
    assert bob["cross_contributor_duplicates"] >= 3 and bob["usable_share"] < 0.6


def test_spec_violations(delivery):
    r = check_video_batch([str(delivery / "alice" / "clean.mp4")], spec=Spec(fps=25, min_width=1280, codecs=["hevc"]),
                          workers=1)
    assert {"video.spec_fps", "video.spec_resolution", "video.spec_codec"} <= {i["check"] for i in r["files"][0]["issues"]}


def test_store_catches_copies_across_batches(delivery, tmp_path):
    store = tmp_path / "hashes.json"
    first = check_video_batch([str(delivery / "alice" / "clean.mp4")], store=str(store), workers=1)
    from rdq.rawvideo import write_outputs

    write_outputs(first)
    second = check_video_batch([str(delivery / "bob" / "reupload.mp4")], store=str(store), workers=1)
    issues = second["files"][0]["issues"]
    assert any(i["check"] == "video.near_duplicate" and "earlier batch" in i["message"] for i in issues), issues


def test_unreadable_file(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video at all" * 100)
    r = check_video_batch([str(bad)], workers=1)
    assert r["files"][0]["status"] == "critical" and r["files"][0]["usable_s"] == 0


def test_cli_writes_reports(delivery, tmp_path, capsys):
    rc = main(["check-video", str(delivery / "alice"), "--json", str(tmp_path / "r.json"),
               "--html", str(tmp_path / "r.html"), "--csv", str(tmp_path / "r.csv"), "--workers", "1"])
    assert rc == 1     # the truncated file is critical
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["summary"]["files"] == 4
    assert "Video delivery report" in (tmp_path / "r.html").read_text()
    assert "truncated.mp4" in (tmp_path / "r.csv").read_text()
    assert "usable" in capsys.readouterr().out
