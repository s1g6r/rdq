"""rdq check --video-full on a synthetic LeRobot v3 dataset whose episodes share one real mp4."""

import shutil
import subprocess

import pytest

pytest.importorskip("av")
pytest.importorskip("cv2")
if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg is needed to join the episode clips", allow_module_level=True)

from rdq import synthetic as syn
from rdq import synthvideo as sv
from rdq.dataset import load_dataset
from rdq.video import check_videos_full

FPS, SECONDS, N = 30, 10, 8


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lerobot")
    eps = syn.make_episodes(N, length=FPS * SECONDS)
    root = syn.write_dataset(tmp / "ds", eps, fps=FPS)
    clips = tmp / "clips"
    clips.mkdir()
    defects = {3: {"freeze": (3, 5.5)}, 6: {"black": (2, 4)}}
    paths = []
    for k in range(N):
        seed = 101 if k == 5 else 100 + k        # episode 5 is a second copy of episode 1's footage
        spec = sv.ClipSpec(seconds=(len(eps[k]["action"]) + 0.5) / FPS, fps=FPS, width=320, height=180, seed=seed, scene_seed=9,
                           static_camera=True, defects=defects.get(k, {}))
        p = clips / f"ep{k}.mp4"
        sv.render_clip(p, spec)
        paths.append(p)
    lst = clips / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in paths))
    out = root / "videos/observation.images.front/chunk-000/file-000.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                   check=True)
    return root


@pytest.fixture(scope="module")
def findings(dataset):
    return {f.check: f for f in check_videos_full(load_dataset(dataset), workers=1)}


def test_defects_land_on_the_right_episodes(findings):
    assert findings["video.frozen_span"].episodes == [3]
    assert findings["video.black_span"].episodes == [6]
    assert "video.decode_error" not in findings
    assert "video.viewpoint_change" not in findings      # one fixed camera throughout


def test_duplicated_episode_is_found_and_other_takes_are_not(findings):
    d = findings["video.duplicate_episode"]
    assert d.episodes == [5]
    assert {(p["original"], p["copy"]) for p in d.details["pairs"]} == {(1, 5)}


def test_cli_flag(dataset, tmp_path):
    from rdq.cli import main

    main(["check", str(dataset), "--video-full", "--json", str(tmp_path / "r.json"), "--no-color"])
    assert "video.duplicate_episode" in (tmp_path / "r.json").read_text()


def test_camera_moved_between_sessions(tmp_path):
    eps = syn.make_episodes(8, length=FPS * 6)
    root = syn.write_dataset(tmp_path / "ds", eps, fps=FPS)
    clips = tmp_path / "clips"
    clips.mkdir()
    paths = []
    for k in range(8):   # episodes 0-3 from one camera position, 4-7 from another
        spec = sv.ClipSpec(seconds=(len(eps[k]["action"]) + 0.5) / FPS, fps=FPS, width=320, height=180, seed=200 + k,
                           scene_seed=1 if k < 4 else 2, static_camera=True)
        sv.render_clip(clips / f"ep{k}.mp4", spec)
        paths.append(clips / f"ep{k}.mp4")
    lst = clips / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in paths))
    out = root / "videos/observation.images.front/chunk-000/file-000.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)],
                   check=True)
    found = {f.check: f for f in check_videos_full(load_dataset(root), workers=1)}
    assert found["video.viewpoint_change"].details["groups"] == ["episodes 0-3", "episodes 4-7"]
    assert "video.blur_span" not in found
