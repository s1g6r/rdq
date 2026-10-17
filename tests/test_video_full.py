"""rdq check --video-full on a synthetic LeRobot v3 dataset whose episodes share one real mp4."""

import shutil
import subprocess

import numpy as np
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


def _records(make_episode, n=24):
    return [{"episode": e, "dec": {"samples": {"thumb": make_episode(e)}}} for e in range(n)]


def test_viewpoints_hand_held_camera_carried_elsewhere_is_not_a_viewpoint_change():
    """A UMI-style gripper camera: the fingers sit still in the frame, everything else moves with the hand.
    Halfway through, the operator works in another room (different overall brightness and texture)."""
    from rdq.video import viewpoints

    rng = np.random.default_rng(0)
    texture = np.kron(rng.uniform(-70, 70, size=(9, 16)), np.ones((4, 4)))     # 36 x 64
    down = np.linspace(40, 200, 18)[:, None] * np.ones((1, 32))     # looking down at a bright table
    up = down[::-1] * 0.8 + np.linspace(0, 60, 32)[None]            # held upright toward a window

    def ep(e):
        layout = down if e < 12 else up
        th = np.empty((30, 18, 32))
        for k in range(30):                                         # the hand drifts around
            y, x = rng.integers(0, 18), rng.integers(0, 32)
            th[k] = layout + texture[y:y + 18, x:x + 32]
        th[:, 12:, 10:22] = 230                                     # gripper fingers, fixed in frame
        th[:, :, :3] = th[:, :, -3:] = 0                            # fisheye border
        return th.clip(0, 255).astype(np.uint8)

    assert viewpoints(_records(ep)) is None


def test_viewpoints_fixed_camera_moved_between_sessions_is_found():
    from rdq.video import viewpoints

    rng = np.random.default_rng(1)
    room_a = rng.uniform(0, 255, size=(18, 32))
    room_b = np.roll(room_a, 9, axis=1)[::-1]                        # the same room from another spot

    def ep(e):
        th = np.repeat((room_a if e < 10 else room_b)[None], 30, 0) + rng.normal(0, 2, size=(30, 18, 32))
        for k in range(30):                                          # the arm moves through a small part
            th[k, 4:9, (k % 20):(k % 20) + 6] = 255
        return th.clip(0, 255).astype(np.uint8)

    v = viewpoints(_records(ep))
    assert v is not None and v["ranges"] == ["episodes 0-9", "episodes 10-23"]


def test_single_camera_needs_a_close_match_to_call_a_duplicate():
    """berkeley_mvp (one wrist camera, 5 fps): other takes of the same grasp scored 0.80-0.85."""
    from rdq.video import judge_duplicates

    def hit(corr, kind="partial_overlap"):
        return [{"camera": "wrist", "kind": kind, "overlap_s": 8.0, "share_of_shorter": 0.8, "motion_corr": corr}]

    out = {f.check: f for f in judge_duplicates({(11, 52): hit(0.81), (3, 4): hit(0.97)}, n_cams=1)}
    assert out["video.duplicate_episode"].episodes == [4]
    assert [p["copy"] for p in out["video.possible_duplicate"].details["pairs"]] == [52]
    out = {f.check for f in judge_duplicates({(1, 2): hit(0.5, "identical_file")}, n_cams=1)}
    assert out == {"video.duplicate_episode"}
    two = [{"camera": c, "kind": "partial_overlap", "overlap_s": 8.0, "share_of_shorter": 0.8, "motion_corr": 0.82}
           for c in ("top", "wrist")]
    assert {f.check for f in judge_duplicates({(5, 6): two}, n_cams=2)} == {"video.duplicate_episode"}


def test_frozen_footage_in_a_simulated_dataset_is_a_still_scene(dataset, tmp_path):
    """Rendered footage has no sensor noise, so a still scene repeats frames exactly (zima-robot, robosuite)."""
    from rdq.video import is_simulated

    sim = tmp_path / "sim"
    shutil.copytree(dataset, sim)
    (sim / "README.md").write_text("---\nlicense: mit\ntags:\n- LeRobot\n- robosuite\n---\n# sim data\n")
    ds = load_dataset(sim)
    assert is_simulated(ds) and not is_simulated(load_dataset(dataset))
    f = {x.check: x for x in check_videos_full(ds, workers=1)}
    assert f["video.frozen_span"].severity == "info" and "simulated" in f["video.frozen_span"].message
    assert f["video.black_span"].severity == "warning"      # a black render is still a problem
