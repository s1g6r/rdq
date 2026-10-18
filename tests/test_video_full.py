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
    eps[5] = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in eps[1].items()}  # a copy duplicates the data too
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

    # berkeley_rpt (one wrist camera, 30 fps): other takes of the same grasp scored 0.90-0.93
    out = {f.check: f for f in judge_duplicates({(11, 52): hit(0.81), (49, 563): hit(0.93), (3, 4): hit(0.99)}, n_cams=1)}
    assert out["video.duplicate_episode"].episodes == [4]
    assert sorted(p["copy"] for p in out["video.possible_duplicate"].details["pairs"]) == [52, 563]
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


def test_viewpoints_fixed_camera_with_a_shaky_first_session_is_still_found():
    """EPITECH-LILLE/eval_training: the camera was moved after episode 19, and in the first session most of the
    picture changes within each episode, so a test on still pixels alone found nothing to compare."""
    from rdq.video import viewpoints

    rng = np.random.default_rng(2)
    room_a = rng.uniform(0, 255, size=(18, 32))
    room_b = np.roll(room_a, 11, axis=1)[::-1]

    def ep(e):
        if e < 20:
            th = np.repeat(room_a[None], 30, 0) + rng.normal(0, 25, size=(30, 18, 32))   # lots of change
        else:
            th = np.repeat(room_b[None], 30, 0) + rng.normal(0, 2, size=(30, 18, 32))
            for k in range(30):
                th[k, 4:9, (k % 20):(k % 20) + 6] = 255
        return th.clip(0, 255).astype(np.uint8)

    v = viewpoints(_records(ep, n=60))
    assert v is not None and v["ranges"] == ["episodes 0-19", "episodes 20-59"]


def _bg(img):
    b = img.ravel().astype(np.float32)
    return (b - b.mean()) / (b.std() + 1e-6)


def test_cameras_swapped_between_sessions():
    """EPITECH-LILLE/eval_training: in episodes 0-19 'front' carries the gripper view and 'back' the overhead one."""
    from rdq.video import camera_swaps

    rng = np.random.default_rng(3)
    overhead, gripper = rng.uniform(0, 255, (18, 32)), rng.uniform(0, 255, (18, 32))
    noisy = lambda v: _bg(v + rng.normal(0, 15, v.shape))  # noqa: E731
    front = {e: noisy(gripper if e < 10 else overhead) for e in range(30)}
    back = {e: noisy(overhead if e < 10 else gripper) for e in range(30)}
    assert camera_swaps({"front": front, "back": back}) == [("back", "front", list(range(10)))]


def test_moved_camera_or_similar_views_are_not_a_swap():
    from rdq.video import camera_swaps

    rng = np.random.default_rng(4)
    v1, v2, v3 = (rng.uniform(0, 255, (18, 32)) for _ in range(3))
    noisy = lambda v: _bg(v + rng.normal(0, 15, v.shape))  # noqa: E731
    moved = {"front": {e: noisy(v1 if e < 10 else v3) for e in range(30)}, "back": {e: noisy(v2) for e in range(30)}}
    assert camera_swaps(moved) == []
    twins = {"left": {e: noisy(v1) for e in range(30)}, "right": {e: noisy(v1 * 0.9 + 10) for e in range(30)}}
    assert camera_swaps(twins) == []


def test_cameras_swapped_when_the_other_camera_also_moved():
    """The real EPITECH layout: 0-19 swapped; the overhead camera sits in one place for 20-39 and another for
    40-59, and the first session's overhead view (stored as 'back') resembles the third session's."""
    from rdq.video import camera_swaps

    rng = np.random.default_rng(5)
    gripper, over1, over2 = (rng.uniform(0, 255, (18, 32)) for _ in range(3))
    over0 = over2 * 0.9 + rng.uniform(0, 25, (18, 32))          # session 1's overhead ~ session 3's
    noisy = lambda v: _bg(v + rng.normal(0, 15, v.shape))  # noqa: E731
    front = {e: noisy(gripper if e < 20 else over1 if e < 40 else over2) for e in range(60)}
    back = {e: noisy(over0 if e < 20 else gripper) for e in range(60)}
    assert camera_swaps({"front": front, "back": back}) == [("back", "front", list(range(20)))]


def test_scattered_matches_from_a_wrist_camera_are_not_a_swap():
    """allenai molmoact cup stacking: the left wrist camera sometimes points at what the right camera sees."""
    from rdq.video import camera_swaps

    rng = np.random.default_rng(6)
    table, top = rng.uniform(0, 255, (18, 32)), rng.uniform(0, 255, (18, 32))
    noisy = lambda v, n=15: _bg(v + rng.normal(0, n, v.shape))  # noqa: E731
    odd = {7, 12, 19, 22, 25, 28, 31}
    wrist = {e: noisy(table if e in odd else rng.uniform(0, 255, (18, 32))) for e in range(35)}
    right = {e: noisy(table * 0.5 + top * 0.5 if e in odd else table) for e in range(35)}
    assert camera_swaps({"left": wrist, "right": right}) == []


def test_one_odd_episode_is_not_a_swap():
    """allenai molmoact cup stacking, episode 34: one coincidental match is not a session of swapped cameras."""
    from rdq.video import camera_swaps

    rng = np.random.default_rng(7)
    v1, v2 = rng.uniform(0, 255, (18, 32)), rng.uniform(0, 255, (18, 32))
    noisy = lambda v: _bg(v + rng.normal(0, 15, v.shape))  # noqa: E731
    a = {e: noisy(v2 if e == 34 else v1) for e in range(35)}
    b = {e: noisy(v1 if e == 34 else v2) for e in range(35)}
    assert camera_swaps({"a": a, "b": b}) == []


def test_viewpoint_groups_skip_episodes_too_short_to_measure():
    """ukachyuthan/imitation_learning_v2 crashed (KeyError) when a moved camera met an episode with < 4 samples."""
    from rdq.video import viewpoints

    rng = np.random.default_rng(8)
    room_a = rng.uniform(0, 255, size=(18, 32))
    room_b = np.roll(room_a, 9, axis=1)[::-1]
    recs = [{"episode": e, "dec": {"samples": {"thumb": (np.repeat((room_a if e < 10 else room_b)[None], 30, 0)
                                                        + rng.normal(0, 2, (30, 18, 32))).clip(0, 255).astype(np.uint8)}}}
            for e in range(24)]
    recs.append({"episode": 57, "dec": {"samples": {"thumb": np.zeros((2, 18, 32), np.uint8)}}})
    v = viewpoints(recs)
    assert v is not None and 57 not in v["of"]
    assert v["of"].get(57) is None          # the caller must not index it directly


def test_stereo_pair_is_not_a_swap():
    """lerobot/unitreeh1_rearrange_objects: cam_left and cam_right are the humanoid's stereo head cameras, so they
    see nearly the same picture; when the head turns in a few episodes, each view sits as close to the other
    camera's reference view as to its own, which looked like a swap (episodes 8, 11, 12). Checked by eye: no swap."""
    from rdq.video import camera_swaps

    for seed in range(40):
        rng = np.random.default_rng(seed)
        a, q = rng.uniform(0, 255, (18, 32)), rng.uniform(0, 255, (18, 32))
        noisy = lambda v: _bg(v + rng.normal(0, 15, v.shape))  # noqa: E731
        scene = {e: (0.4 * a + 0.6 * q if e in (8, 11, 12) else a) for e in range(30)}
        left = {e: noisy(scene[e]) for e in range(30)}
        right = {e: noisy(0.8 * scene[e] + 0.2 * q) for e in range(30)}
        assert camera_swaps({"cam_left": left, "cam_right": right}) == [], seed
        # as measured on the real pair: the two views correlate only ~0.6 (the stereo offset), all values in one band
        right = {e: noisy(0.6 * scene[e] + 0.6 * q) for e in range(30)}
        assert camera_swaps({"cam_left": left, "cam_right": right}) == [], seed


def test_near_identical_takes_with_different_robot_data_are_only_a_lead(tmp_path):
    """lerobot/dlr_sara_grid_clamp: a precise robot replays one motion, so two takes film almost the same (one camera,
    motion agreement 0.98-0.99), but the joint data differs as much as any two recordings: not a copy."""
    from rdq.video import data_agreement, judge_duplicates

    eps = syn.make_episodes(12, length=FPS * 6, seed=11)
    eps[9] = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in eps[2].items()}   # a real copy, data and all
    ds = load_dataset(syn.write_dataset(tmp_path / "ds", eps, fps=FPS))
    assert data_agreement(ds, 2, 9) < 0.05
    assert data_agreement(ds, 3, 7) > 0.3
    hit = lambda: [{"camera": "observation.images.image", "kind": "near_duplicate", "overlap_s": 5.5,  # noqa: E731
                    "share_of_shorter": 1.0, "motion_corr": 0.985}]
    found = {f.check: f for f in judge_duplicates({(2, 9): hit(), (3, 7): hit()}, 1, ds)}
    assert found["video.duplicate_episode"].episodes == [9]
    assert [(p["original"], p["copy"]) for p in found["video.possible_duplicate"].details["pairs"]] == [(3, 7)]


def test_one_frame_episode_of_a_1fps_camera_is_not_a_decode_error():
    """lerobot/uiuc_d3field records at 1 fps and has five one-frame episodes; their single frame was measured as
    1/30 s of video against a 1 s span, so the episodes were reported as video that ends early."""
    from rdq import rawvideo as rv

    n = 20
    t = np.arange(n, dtype=float)
    dec = {"frames": n, "t": t, "mean": np.full(n, 120.0), "std": np.full(n, 40.0), "dark": np.zeros(n),
           "mad": np.full(n, 3.0), "ident": np.zeros(n, bool), "exact": np.zeros(n, bool),
           "lap_t": t, "lap": np.full(n, 500.0), "decoded_duration_s": float(n)}
    rec = {"path": "file-000.mp4", "meta": {"duration_s": float(n)}, "dec": dec}
    seg = rv.segment_record(rec, 7.0, 8.0, "episode 47")
    assert seg["dec"]["frames"] == 1 and seg["dec"]["decoded_duration_s"] == 1.0
    issues, _ = rv.file_issues(seg, rv.Spec(), None)
    assert not [i for i in issues if i.check in ("video.truncated", "video.unreadable")]


def test_a_few_frames_of_a_slow_camera_are_not_a_copy(tmp_path):
    """lerobot/uiuc_d3field (1 fps): scripted takes with different objects matched over 8 frames of an empty table
    in 2-3 cameras; the robot data is all zeros, so it can't help."""
    from rdq.video import judge_duplicates

    eps = syn.make_episodes(10, length=18, seed=3)
    eps[2] = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in eps[1].items()}  # data gives no evidence against
    ds = load_dataset(syn.write_dataset(tmp_path / "ds", eps, fps=1))
    hits = [{"camera": f"observation.images.image_{k}", "kind": "partial_overlap", "overlap_s": 8.0,
             "share_of_shorter": 0.5, "motion_corr": 0.908} for k in (1, 3)]
    found = {f.check for f in judge_duplicates({(1, 2): hits}, 4, ds)}
    assert found == {"video.possible_duplicate"}
