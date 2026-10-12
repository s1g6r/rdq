"""Optional video checks on a synthetic v3.0 dataset with a real shared mp4."""

import numpy as np
import pytest

from rdq import synthetic as syn
from rdq.dataset import load_dataset

av = pytest.importorskip("av")


def _write_video(path, episodes_frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=fps)
    s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
    s.options = {"g": "10"}  # frequent keyframes so seeking is exact enough
    for frames in episodes_frames:
        for img in frames:
            for pk in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pk)
    for pk in s.encode():
        c.mux(pk)
    c.close()


def _frames(kind, n, rng):
    yy, xx = np.mgrid[0:48, 0:64]
    out = []
    for t in range(n):
        if kind == "black":
            img = np.zeros((48, 64, 3), np.uint8)
        elif kind == "frozen":
            img = (((xx * 4 + yy * 2) % 255)[..., None] * np.ones(3)).astype(np.uint8)
        else:
            img = ((((xx + 3 * t) * 4 + yy * 2) % 255)[..., None] * np.ones(3)).astype(np.uint8)
        out.append(img)
    return out


def test_video_checks(tmp_path):
    rng = np.random.default_rng(0)
    eps = syn.make_episodes(10, length=60)
    root = syn.write_dataset(tmp_path / "ds", eps, fps=30)
    kinds = ["ok"] * 10
    kinds[3], kinds[7] = "frozen", "black"
    _write_video(root / "videos/observation.images.front/chunk-000/file-000.mp4",
                 [_frames(k, len(e["action"]), rng) for k, e in zip(kinds, eps)], fps=30)
    from rdq.video import check_videos

    found = {f.check: f.episodes for f in check_videos(load_dataset(root))}
    assert found.get("video.frozen") == [3]
    assert found.get("video.black") == [7]
    assert "video.decode_error" not in found


def test_missing_video_file_is_reported(tmp_path):
    root = syn.write_dataset(tmp_path / "ds", syn.make_episodes(4, length=40), fps=30)
    from rdq.video import check_videos

    found = {f.check: f for f in check_videos(load_dataset(root))}
    assert found["video.decode_error"].episodes == [0, 1, 2, 3]
