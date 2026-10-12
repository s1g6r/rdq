"""Optional camera checks that decode a few frames per episode (needs PyAV: pip install av).

rdq's default checks read only tables. These checks open the videos and sample a handful
of frames per episode and camera, so they cost seconds per dataset rather than a full
decode. They catch what tables cannot:

  video.decode_error   a segment can't be opened or decoded, or ends before the data does
  video.frozen         every sampled frame of an episode is the same image (a stalled camera)
                       while the robot is moving
  video.black          most sampled frames are black or a single flat colour (covered lens,
                       disconnected camera, exposure failure)
  video.blur           frames are much blurrier than the rest of the dataset (info only)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .checks import CRITICAL, INFO, WARNING, Finding, _eps
from .dataset import Dataset

SAMPLES = 6
FROZEN_DIFF = 0.5  # mean absolute grey-level difference below which two frames are "identical"
FROZEN_SHARE = 0.6  # share of consecutive sampled pairs that must be identical (codec keyframes add small noise)
BLACK_MEAN = 12.0
FLAT_STD = 2.0


def _video_file(ds: Dataset, vkey: str, ep) -> tuple[Path | None, float, float]:
    """Path to the episode's video and the [start, end) seconds of its segment."""
    tmpl = ds.info.get("video_path")
    if not tmpl:
        return None, 0.0, 0.0
    meta = ep.meta or {}
    fps = ds.fps or 30.0
    if f"videos/{vkey}/chunk_index" in meta:  # v3.0: many episodes per file
        path = ds.root / tmpl.format(video_key=vkey, chunk_index=int(meta[f"videos/{vkey}/chunk_index"]),
                                     file_index=int(meta[f"videos/{vkey}/file_index"]))
        start = float(meta.get(f"videos/{vkey}/from_timestamp", 0.0))
        end = float(meta.get(f"videos/{vkey}/to_timestamp", start + ep.length / fps))
        return path, start, end
    chunk = int(ds.info.get("chunks_size") or 1000)
    try:  # v2.x: one file per episode
        path = ds.root / tmpl.format(video_key=vkey, episode_chunk=ep.index // chunk, episode_index=ep.index)
    except (KeyError, IndexError):
        return None, 0.0, 0.0
    return path, 0.0, ep.length / fps


def _gray_small(frame) -> np.ndarray:
    g = frame.to_ndarray(format="gray").astype(np.float32)
    h, w = g.shape
    fy, fx = max(1, h // 96), max(1, w // 128)
    return g[: h - h % fy, : w - w % fx].reshape(h // fy, fy, w // fx, fx).mean((1, 3))


def sample_frames(path: Path, times: list[float]) -> tuple[list[np.ndarray], str | None]:
    """Decode the frames nearest to `times` (seconds). Returns (frames, error)."""
    import av

    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001 - any open failure is a finding
        return [], f"cannot open {path.name}: {type(exc).__name__}"
    try:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        out = []
        for t in sorted(times):
            container.seek(max(0, int(t / tb)), stream=stream, backward=True, any_frame=False)
            got = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                if frame.pts * tb >= t - 0.5 / float(stream.average_rate or 30):
                    got = frame
                    break
            if got is None:
                return out, f"{path.name} ends before t={t:.2f}s"
            out.append(_gray_small(got))
        return out, None
    except Exception as exc:  # noqa: BLE001
        return [], f"decode failed in {path.name}: {type(exc).__name__}: {exc}"
    finally:
        container.close()


def _laplacian_var(g: np.ndarray) -> float:
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def check_videos(ds: Dataset, samples: int = SAMPLES, max_episodes: int | None = None) -> list[Finding]:
    try:
        import av  # noqa: F401
    except ImportError:
        return [Finding("video.not_checked", INFO, "PyAV is not installed (pip install av), so videos were not decoded")]
    out = []
    eps = sorted(ds.episodes.values(), key=lambda e: e.index)
    if max_episodes:
        eps = eps[:max_episodes]
    skey = ds.state_key
    for vkey in ds.video_keys:
        errors, frozen, black, blur = {}, [], [], {}
        for ep in eps:
            path, start, end = _video_file(ds, vkey, ep)
            if path is None:
                continue
            if not path.exists():
                errors[ep.index] = f"missing file {path.relative_to(ds.root)}"
                continue
            span = max(end - start, 1e-3)
            times = [start + span * (k + 0.5) / samples for k in range(samples)]
            frames, err = sample_frames(path, times)
            if err:
                errors[ep.index] = err
                continue
            if len(frames) < 2:
                continue
            diffs = [float(np.abs(a - b).mean()) for a, b in zip(frames[:-1], frames[1:])]
            moving = True
            if skey and skey in ep.features:
                s = ep.features[skey]
                moving = bool(np.nanmax(np.ptp(s, axis=0)) > 1e-6)
            dark = sum(1 for f in frames if f.mean() < BLACK_MEAN or f.std() < FLAT_STD)
            if dark >= len(frames) / 2:
                black.append(ep.index)
            elif np.mean([d < FROZEN_DIFF for d in diffs]) >= FROZEN_SHARE and moving:
                frozen.append(ep.index)
            blur[ep.index] = float(np.median([_laplacian_var(f) for f in frames]))
        if errors:
            out.append(Finding("video.decode_error", CRITICAL,
                               f"camera '{vkey}': video for {_eps(len(errors))} could not be decoded or is shorter than the data",
                               episodes=sorted(errors), details={"video_key": vkey, "errors": {str(k): v for k, v in list(errors.items())[:20]}}))
        if frozen:
            out.append(Finding("video.frozen", WARNING,
                               f"camera '{vkey}': in {_eps(len(frozen))} every sampled frame is the same image while the robot moves "
                               "(a stalled or disconnected camera feed)", episodes=frozen, details={"video_key": vkey}))
        if black:
            out.append(Finding("video.black", WARNING,
                               f"camera '{vkey}': in {_eps(len(black))} most sampled frames are black or one flat colour "
                               "(covered lens, unplugged camera or failed exposure)", episodes=black, details={"video_key": vkey}))
        if len(blur) >= 8:
            v = np.array(list(blur.values()))
            med = float(np.median(v))
            low = [i for i, b in blur.items() if med > 0 and b < 0.2 * med and i not in black]
            if low:
                out.append(Finding("video.blur", INFO,
                                   f"camera '{vkey}': {_eps(len(low))} are much blurrier than the rest of the dataset "
                                   "(focus, motion blur or a smudged lens)", episodes=sorted(low),
                                   details={"video_key": vkey, "median_sharpness": round(med, 2)}))
    return out
