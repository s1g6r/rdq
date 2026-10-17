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

import os
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


# --------------------------------------------------------------------------------------------- #
# Full decode (rdq check --video-full): every frame of every episode, plus duplicate episodes

FULL_CHECKS = {
    # rawvideo check -> (rdq check id, severity floor, what it means for an episode)
    "video.frozen": ("video.frozen_span", WARNING, "the camera image stops changing for a while (stalled feed)"),
    "video.black": ("video.black_span", WARNING, "black frames (covered lens, unplugged camera, failed exposure)"),
    "video.covered": ("video.covered_span", WARNING, "featureless frames (something over the lens)"),
    "video.overexposed": ("video.overexposed_span", WARNING, "washed-out frames"),
    "video.blur": ("video.blur_span", INFO, "blurry stretches without camera motion"),
    "video.idle": ("video.idle_span", INFO, "nothing moving in view for 5 s or more"),
    "video.recording_gap": ("video.recording_gap", WARNING, "gaps of more than a second in the frame timestamps"),
    "video.truncated": ("video.decode_error", CRITICAL, "the video ends before the episode does"),
    "video.unreadable": ("video.decode_error", CRITICAL, "the video can't be decoded"),
    "video.corrupt_frames": ("video.decode_error", CRITICAL, "corrupt frames"),
}
COPY_CHECKS = ("video.near_duplicate", "video.trimmed_copy", "video.same_stream_remuxed", "video.identical_file",
               "video.partial_overlap", "video.loop_within_file")


def check_videos_full(ds: Dataset, cameras: list[str] | None = None, max_episodes: int | None = None,
                      workers: int | None = None) -> list[Finding]:
    """Decode every frame of each episode's video (each shared file once) and report, per camera, the
    episodes with frozen, black, covered or blurry stretches, timing gaps, decode problems, and episodes
    whose footage repeats another episode's (a duplicated episode, often from merging datasets)."""
    try:
        import av  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        return [Finding("video.not_checked", INFO, "--video-full needs PyAV and OpenCV (pip install \"rdq[video]\")")]
    from . import rawvideo as rv

    eps = sorted(ds.episodes.values(), key=lambda e: e.index)
    if max_episodes:
        eps = eps[:max_episodes]
    out: list[Finding] = []
    dup_pairs: dict[tuple[int, int], list[dict]] = {}
    for vkey in (cameras or ds.video_keys):
        segs, missing = [], {}
        for ep in eps:
            path, start, end = _video_file(ds, vkey, ep)
            if path is None:
                continue
            if not path.exists():
                missing[ep.index] = f"missing file {path.relative_to(ds.root)}"
                continue
            segs.append((ep.index, path, start, end))
        files = sorted({str(p) for _, p, _, _ in segs})
        measured: dict[str, dict] = {}
        n_workers = workers or min(4, os.cpu_count() or 1)
        if n_workers > 1 and len(files) > 1:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn")) as ex:
                for f, rec in zip(files, ex.map(rv.measure_file, files)):
                    measured[f] = rec
        else:
            for f in files:
                measured[f] = rv.measure_file(f)
        records = []
        for idx, path, start, end in segs:
            r = rv.segment_record(measured[str(path)], start, end, f"episode {idx}")
            r["contributor"] = vkey
            r["episode"] = idx
            records.append(r)
        if not records:
            if missing:
                out.append(Finding("video.decode_error", CRITICAL, f"camera '{vkey}': video missing for {_eps(len(missing))}",
                                   episodes=sorted(missing), details={"video_key": vkey}))
            continue
        views = viewpoints(records)
        if views:
            for r in records:
                r["contributor"] = f"{vkey}#view{views['of'][r['episode']]}"   # sharpness is judged per viewpoint
            out.append(Finding("video.viewpoint_change", WARNING,
                               f"camera '{vkey}': the camera was moved between episodes, so the dataset mixes "
                               f"{len(views['groups'])} viewpoints ({'; '.join(views['ranges'])}). A policy trained on one "
                               "framing can fail on the other; keep the camera fixed or record enough of each.",
                               episodes=sorted(e for g in views["groups"][1:] for e in g),
                               details={"video_key": vkey, "groups": views["ranges"]}))
        rep = rv.check_video_batch([], records=records, workers=1)
        ep_of = {r["path"]: r["episode"] for r in records}
        by_check: dict[str, dict[int, float]] = {}
        decode_err = dict(missing)
        for f in rep["files"]:
            e = ep_of[f["path"]]
            for i in f["issues"]:
                if i["check"] not in FULL_CHECKS:
                    continue
                cid, floor, _ = FULL_CHECKS[i["check"]]
                if cid == "video.decode_error":
                    decode_err[e] = i["message"]
                    continue
                by_check.setdefault(i["check"], {})[e] = by_check.get(i["check"], {}).get(e, 0.0) + (i.get("seconds") or 0.0)
        for check, per_ep in by_check.items():
            cid, sev, what = FULL_CHECKS[check]
            if check == "video.blur":   # warn only where blur takes a large share of the episode
                sev = WARNING if any(v > 0.1 * max(1e-9, _dur(rep, ep_of, e)) for e, v in per_ep.items()) else INFO
            total = sum(per_ep.values())
            out.append(Finding(cid, sev, f"camera '{vkey}': {what} in {_eps(len(per_ep))} ({total:.0f} s in all)",
                               episodes=sorted(per_ep), details={"video_key": vkey,
                                                                 "seconds_by_episode": {str(k): round(v, 1) for k, v in per_ep.items()}}))
        if decode_err:
            out.append(Finding("video.decode_error", CRITICAL,
                               f"camera '{vkey}': video for {_eps(len(decode_err))} is missing, can't be decoded or ends early",
                               episodes=sorted(decode_err), details={"video_key": vkey,
                                                                     "errors": {str(k): v for k, v in list(decode_err.items())[:20]}}))
        for d in rep["duplicates"]:
            if d["kind"] == "loop_within_file":
                continue
            a, b = ep_of.get(d["a"]), ep_of.get(d["b"])
            if a is None or b is None:
                continue
            ev = d.get("evidence") or {}
            dup_pairs.setdefault((a, b), []).append({"camera": vkey, "kind": d["kind"], "overlap_s": d.get("overlap_s"),
                                                     "share_of_shorter": d.get("share_of_shorter"),
                                                     "motion_corr": ev.get("motion_corr")})
    # A duplicated episode is duplicated in every camera (they record together). A match in one camera of
    # a multi-camera dataset, or one covering a small part of the episode, is only a lead.
    n_cams = len(cameras or ds.video_keys)
    confirmed, leads = [], []
    for (a, b), hits in sorted(dup_pairs.items()):
        cams = {h["camera"] for h in hits}
        share = max((h.get("share_of_shorter") or (1.0 if h["kind"] in rv.STRONG_DUP else 0.0)) for h in hits)
        entry = {"original": a, "copy": b, "cameras": sorted(cams), "kind": hits[0]["kind"],
                 "overlap_s": max((h.get("overlap_s") or 0) for h in hits), "share_of_shorter": share,
                 "motion_corr": max((h.get("motion_corr") or 0) for h in hits)}
        if (len(cams) >= min(2, n_cams)) and share >= 0.5:
            confirmed.append(entry)
        else:
            leads.append(entry)
    if confirmed:
        eps_c = sorted({e["copy"] for e in confirmed})
        out.append(Finding("video.duplicate_episode", WARNING,
                           f"{_eps(len(eps_c))} repeat another episode's footage in every camera checked: the same "
                           "movements at the same moments (a copy from a merge or upload mistake, or a near-identical "
                           "recording that adds little)",
                           episodes=eps_c, details={"pairs": confirmed[:50]}))
    if leads:
        out.append(Finding("video.possible_duplicate", INFO,
                           f"{len(leads)} pair(s) of episodes share a stretch of matching footage in one camera only, "
                           "or over less than half an episode; worth a look, not counted", details={"pairs": leads[:50]}))
    return out


VIEW_SAME_CORR = 0.6     # correlation of two episodes' background thumbnails above which they share a viewpoint
VIEW_STABLE_STD = 8.0    # a thumbnail pixel is background if it varies less than this (grey levels) within an episode
VIEW_MIN_STABLE = 0.25   # share of pixels that must be background in both episodes to compare them


def viewpoints(records: list[dict]) -> dict | None:
    """Groups of episodes filmed from clearly different camera positions. Each episode's background is the
    median of its 2 fps thumbnails, and only the pixels that stay put within the episode count (the room
    for a fixed camera; just the gripper fingers and lens edge for a hand-held or wrist camera, which do
    not change when the operator walks somewhere else). Episodes join the group whose first episode's
    background they correlate with best on the pixels that are background in both, if that correlation is
    at least 0.6. Reported only when the groups are few, each holds at least three episodes and a tenth
    of the dataset, and each is a contiguous run of episode numbers (a camera moved between sessions)."""
    bgs, masks = {}, {}
    for r in records:
        th = ((r.get("dec") or {}).get("samples") or {}).get("thumb")
        if th is not None and len(th) >= 4:
            a = th.astype(np.float32)
            bgs[r["episode"]] = np.median(a, axis=0).ravel()
            masks[r["episode"]] = (a.std(axis=0) < VIEW_STABLE_STD).ravel()
    if len(bgs) < 6:
        return None
    # a camera that moves during the episodes has little background: if most episodes are like that,
    # a "different viewpoint" would just be a different place the camera was carried to
    if np.median([m.mean() for m in masks.values()]) < VIEW_MIN_STABLE:
        return None

    def corr(e: int, h: int) -> float:
        m = masks[e] & masks[h]
        if m.mean() < VIEW_MIN_STABLE:
            return float("nan")
        a, b = bgs[e][m], bgs[h][m]
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float(np.mean(a * b))

    groups: list[list[int]] = []
    heads: list[int] = []
    unsure = 0
    for e in sorted(bgs):
        cs = [corr(e, h) for h in heads]
        if cs and all(np.isnan(c) for c in cs):
            unsure += 1
            continue
        cs = [(-2.0 if np.isnan(c) else c) for c in cs]
        if cs and max(cs) >= VIEW_SAME_CORR:
            groups[int(np.argmax(cs))].append(e)
        else:
            groups.append([e])
            heads.append(e)
    n = len(bgs) - unsure
    if unsure > 0.1 * len(bgs):
        return None
    big = [g for g in groups if len(g) >= max(3, 0.1 * n)]
    if len(big) < 2 or len(groups) > 4 or sum(len(g) for g in big) < 0.9 * n:
        return None
    eps_sorted = sorted(e for g in groups for e in g)
    for g in big:   # contiguous in episode order (allowing the episodes we could not measure)
        lo, hi = eps_sorted.index(g[0]), eps_sorted.index(g[-1])
        if hi - lo + 1 != len(g):
            return None
    of = {e: k for k, g in enumerate(groups) for e in g}
    return {"groups": big, "of": of, "ranges": [f"episodes {g[0]}-{g[-1]}" for g in big]}


def _dur(rep: dict, ep_of: dict, e: int) -> float:
    for f in rep["files"]:
        if ep_of.get(f["path"]) == e:
            return f["duration_s"] or 0.0
    return 0.0
