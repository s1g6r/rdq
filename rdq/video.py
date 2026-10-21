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
SIM_TAGS = {"simulation", "simulated", "sim", "mujoco", "robosuite", "isaac", "isaacsim", "isaac-lab", "isaaclab",
            "maniskill", "robotwin", "robocasa", "libero", "pybullet", "gazebo", "metaworld", "meta-world",
            "genesis", "sapien", "habitat", "synthetic"}


def is_simulated(ds: Dataset) -> bool:
    """True when the dataset card's tags, or its robot type, say the footage is rendered. Rendered footage
    has no sensor noise, so a still scene repeats frames exactly, which in real footage means a stalled feed."""
    import re

    words: set[str] = set()
    readme = Path(ds.root) / "README.md"
    if readme.exists():
        text = readme.read_text(errors="replace")
        m = re.match(r"---\s*\n(.*?)\n---", text, re.S)
        if m:
            try:
                import yaml

                meta = yaml.safe_load(m.group(1)) or {}
                tags = meta.get("tags") or []
                words |= {str(t).strip().lower() for t in (tags if isinstance(tags, list) else [tags])}
            except Exception:  # noqa: BLE001 - a malformed card header just means no tags
                pass
    robot = str((ds.info or {}).get("robot_type") or "").lower()
    words |= set(re.split(r"[^a-z0-9]+", robot))
    return bool(words & SIM_TAGS) or robot.endswith("_sim") or robot.startswith("sim_")


ROBOT_STILL = 0.05   # the robot "is still" during a stretch when it ranges less than 5% of its whole-episode range


def robot_moving_during(ds: Dataset, episode: int, spans: list) -> bool | None:
    """Whether the robot moved while a frozen stretch of video was recorded, from its own state log (or its
    actions when there is no state): True, False, or None when there's nothing to judge with.

    A picture that stops while the robot moves is a capture problem (a stalled feed or renderer). A picture
    that stops while the robot is still too is a pause, idle padding or a deliberate no-task clip."""
    from .checks import _extent

    ep = ds.episodes.get(episode)
    key = ds.state_key or ds.action_key
    if ep is None or not key or key not in ep.features or not spans:
        return None
    x = np.asarray(ep.features[key], dtype=float)
    if ep.timestamps is not None and len(ep.timestamps) == len(x):
        t = np.asarray(ep.timestamps, dtype=float) - float(ep.timestamps[0])
    elif ds.fps:
        t = np.arange(len(x)) / float(ds.fps)
    else:
        return None
    whole = _extent(x)
    if whole <= 1e-9:
        return False
    seen = False
    for a, b in spans:
        m = (t >= float(a) - 1e-6) & (t < float(b) + 1e-6)
        if m.sum() < 3:
            continue
        seen = True
        if _extent(x[m]) >= ROBOT_STILL * whole:
            return True
    return False if seen else None


COPY_CHECKS = ("video.near_duplicate", "video.trimmed_copy", "video.same_stream_remuxed", "video.identical_file",
               "video.partial_overlap", "video.loop_within_file")



def _covered(span, spans, tol: float = 0.5) -> bool:
    """Whether [a, b) lies inside the union of spans, give or take tol seconds at each end."""
    a, b = float(span[0]) + tol, float(span[1]) - tol
    if b <= a:
        return any(float(c) <= float(span[0]) + tol and float(d) >= float(span[1]) - tol for c, d in spans)
    pos = a
    for c, d in sorted((float(c), float(d)) for c, d in spans):
        if c > pos:
            break
        pos = max(pos, d)
        if pos >= b:
            return True
    return pos >= b


def _frozen_findings(ds: Dataset, pending: dict, decoded: dict, simulated: bool) -> list[Finding]:
    """Frozen stretches, split by what else was happening at the time (an idea from the HF forum: tell a
    camera fault from a deliberate idle clip by comparing the picture with the robot's own log).

    A camera fault: the robot kept moving, or another camera of the same episode kept changing.
    A pause: the robot was still and every other camera froze too (idle padding, a no-task clip)."""
    out = []
    for vkey, (per_ep, spans_of, _eps_decoded) in pending.items():
        verdict = {}
        for e in per_ep:
            spans = spans_of.get(e, [])
            other_live = False
            for c, eps_c in decoded.items():   # another camera that recorded this episode and didn't freeze then
                if c == vkey or e not in eps_c or not spans:
                    continue
                c_spans = pending[c][1].get(e, []) if c in pending else []
                if not all(_covered(sp, c_spans) for sp in spans):
                    other_live = True
            robot = robot_moving_during(ds, e, spans)
            verdict[e] = "moving" if (robot is True or other_live) else ("still" if robot is False else "unknown")
        text = {
            "moving": ("the image stops changing while the robot or another camera keeps moving (the renderer or recording stalled)"
                       if simulated else
                       "the image stops changing while the robot or another camera keeps moving (a stalled or dropped camera feed)"),
            "unknown": ("the image stops changing for a while; the footage is simulated, so this is most likely a still "
                        "scene (rendered frames have no sensor noise), not a stalled feed") if simulated
            else FULL_CHECKS["video.frozen"][2],
            "still": ("the image stops changing while the robot and the other cameras are still too (a pause, idle "
                      "padding or a deliberate no-task clip rather than a camera fault)") if simulated else
            ("the image repeats exactly while the robot and the other cameras are still too: little is lost for "
             "training, though identical frames from a real camera mean the feed repeated its last frame"),
        }
        sev = {"moving": WARNING, "unknown": INFO if simulated else WARNING, "still": INFO}
        for v in ("moving", "unknown", "still"):
            eps_g = sorted(e for e in per_ep if verdict[e] == v)
            if not eps_g:
                continue
            total = sum(per_ep[e] for e in eps_g)
            out.append(Finding("video.frozen_span", sev[v], f"camera '{vkey}': {text[v]} in {_eps(len(eps_g))} ({total:.0f} s in all)",
                               episodes=eps_g, details={"video_key": vkey, "robot_or_other_camera": v,
                                                        "seconds_by_episode": {str(k): round(per_ep[k], 1) for k in eps_g}}))
    return out

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
    simulated = is_simulated(ds)
    cam_bgs: dict[str, dict[int, np.ndarray]] = {}
    frozen_pending: dict[str, tuple] = {}
    decoded_eps: dict[str, set] = {}
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
            fdur = ((measured[str(path)].get("dec") or {}).get("decoded_duration_s") or 0.0)
            if fdur and start >= fdur - 1e-3:   # the file ends before this episode begins (lerobot/berkeley_rpt 409, 829)
                missing[idx] = (f"no footage: {path.name} ends at {fdur:.1f} s, where this episode should start "
                                f"({start:.1f}-{end:.1f} s)")
                continue
            r = rv.segment_record(measured[str(path)], start, end, f"episode {idx}")
            r["contributor"] = vkey
            r["episode"] = idx
            records.append(r)
        if not records:
            if missing:
                out.append(Finding("video.decode_error", CRITICAL, f"camera '{vkey}': video missing for {_eps(len(missing))}",
                                   episodes=sorted(missing), details={"video_key": vkey}))
            continue
        decoded_eps[vkey] = {r["episode"] for r in records}
        cam_bgs[vkey] = episode_backgrounds(records)
        views = viewpoints(records)
        if views:
            for r in records:   # sharpness is judged per viewpoint; episodes too short to place keep the camera's norm
                g = views["of"].get(r["episode"])
                if g is not None:
                    r["contributor"] = f"{vkey}#view{g}"
            out.append(Finding("video.viewpoint_change", WARNING,
                               f"camera '{vkey}': the camera was moved between episodes, so the dataset mixes "
                               f"{len(views['groups'])} viewpoints ({'; '.join(views['ranges'])}). A policy trained on one "
                               "framing can fail on the other; keep the camera fixed or record enough of each.",
                               episodes=sorted(e for g in views["groups"][1:] for e in g),
                               details={"video_key": vkey, "groups": views["ranges"]}))
        rep = rv.check_video_batch([], records=records, workers=1)
        ep_of = {r["path"]: r["episode"] for r in records}
        by_check: dict[str, dict[int, float]] = {}
        frozen_spans: dict[int, list] = {}
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
                if i["check"] == "video.frozen":
                    frozen_spans.setdefault(e, []).extend(i.get("spans") or [])
        if "video.frozen" in by_check:   # judged after all cameras are decoded (see _frozen_findings)
            frozen_pending[vkey] = (by_check.pop("video.frozen"), frozen_spans, {r["episode"] for r in records})
        for check, per_ep in by_check.items():
            cid, sev, what = FULL_CHECKS[check]
            if check == "video.blur":   # warn only where blur takes a large share of the episode
                sev = WARNING if any(v > 0.1 * max(1e-9, _dur(rep, ep_of, e)) for e, v in per_ep.items()) else INFO
            total = sum(per_ep.values())
            out.append(Finding(cid, sev, f"camera '{vkey}': {what} in {_eps(len(per_ep))} ({total:.0f} s in all)",
                               episodes=sorted(per_ep), details={"video_key": vkey,
                                                                 "seconds_by_episode": {str(k): round(v, 1) for k, v in per_ep.items()}}))
        if decode_err:
            no_footage = [e for e, m in decode_err.items() if str(m).startswith("no footage")]
            if len(no_footage) == len(decode_err):
                what = (f"camera '{vkey}': {_eps(len(decode_err))} {'has' if len(decode_err) == 1 else 'have'} no footage: the video file ends before the "
                        "episode starts, so every frame requested for them is missing")
            else:
                what = f"camera '{vkey}': video for {_eps(len(decode_err))} is missing, can't be decoded or ends early"
            out.append(Finding("video.decode_error", CRITICAL, what,
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
    out += _frozen_findings(ds, frozen_pending, decoded_eps, simulated)
    out += judge_duplicates(dup_pairs, len(cameras or ds.video_keys), ds)
    for a, b, eps_sw in camera_swaps(cam_bgs):
        out.append(Finding("video.cameras_swapped", WARNING,
                           f"cameras '{a}' and '{b}' look swapped in {_eps(len(eps_sw))}: each shows what the other "
                           "shows in the rest of the dataset (camera devices that changed places between sessions). "
                           "A policy would get the wrong view in that slot.",
                           episodes=eps_sw, details={"cameras": [a, b]}))
    return out


SWAP_CORR = 0.6   # backgrounds this alike count as the same view
SWAP_MAX_RUNS = 3  # a swap spans at most this many runs of consecutive episodes (recording sessions)
SWAP_MIN_EPISODES = 3  # and at least this many episodes: one odd episode is usually a wrist camera's coincidence
SWAP_ALIKE = 0.45  # cameras whose views correlate this much in a typical episode (a stereo pair) are not judged
SWAP_MARGIN = 0.15  # a crossing episode matches the other camera's view clearly better than its own


def episode_backgrounds(records: list[dict]) -> dict[int, np.ndarray]:
    """Each episode's background: the median of its 2 fps thumbnails, normalised (mean 0, sd 1)."""
    out = {}
    for r in records:
        th = ((r.get("dec") or {}).get("samples") or {}).get("thumb")
        if th is not None and len(th) >= 4:
            b = np.median(th.astype(np.float32), axis=0).ravel()
            out[r["episode"]] = (b - b.mean()) / (b.std() + 1e-6)
    return out


def camera_swaps(cam_bgs: dict[str, dict[int, np.ndarray]]) -> list[tuple[str, str, list[int]]]:
    """Pairs of cameras whose streams trade places in some episodes (in at most three runs of consecutive episodes, i.e. sessions). Against a reference episode r, an episode
    "crosses" when one camera shows what the other camera showed at r while neither shows its own view from r
    (the other view may have changed too, e.g. a camera that was also moved). The episodes split into those that
    cross and those that keep r's assignment; the smaller side is reported as swapped. References are tried among
    episodes typical of each camera, and the split must cover most episodes. Cameras whose views look alike in a
    typical episode (a stereo pair) are not judged, and a crossing must match the other view clearly better."""
    out = []
    cams = sorted(cam_bgs)
    c = lambda x, y: float(np.mean(x * y))  # noqa: E731
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            A, B = cam_bgs[a], cam_bgs[b]
            eps = sorted(set(A) & set(B))
            if len(eps) < 6:
                continue
            # two cameras that see much the same picture (a stereo pair: 0.46-0.76 in lerobot/unitreeh1_*) can't
            # be told apart, so a swap between them is invisible and any "crossing" is noise around the threshold
            if float(np.median([c(A[e], B[e]) for e in eps])) >= SWAP_ALIKE:
                continue
            refs: list[int] = []
            for X in (A, B):                       # first episode of each view group, per camera
                heads: list[int] = []
                for e in eps:
                    if not any(c(X[e], X[h]) >= SWAP_CORR for h in heads):
                        heads.append(e)
                refs += heads[:6]
            best: list[int] = []
            for r in dict.fromkeys(refs):
                cross, keep = [], []
                for e in eps:
                    oa, ob = c(A[e], A[r]), c(B[e], B[r])
                    own_a, own_b = oa >= SWAP_CORR, ob >= SWAP_CORR
                    xc = max(c(A[e], B[r]), c(B[e], A[r]))
                    if not own_a and not own_b and xc >= SWAP_CORR and xc >= max(oa, ob) + SWAP_MARGIN:
                        cross.append(e)
                    elif own_a or own_b:
                        keep.append(e)
                if not cross or len(keep) < 3 or len(cross) + len(keep) < 0.8 * len(eps):
                    continue
                minority = cross if len(cross) < len(keep) else keep if len(keep) < len(cross) else []
                pos = [eps.index(e) for e in minority]
                runs = 1 + sum(1 for x, y in zip(pos, pos[1:]) if y != x + 1) if pos else 0
                # devices change order between sessions, so a real swap covers a few runs of consecutive episodes;
                # scattered matches come from a moving (wrist) camera that happens to point at the other view
                if len(minority) >= SWAP_MIN_EPISODES and runs <= SWAP_MAX_RUNS and len(minority) > len(best):
                    best = minority
            if best:
                out.append((a, b, best))
    return out

def _traj_distance(x: np.ndarray, y: np.ndarray, step: int) -> float:
    """Smallest mean absolute difference between two (T, D) trajectories over any alignment that overlaps at least
    half of the shorter one (sampled every `step` frames). A copy, even a trimmed one, scores about zero."""
    x, y = x[::step], y[::step]
    n = min(len(x), len(y))
    if n < 3:
        return float("nan")
    minov = max(3, n // 2)
    best = float("inf")
    for sh in range(-(len(y) - minov), len(x) - minov + 1):
        i0, j0 = max(0, sh), max(0, -sh)
        m = min(len(x) - i0, len(y) - j0)
        if m < minov:
            continue
        d = float(np.mean(np.abs(x[i0:i0 + m] - y[j0:j0 + m])))
        best = min(best, d)
    return best


def data_agreement(ds: Dataset, a: int, b: int) -> float | None:
    """How alike the robot data (state, else action) of episodes a and b is, relative to unrelated episode
    pairs: about 0 for a copy, about 1 for two separate recordings. None when there is no data to compare."""
    key = ds.state_key or ds.action_key
    feats = {e: ep.features.get(key) for e, ep in ds.episodes.items() if key and ep.features.get(key) is not None}
    if a not in feats or b not in feats or len(feats) < 4:
        return None
    sample = np.concatenate([feats[e] for e in list(feats)[:: max(1, len(feats) // 50)]])
    sd = sample.std(axis=0)
    sd[sd < 1e-9] = 1.0
    z = lambda e: np.asarray(feats[e], dtype=np.float64) / sd  # noqa: E731
    step = max(1, max(len(feats[a]), len(feats[b])) // 150)
    pair = _traj_distance(z(a), z(b), step)
    others = [e for e in sorted(feats) if e not in (a, b)]
    others = others[:: max(1, len(others) // 8)][:8]
    typical = [d for e in others for d in (_traj_distance(z(a), z(e), step), _traj_distance(z(b), z(e), step))
               if np.isfinite(d)]
    if not typical or not np.isfinite(pair) or np.median(typical) <= 1e-9:
        return None
    return float(pair / np.median(typical))


def judge_duplicates(dup_pairs: dict[tuple[int, int], list[dict]], n_cams: int,
                     ds: Dataset | None = None) -> list[Finding]:
    """Turn per-camera matches between episode pairs into findings. A duplicated episode is duplicated in
    every camera (they record together). A match in one camera of a multi-camera dataset, one covering a
    small part of the episode, or (with a single camera) one whose movements agree less than a copy's
    would, is only a lead."""
    from . import rawvideo as rv

    out: list[Finding] = []
    confirmed, leads = [], []
    for (a, b), hits in sorted(dup_pairs.items()):
        cams = {h["camera"] for h in hits}
        share = max((h.get("share_of_shorter") or (1.0 if h["kind"] in rv.STRONG_DUP else 0.0)) for h in hits)
        entry = {"original": a, "copy": b, "cameras": sorted(cams), "kind": hits[0]["kind"],
                 "overlap_s": max((h.get("overlap_s") or 0) for h in hits), "share_of_shorter": share,
                 "motion_corr": max((h.get("motion_corr") or 0) for h in hits)}
        strong = any(h["kind"] in ("identical_file", "same_stream_remuxed") for h in hits)
        # with a single camera there is no second view to confirm a copy, and takes of one task from a wrist
        # camera move alike; a copy from a merge or upload mistake matches almost exactly
        single_ok = n_cams >= 2 or strong or entry["motion_corr"] >= DUP_SINGLE_CAM_CORR
        # a copy duplicates the robot data too; a precise robot replaying one motion films near-identical takes
        # whose joint data still differs as much as any two recordings do
        agree = data_agreement(ds, a, b) if ds is not None and not strong else None
        if agree is not None:
            entry["data_agreement"] = round(agree, 2)
        data_ok = agree is None or agree <= DUP_DATA_RATIO
        # a few frames of a slow camera are not enough to tell a copy from scripted takes that start alike
        # (lerobot/uiuc_d3field, 1 fps: 8 frames of an empty table before the arm comes in, different objects)
        fps = (ds.fps if ds is not None else None) or 0
        frames_ok = strong or not fps or entry["overlap_s"] * fps >= DUP_MIN_FRAMES
        if (len(cams) >= min(2, n_cams)) and share >= 0.5 and single_ok and data_ok and frames_ok:
            confirmed.append(entry)
        else:
            leads.append(entry)
    if confirmed:
        eps_c = sorted({e["copy"] for e in confirmed})
        out.append(Finding("video.duplicate_episode", WARNING,
                           f"{_eps(len(eps_c))} {'repeats' if len(eps_c) == 1 else 'repeat'} another episode's footage in every camera checked: the same "
                           "movements at the same moments (a copy from a merge or upload mistake, or a near-identical "
                           "recording that adds little)",
                           episodes=eps_c, details={"pairs": confirmed[:50]}))
    if leads:
        out.append(Finding("video.possible_duplicate", INFO,
                           f"{len(leads)} pair(s) of episodes share a stretch of matching footage in one camera only, "
                           "or over less than half an episode, or (with a single camera) with movements that agree less than a "
                           "copy's would, or whose robot data differs as much as two separate recordings; worth a look, not counted", details={"pairs": leads[:50]}))
    return out


DUP_SINGLE_CAM_CORR = 0.97  # motion agreement a duplicate needs when the dataset has only one camera
DUP_MIN_FRAMES = 15  # a copy must match over at least this many frames of the dataset's own rate
DUP_DATA_RATIO = 0.3  # a duplicate's robot data must be at least this much closer than unrelated episodes' (0 = copy)
VIEW_SAME_CORR = 0.6     # correlation of two episodes' background thumbnails above which they share a viewpoint
VIEW_STABLE_STD = 8.0    # a thumbnail pixel is still if it varies less than this (grey levels) within an episode
VIEW_MIN_STABLE = 0.4    # a camera with less of its picture still than this (median episode) is moving: not judged


def viewpoints(records: list[dict]) -> dict | None:
    """Groups of episodes filmed from clearly different camera positions. Each episode's background is the
    median of its 2 fps thumbnails (what stays put while the arm moves). Episodes join the group whose first
    episode's background they correlate with best, if that correlation is at least 0.6. Reported only when the
    groups are few, each holds at least three episodes and a tenth of the dataset, and each is a contiguous run
    of episode numbers (a camera moved between sessions). A hand-held or wrist camera, where most of the
    picture changes within an episode (less than 40% of it still in the median episode), is not judged: its
    "background" is wherever the operator happened to point it."""
    bgs, still = {}, []
    for r in records:
        th = ((r.get("dec") or {}).get("samples") or {}).get("thumb")
        if th is not None and len(th) >= 4:
            a = th.astype(np.float32)
            b = np.median(a, axis=0).ravel()
            bgs[r["episode"]] = (b - b.mean()) / (b.std() + 1e-6)
            still.append(float((a.std(axis=0) < VIEW_STABLE_STD).mean()))
    if len(bgs) < 6 or np.median(still) < VIEW_MIN_STABLE:
        return None
    groups: list[list[int]] = []
    heads: list[np.ndarray] = []
    for e in sorted(bgs):
        cs = [float(np.mean(bgs[e] * h)) for h in heads]
        if cs and max(cs) >= VIEW_SAME_CORR:
            groups[int(np.argmax(cs))].append(e)
        else:
            groups.append([e])
            heads.append(bgs[e])
    big = [g for g in groups if len(g) >= max(3, 0.1 * len(bgs))]
    if len(big) < 2 or len(groups) > 4 or sum(len(g) for g in big) < 0.9 * len(bgs):
        return None
    eps_sorted = sorted(bgs)
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
