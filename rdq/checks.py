"""Quality checks for LeRobot datasets.

Every check returns a list of Finding objects. A finding is either about the
whole dataset (episodes == []) or about specific episodes. Thresholds live in
Thresholds so they can be tuned and reported alongside the results.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np

from .dataset import Dataset, Episode

CRITICAL = "critical"  # data is wrong or unusable for training
WARNING = "warning"  # data is suspicious and probably hurts training
INFO = "info"  # worth knowing, not necessarily a problem

PLACEHOLDER_TASKS = {
    "",
    "none",
    "null",
    "nan",
    "task",
    "default",
    "default task",
    "test",
    "todo",
    "my task",
    "your task",
    "task description",
    "a task",
    "placeholder",
    "n/a",
}


@dataclass
class Thresholds:
    fps_tolerance: float = 0.05  # median frame interval may differ from 1/fps by 5%
    gap_factor: float = 1.5  # an interval longer than 1.5 frames counts as a dropped frame
    short_episode_frames: int = 10
    short_episode_ratio: float = 0.2  # shorter than 20% of the median length
    robust_z: float = 4.0  # robust z-score above which an episode is an outlier
    idle_motion_ratio: float = 0.05  # moves less than 5% of the median episode
    idle_edge_ratio: float = 0.4  # more than 40% of frames idle at start or end
    spike_fraction: float = 0.3  # a glitch moves a channel by 30% of its full range and back
    spike_window: int = 7  # running-median window; removes glitches of up to 3 frames
    stats_rel_tolerance: float = 0.05  # stats.json may differ from recomputed by 5% of range
    lag_max: int = 15  # frames searched when aligning action with state
    lag_deviation: int = 4  # frames an episode's lag may differ from the dataset median
    min_episodes_for_outliers: int = 8


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    episodes: list[int] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _eps(n: int) -> str:
    return f"{n} episode" if n == 1 else f"{n} episodes"


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        # Fall back to the mean absolute deviation so a mostly-constant column
        # still gives sensible scores instead of division by zero.
        scale = np.nanmean(np.abs(values - median))
        if not np.isfinite(scale) or scale < 1e-12:
            return np.zeros_like(values)
    return (values - median) / scale


def _declared_dim(feature: dict) -> int | None:
    shape = (feature or {}).get("shape")
    if not shape:
        return None
    try:
        return int(np.prod([int(s) for s in shape]))
    except (TypeError, ValueError):
        return None


def _flatten(value) -> list:
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_flatten(v))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flatten(v) if isinstance(v, (list, tuple, dict)) else [v])
        return out
    return [value]


def _names_list(feature: dict) -> list | None:
    """Per-channel names, flattened.

    LeRobot datasets write names as a list, a list wrapped in another list
    (older v2.0 convention), or a dict such as {"motors": [...]}. All are valid.
    """
    names = (feature or {}).get("names")
    if names is None:
        return None
    flat = _flatten(names)
    return flat or None


def names_status(feature: dict) -> str:
    """ok | unnamed | placeholder (one label for many channels) | mismatch."""
    declared = _declared_dim(feature)
    names = _names_list(feature)
    if names is None:
        return "unnamed"
    if declared is None or len(names) == declared:
        return "ok"
    shape = (feature or {}).get("shape") or []
    try:
        last = int(shape[-1])
    except (TypeError, ValueError, IndexError):
        last = None
    if last is not None and len(names) == last:
        return "ok"  # e.g. an action chunk of shape [20, 14] named by its 14 channels
    if len(names) == 1 and declared > 1:
        return "placeholder"
    return "mismatch"


# --------------------------------------------------------------------------- #
# Dataset-level checks
# --------------------------------------------------------------------------- #


def check_info(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    for key in ("codebase_version", "fps", "features"):
        if key not in ds.info:
            out.append(Finding("info.missing_key", CRITICAL, f"meta/info.json is missing '{key}'"))
    if ds.fps is not None and ds.fps <= 0:
        out.append(Finding("info.bad_fps", CRITICAL, f"fps is {ds.fps}, which is impossible"))
    if not ds.info.get("robot_type"):
        out.append(
            Finding(
                "info.no_robot_type",
                INFO,
                "robot_type is not set, so nobody can tell which robot recorded this data",
            )
        )
    for msg in ds.load_warnings:
        out.append(Finding("info.load_warning", WARNING, msg))
    return out


def check_feature_shapes(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    for key in ds.numeric_keys:
        feature = ds.features.get(key, {})
        declared = _declared_dim(feature)
        names = _names_list(feature)
        status = names_status(feature)
        if status == "mismatch":
            out.append(
                Finding(
                    "schema.names_shape_mismatch",
                    WARNING,
                    f"'{key}' declares shape {declared} but lists {len(names)} names",
                    details={"feature": key, "shape": declared, "names": len(names)},
                )
            )
        elif status == "placeholder" and key in (ds.action_key, ds.state_key):
            out.append(
                Finding(
                    "schema.unnamed_channels",
                    INFO,
                    f"'{key}' has {declared} channels but one generic name ({names[0]!r}), so nobody "
                    "can tell which joint each value is without outside documentation",
                    details={"feature": key, "shape": declared, "name": str(names[0])},
                )
            )
        actual_dims = {ep.features[key].shape[1] for ep in ds.episodes.values() if key in ep.features}
        if declared is not None and actual_dims and actual_dims != {declared}:
            out.append(
                Finding(
                    "schema.shape_mismatch",
                    CRITICAL,
                    f"'{key}' is declared with {declared} values per frame but the data has "
                    f"{sorted(actual_dims)}",
                    details={"feature": key, "declared": declared, "actual": sorted(actual_dims)},
                )
            )
    a, s = ds.action_key, ds.state_key
    if a and s:
        da = _declared_dim(ds.features.get(a, {}))
        dsd = _declared_dim(ds.features.get(s, {}))
        if da and dsd and da != dsd:
            out.append(
                Finding(
                    "schema.action_state_dims_differ",
                    INFO,
                    f"action has {da} values but {s} has {dsd}; fine for some robots, but it "
                    "matters when mixing this dataset with others",
                    details={"action": da, "state": dsd},
                )
            )
    if not a:
        out.append(Finding("schema.no_action", CRITICAL, "no 'action' feature, so a policy cannot learn from it"))
    if not s:
        out.append(Finding("schema.no_state", WARNING, "no observation.state feature"))
    return out


def check_counts(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    actual_eps = len(ds.episodes)
    actual_frames = sum(ep.length for ep in ds.episodes.values())
    declared_eps = ds.info.get("total_episodes")
    declared_frames = ds.info.get("total_frames")
    truncated = any("max_frames" in w for w in ds.load_warnings)
    if not truncated:
        if declared_eps is not None and declared_eps != actual_eps:
            out.append(
                Finding(
                    "counts.episodes",
                    WARNING,
                    f"info.json says {declared_eps} episodes but the data has {actual_eps}",
                    details={"declared": declared_eps, "actual": actual_eps},
                )
            )
        if declared_frames is not None and declared_frames != actual_frames:
            out.append(
                Finding(
                    "counts.frames",
                    WARNING,
                    f"info.json says {declared_frames} frames but the data has {actual_frames}",
                    details={"declared": declared_frames, "actual": actual_frames},
                )
            )
    if ds.episodes:
        idx = sorted(ds.episodes)
        expected = set(range(idx[0], idx[-1] + 1))
        gaps = sorted(expected - set(idx))
        if gaps and not truncated:
            out.append(
                Finding(
                    "counts.episode_gaps",
                    WARNING,
                    f"{len(gaps)} episode indices are missing from the data",
                    details={"missing": gaps[:50]},
                )
            )
    if ds.episodes_meta is not None and "length" in ds.episodes_meta.columns:
        bad = []
        for ep in ds.episodes.values():
            meta_len = ep.meta.get("length")
            if meta_len is not None and int(meta_len) != ep.length:
                bad.append(ep.index)
        if bad:
            out.append(
                Finding(
                    "counts.length_mismatch",
                    CRITICAL,
                    f"{_eps(len(bad))} have a different number of frames than the metadata says",
                    episodes=bad,
                )
            )
    return out


def check_files(ds: Dataset, th: Thresholds) -> list[Finding]:
    if not ds.orphan_files:
        return []
    return [
        Finding(
            "files.orphaned",
            INFO,
            f"{len(ds.orphan_files)} data file(s) are not referenced by the episode metadata, probably "
            "left over from an earlier version; LeRobot ignores them, but other tools that read every file will not",
            details={"files": ds.orphan_files[:20]},
        )
    ]


def check_tasks(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    if not ds.tasks:
        out.append(Finding("tasks.missing", CRITICAL, "there is no task list, so language-conditioned policies get no instruction"))
        return out
    placeholder = {i: t for i, t in ds.tasks.items() if t.strip().lower().strip(".") in PLACEHOLDER_TASKS}
    if placeholder:
        eps = [
            ep.index
            for ep in ds.episodes.values()
            if ep.task_index is not None and set(np.unique(ep.task_index)) & set(placeholder)
        ]
        if len(eps) == len(ds.episodes):
            # Every episode shares the bad label: a dataset-level labelling problem, not a
            # reason to drop episodes whose motion data may be perfectly fine.
            eps = []
        out.append(
            Finding(
                "tasks.placeholder",
                WARNING,
                f"{len(placeholder)} task description(s) are empty or placeholders: "
                + ", ".join(repr(t) for t in list(placeholder.values())[:5]),
                episodes=eps,
                details={"tasks": list(placeholder.values())},
            )
        )
    short = {i: t for i, t in ds.tasks.items() if t not in placeholder.values() and 0 < len(t.split()) <= 2}
    if short:
        out.append(
            Finding(
                "tasks.too_short",
                INFO,
                f"{len(short)} task description(s) are one or two words, which gives a language "
                "model little to go on: " + ", ".join(repr(t) for t in list(short.values())[:5]),
                details={"tasks": list(short.values())},
            )
        )
    unknown_eps = []
    for ep in ds.episodes.values():
        if ep.task_index is not None and not set(np.unique(ep.task_index)) <= set(ds.tasks):
            unknown_eps.append(ep.index)
    if unknown_eps:
        out.append(
            Finding(
                "tasks.unknown_index",
                CRITICAL,
                f"{_eps(len(unknown_eps))} point to task indices that are not in the task list",
                episodes=unknown_eps,
            )
        )
    return out


def check_values(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    nan_eps: dict[str, list[int]] = {}
    for ep in ds.episodes.values():
        for key, arr in ep.features.items():
            if not np.all(np.isfinite(arr)):
                nan_eps.setdefault(key, []).append(ep.index)
    for key, eps in nan_eps.items():
        out.append(
            Finding(
                "values.non_finite",
                CRITICAL,
                f"'{key}' contains NaN or infinite values in {_eps(len(eps))}",
                episodes=eps,
                details={"feature": key},
            )
        )
    for key in (ds.action_key, ds.state_key):
        if not key:
            continue
        arrays = [ep.features[key] for ep in ds.episodes.values() if key in ep.features]
        if not arrays:
            continue
        widths = {a.shape[1] for a in arrays}
        if len(widths) != 1:
            continue
        stacked = np.concatenate(arrays)
        stacked = stacked[np.all(np.isfinite(stacked), axis=1)]
        if len(stacked) < 2:
            continue
        constant = [int(d) for d in np.where(np.ptp(stacked, axis=0) < 1e-9)[0]]
        if constant:
            names = _names_list(ds.features.get(key, {}))
            labels = [names[d] if names and d < len(names) else f"dim {d}" for d in constant]
            out.append(
                Finding(
                    "values.dead_dimension",
                    INFO,
                    f"'{key}' has {len(constant)} channel(s) that never change across the whole "
                    f"dataset ({', '.join(map(str, labels[:6]))}): unused padding, an unused joint, "
                    "or a disconnected sensor or motor",
                    details={"feature": key, "dims": constant},
                )
            )
    return out


def check_stats(ds: Dataset, th: Thresholds) -> list[Finding]:
    """Compare meta/stats.json with statistics recomputed from the data.

    Policies normalise inputs with these stats, so stale stats silently distort
    training even when every frame is fine.
    """
    out = []
    truncated = any("max_frames" in w for w in ds.load_warnings)
    if ds.stats is None:
        if not ds.has_episode_stats:
            out.append(Finding("stats.missing", WARNING, "no normalisation statistics at all (no stats.json or per-episode stats)"))
        return out
    if truncated:
        return out
    for key in (ds.action_key, ds.state_key):
        if not key or key not in ds.stats:
            if key:
                out.append(Finding("stats.feature_missing", WARNING, f"stats.json has no entry for '{key}'"))
            continue
        arrays = [ep.features[key] for ep in ds.episodes.values()]
        if len({a.shape[1] for a in arrays}) != 1:
            continue
        data = np.concatenate(arrays)
        data = data[np.all(np.isfinite(data), axis=1)]
        if not len(data):
            continue
        entry = ds.stats[key]
        span = np.ptp(data, axis=0)
        span = np.where(span < 1e-9, 1.0, span)
        problems = {}
        recomputed = {"min": data.min(0), "max": data.max(0), "mean": data.mean(0)}
        for stat, value in recomputed.items():
            if stat not in entry:
                continue
            stored = np.asarray(entry[stat], dtype=np.float64).reshape(-1)
            if stored.shape != value.shape:
                problems[stat] = f"stored {stored.shape[0]} values, data has {value.shape[0]}"
                continue
            rel = np.abs(stored - value) / span
            if np.nanmax(rel) > th.stats_rel_tolerance:
                problems[stat] = f"off by up to {np.nanmax(rel):.0%} of the value range"
        if problems:
            out.append(
                Finding(
                    "stats.stale",
                    WARNING,
                    f"stats.json for '{key}' does not match the data ("
                    + "; ".join(f"{k}: {v}" for k, v in problems.items())
                    + "). Normalisation during training will be wrong.",
                    details={"feature": key, "problems": problems},
                )
            )
    return out


def check_duplicates(ds: Dataset, th: Thresholds) -> list[Finding]:
    key = ds.action_key or ds.state_key
    if not key:
        return []
    seen: dict[str, int] = {}
    dupes: list[tuple[int, int]] = []
    for ep in ds.episodes.values():
        arr = ep.features.get(key)
        if arr is None or len(arr) < 2:
            continue
        digest = hashlib.sha1(np.ascontiguousarray(np.round(arr, 6)).tobytes()).hexdigest()
        if digest in seen:
            dupes.append((seen[digest], ep.index))
        else:
            seen[digest] = ep.index
    if not dupes:
        return []
    return [
        Finding(
            "episodes.duplicate",
            WARNING,
            f"{len(dupes)} episode(s) are exact copies of an earlier episode",
            episodes=[b for _, b in dupes],
            details={"pairs": dupes[:50]},
        )
    ]


def check_video_alignment(ds: Dataset, th: Thresholds) -> list[Finding]:
    """Check that each episode's video span matches its data span (v3.0 metadata)."""
    if not ds.video_keys or ds.episodes_meta is None or not ds.fps:
        return []
    out = []
    frame = 1.0 / ds.fps
    for vkey in ds.video_keys:
        start_col, end_col = f"videos/{vkey}/from_timestamp", f"videos/{vkey}/to_timestamp"
        if start_col not in ds.episodes_meta.columns or end_col not in ds.episodes_meta.columns:
            continue
        bad = []
        worst = 0.0
        for ep in ds.episodes.values():
            start, end = ep.meta.get(start_col), ep.meta.get(end_col)
            if start is None or end is None:
                continue
            video_seconds = float(end) - float(start)
            data_seconds = ep.length / ds.fps
            diff = abs(video_seconds - data_seconds)
            if diff > 1.5 * frame:
                bad.append(ep.index)
                worst = max(worst, diff)
        if bad:
            out.append(
                Finding(
                    "video.span_mismatch",
                    WARNING,
                    f"camera '{vkey}': in {_eps(len(bad))} the video segment and the recorded "
                    f"data differ in length (worst {worst:.2f}s); common after trimming or editing "
                    "episodes, and frames may no longer line up with actions",
                    episodes=bad,
                    details={"video_key": vkey, "worst_seconds": round(worst, 3)},
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Episode-level checks
# --------------------------------------------------------------------------- #


def check_timestamps(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    non_monotonic, fps_off, gaps, frame_idx_bad = [], [], {}, []
    for ep in ds.episodes.values():
        if ep.frame_index is not None and len(ep.frame_index) > 1:
            if not np.array_equal(ep.frame_index, np.arange(ep.frame_index[0], ep.frame_index[0] + ep.length)):
                frame_idx_bad.append(ep.index)
        if ep.timestamps is None or len(ep.timestamps) < 3:
            continue
        dt = np.diff(ep.timestamps)
        if np.any(dt <= 0):
            non_monotonic.append(ep.index)
            continue
        if ds.fps:
            expected = 1.0 / ds.fps
            if abs(np.median(dt) - expected) / expected > th.fps_tolerance:
                fps_off.append(ep.index)
            n_gaps = int(np.sum(dt > th.gap_factor * expected))
            if n_gaps:
                gaps[ep.index] = n_gaps
    if non_monotonic:
        out.append(
            Finding(
                "time.non_monotonic",
                CRITICAL,
                f"{_eps(len(non_monotonic))} have timestamps that go backwards or repeat",
                episodes=non_monotonic,
            )
        )
    if fps_off:
        out.append(
            Finding(
                "time.fps_mismatch",
                WARNING,
                f"{_eps(len(fps_off))} were not recorded at the declared {ds.fps:g} fps",
                episodes=fps_off,
            )
        )
    if gaps:
        total = sum(gaps.values())
        out.append(
            Finding(
                "time.dropped_frames",
                WARNING,
                f"{_eps(len(gaps))} have gaps in time ({total} dropped-frame gaps in total)",
                episodes=sorted(gaps),
                details={"gaps_per_episode": {str(k): v for k, v in sorted(gaps.items())[:100]}},
            )
        )
    if frame_idx_bad:
        out.append(
            Finding(
                "time.frame_index",
                WARNING,
                f"{_eps(len(frame_idx_bad))} have frame_index values that skip or repeat",
                episodes=frame_idx_bad,
            )
        )
    return out


def check_lengths(ds: Dataset, th: Thresholds) -> list[Finding]:
    eps = list(ds.episodes.values())
    if not eps:
        return []
    lengths = np.array([ep.length for ep in eps])
    median = float(np.median(lengths))
    short = [
        ep.index
        for ep in eps
        if ep.length < th.short_episode_frames or ep.length < th.short_episode_ratio * median
    ]
    out = []
    if short:
        out.append(
            Finding(
                "episodes.too_short",
                WARNING,
                f"{_eps(len(short))} are much shorter than usual (median is {median:.0f} frames); "
                "often an aborted recording",
                episodes=short,
            )
        )
    if len(eps) >= th.min_episodes_for_outliers:
        z = _robust_z(np.log(lengths))
        long = [ep.index for ep, zi in zip(eps, z) if zi > th.robust_z]
        if long:
            out.append(
                Finding(
                    "episodes.too_long",
                    INFO,
                    f"{_eps(len(long))} are far longer than usual; check for recordings left running",
                    episodes=long,
                )
            )
    return out


def _motion(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    finite = arr[np.all(np.isfinite(arr), axis=1)]
    if len(finite) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(finite, axis=0), axis=1)))


def _medfilt(arr: np.ndarray, window: int) -> np.ndarray:
    """Running median along time, edges padded, same length as the input."""
    from numpy.lib.stride_tricks import sliding_window_view

    half = window // 2
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    return np.median(sliding_window_view(padded, window, axis=0), axis=-1)


def _extent(arr: np.ndarray, window: int = 5) -> float:
    """How far the trajectory ranges, ignoring jitter and short glitches.

    A running median removes glitches of up to two frames, and taking the
    maximum (not a percentile) keeps short but real movements that are
    followed by a long wait.
    """
    finite = arr[np.all(np.isfinite(arr), axis=1)]
    if len(finite) < 2:
        return 0.0
    if len(finite) >= window:
        from numpy.lib.stride_tricks import sliding_window_view

        half = window // 2
        padded = np.pad(finite, ((half, half), (0, 0)), mode="edge")  # keep moves in the first/last frames
        finite = np.median(sliding_window_view(padded, window, axis=0), axis=-1)
    # Range of the filtered path: the median filter keeps real edges but drops 1-2 frame spikes.
    return float(np.linalg.norm(np.ptp(finite, axis=0)))


def check_idle(ds: Dataset, th: Thresholds) -> list[Finding]:
    key = ds.action_key or ds.state_key
    if not key:
        return []
    eps = [ep for ep in ds.episodes.values() if key in ep.features and ep.length > 2]
    if len(eps) < 3:
        return []
    # Spatial extent, not path length: sensor jitter adds path length without moving anywhere.
    motions = np.array([_extent(ep.features[key]) for ep in eps])
    median = float(np.median(motions))
    out = []
    if median > 0:
        idle = [ep.index for ep, m in zip(eps, motions) if m < th.idle_motion_ratio * median]
        if idle:
            out.append(
                Finding(
                    "episodes.idle",
                    CRITICAL,
                    f"{_eps(len(idle))} barely move (less than {th.idle_motion_ratio:.0%} of a "
                    "typical episode); the robot was probably not doing the task",
                    episodes=idle,
                )
            )
    # Idle padding at the start or end of an episode.
    all_steps = np.concatenate(
        [np.linalg.norm(np.diff(ep.features[key], axis=0), axis=1) for ep in eps]
    )
    all_steps = all_steps[np.isfinite(all_steps)]
    if len(all_steps) == 0:
        return out
    still = max(float(np.percentile(all_steps, 20)) * 0.25, 1e-9)
    padded = []
    for ep in eps:
        steps = np.linalg.norm(np.diff(ep.features[key], axis=0), axis=1)
        moving = np.where(steps > still)[0]
        if len(moving) == 0:
            continue
        head = moving[0] / len(steps)
        tail = (len(steps) - 1 - moving[-1]) / len(steps)
        if head > th.idle_edge_ratio or tail > th.idle_edge_ratio:
            padded.append(ep.index)
    if padded:
        out.append(
            Finding(
                "episodes.idle_padding",
                WARNING,
                f"{_eps(len(padded))} spend more than {th.idle_edge_ratio:.0%} of their frames "
                "standing still at the start or end",
                episodes=padded,
            )
        )
    return out


def _action_tracks_state(ds: Dataset, th: Thresholds, sample: int = 40) -> bool:
    """True when actions are absolute positions that the measured state follows.

    Delta or velocity actions jump from frame to frame by design, so smoothness
    checks must not be applied to them.
    """
    a, s = ds.action_key, ds.state_key
    if not a or not s:
        return False
    eps = [
        ep
        for ep in ds.episodes.values()
        if a in ep.features
        and s in ep.features
        and ep.features[a].shape == ep.features[s].shape
        and ep.length > 2 * th.lag_max
        and np.all(np.isfinite(ep.features[a]))
        and np.all(np.isfinite(ep.features[s]))
    ][:sample]
    if not eps:
        return False
    corrs = []
    for ep in eps:
        errors, _ = _lag_errors(ep.features[a], ep.features[s], th.lag_max)
        if len(errors) and np.isfinite(errors).any():
            # Lowest alignment error of standardised channels: 0 = identical, ~2 = unrelated.
            corrs.append(1 - float(np.nanmin(errors)) / 2)
    return bool(corrs) and float(np.median(corrs)) >= 0.6


def check_spikes(ds: Dataset, th: Thresholds) -> list[Finding]:
    """Short out-and-back glitches.

    The signal is compared with its running median. A real step, like a gripper
    closing or an angle wrapping, survives a median filter and leaves no residual;
    a 1-3 frame dropout that jumps away and comes back does not. The threshold
    adapts to each dataset's normal jitter, and actions are only checked when
    they are absolute positions.
    """
    out = []
    keys = [ds.state_key]
    if _action_tracks_state(ds, th):
        keys.append(ds.action_key)
    for key in keys:
        if not key:
            continue
        eps = [
            ep
            for ep in ds.episodes.values()
            if key in ep.features and ep.length >= th.spike_window and np.all(np.isfinite(ep.features[key]))
        ]
        if not eps or len({ep.features[key].shape[1] for ep in eps}) != 1:
            continue
        values = np.concatenate([ep.features[key] for ep in eps])
        if len(values) < 20:
            continue
        span = np.percentile(values, 99, axis=0) - np.percentile(values, 1, axis=0)
        span = np.where(span < 1e-9, np.inf, span)
        residuals = {ep.index: np.abs(ep.features[key] - _medfilt(ep.features[key], th.spike_window)) for ep in eps}
        # Normal jitter: the median over episodes of each episode's 99th-percentile residual.
        # Pooling all frames would let the glitches themselves set the bar once they make up
        # more than 1% of frames (a few bad episodes with 3-frame dropouts is enough).
        typical = np.median(np.stack([np.percentile(r, 99, axis=0) for r in residuals.values()]), axis=0)
        limit = np.maximum(th.spike_fraction * span, 6.0 * typical)
        flagged = {}
        for idx, resid in residuals.items():
            n = int(np.sum(np.any(resid > limit, axis=1)))
            if n:
                flagged[idx] = n
        if flagged:
            out.append(
                Finding(
                    "episodes.spikes",
                    WARNING,
                    f"'{key}' has short glitches in {_eps(len(flagged))}: a channel jumps far outside its "
                    "normal motion for a few frames and comes back; usually a sensor glitch or a teleop dropout",
                    episodes=sorted(flagged),
                    details={"feature": key, "glitch_frames_per_episode": {str(k): v for k, v in list(flagged.items())[:100]}},
                )
            )
    return out


def _lag_errors(action: np.ndarray, state: np.ndarray, max_lag: int) -> tuple[np.ndarray, float]:
    """Mean squared error between action and state *velocities* at lags k = -max_lag..max_lag.

    Index i of the returned array corresponds to lag k = i - max_lag. Negative
    lags mean the recorded action trails the observation, the signature of an
    action stream written late.

    Velocities (two-frame differences), each channel scaled by its own RMS, are
    compared rather than positions: they ignore constant calibration offsets and
    gain differences between commanded and measured positions, and unlike
    mean-removed positions they still pin down the lag when the arm moves at a
    steady speed (a time shift of a straight-line motion is only an offset).
    Also returns the zero-lag position correlation, used to decide whether the
    action stream tracks the state at all.
    """
    active = (action.std(0) > 1e-6) & (state.std(0) > 1e-6)
    if not np.any(active) or len(action) < 8:
        return np.array([]), float("nan")
    a = action[:, active]
    s = state[:, active]
    w = 2
    va = a[w:] - a[:-w]
    vs = s[w:] - s[:-w]
    va = va / (np.sqrt((va**2).mean(0)) + 1e-12)
    vs = vs / (np.sqrt((vs**2).mean(0)) + 1e-12)
    n = len(va)
    errors = []
    for k in range(-max_lag, max_lag + 1):
        if abs(k) > n - 5:
            errors.append(np.inf)
            continue
        diff = vs[k:] - va[: n - k] if k >= 0 else vs[: n + k] - va[-k:]
        errors.append(float(np.mean(diff**2)))
    a = (a - a.mean(0)) / a.std(0)
    s = (s - s.mean(0)) / s.std(0)
    n = len(a)
    corr = float(np.mean(np.sum(a * s, axis=0) / n))
    return np.array(errors), corr


def check_action_state_sync(ds: Dataset, th: Thresholds) -> list[Finding]:
    """For position-controlled robots the commanded action should lead the measured state.

    An episode whose best lag is far from the dataset's typical lag has its
    action and observation streams shifted relative to each other.
    """
    a, s = ds.action_key, ds.state_key
    if not a or not s:
        return []
    eps = [
        ep
        for ep in ds.episodes.values()
        if a in ep.features
        and s in ep.features
        and ep.features[a].shape == ep.features[s].shape
        and ep.length > 2 * th.lag_max
        and np.all(np.isfinite(ep.features[a]))
        and np.all(np.isfinite(ep.features[s]))
    ]
    if len(eps) < th.min_episodes_for_outliers:
        return []
    curves = {ep.index: _lag_errors(ep.features[a], ep.features[s], th.lag_max) for ep in eps}
    curves = {i: c for i, c in curves.items() if len(c[0])}
    if len(curves) < th.min_episodes_for_outliers:
        return []
    corrs = np.array([c for _, c in curves.values()])
    if np.nanmedian(corrs) < 0.6:
        # Actions do not track state in this dataset (e.g. velocity control), so lag is meaningless.
        return []
    m = th.lag_max
    best = {i: int(np.argmin(err)) - m for i, (err, _) in curves.items()}  # lag in frames
    typical = int(np.median(list(best.values())))
    shifted = []
    for i, (err, _) in curves.items():
        k = best[i]
        if abs(k - typical) <= th.lag_deviation:
            continue
        # Only flag when the shifted alignment is clearly better, not a flat error curve.
        if err[typical + m] > 2.0 * err[k + m] + 1e-9:
            shifted.append(i)
    results = {i: (best[i], curves[i][1]) for i in curves}
    uncorrelated = [i for i, (k, r) in results.items() if r < 0.3]
    out = []
    if shifted:
        out.append(
            Finding(
                "sync.action_state_lag",
                WARNING,
                f"in {_eps(len(shifted))} the action stream is shifted against the state stream "
                f"(typical lag is {typical} frames); suggests a recording sync problem",
                episodes=shifted,
                details={"typical_lag": typical, "lags": {str(i): int(results[i][0]) for i in shifted[:100]}},
            )
        )
    if uncorrelated:
        out.append(
            Finding(
                "sync.action_state_uncorrelated",
                WARNING,
                f"in {_eps(len(uncorrelated))} the robot's state does not follow its actions at all",
                episodes=uncorrelated,
            )
        )
    return out


def check_trajectory_outliers(ds: Dataset, th: Thresholds) -> list[Finding]:
    """Episodes whose overall speed or range of motion is extreme for this dataset."""
    key = ds.action_key or ds.state_key
    if not key:
        return []
    eps = [
        ep
        for ep in ds.episodes.values()
        if key in ep.features and ep.length > 2 and np.all(np.isfinite(ep.features[key]))
    ]
    if len(eps) < th.min_episodes_for_outliers or len({ep.features[key].shape[1] for ep in eps}) != 1:
        return []
    speed = np.array([np.log1p(_motion(ep.features[key]) / ep.length) for ep in eps])
    extent = np.array([np.log1p(_extent(ep.features[key])) for ep in eps])
    worst = np.maximum(np.abs(_robust_z(speed)), np.abs(_robust_z(extent)))
    flagged = [ep.index for ep, w in zip(eps, worst) if w > th.robust_z * 1.5]
    if not flagged:
        return []
    return [
        Finding(
            "episodes.trajectory_outlier",
            INFO,
            f"{_eps(len(flagged))} move far faster, slower, wider or narrower than the rest; "
            "could be a different task, a different setup, or a failed attempt",
            episodes=flagged,
        )
    ]


DATASET_CHECKS = [
    check_info,
    check_feature_shapes,
    check_counts,
    check_files,
    check_tasks,
    check_values,
    check_stats,
    check_duplicates,
    check_video_alignment,
]
EPISODE_CHECKS = [
    check_timestamps,
    check_lengths,
    check_idle,
    check_spikes,
    check_action_state_sync,
    check_trajectory_outliers,
]


def run_checks(ds: Dataset, th: Thresholds | None = None) -> list[Finding]:
    th = th or Thresholds()
    findings: list[Finding] = []
    for check in DATASET_CHECKS + EPISODE_CHECKS:
        try:
            findings.extend(check(ds, th))
        except Exception as exc:  # one broken check should not hide the others
            findings.append(
                Finding(f"internal.{check.__name__}", INFO, f"check could not run: {type(exc).__name__}: {exc}")
            )
    return findings
