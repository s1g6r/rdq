"""Quality checks for LeRobot datasets.

Every check returns a list of Finding objects. A finding is either about the
whole dataset (episodes == []) or about specific episodes. Thresholds live in
Thresholds so they can be tuned and reported alongside the results.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import re

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
    no_motion_ratio: float = 8.0  # a channel moves if its range is 8x its median frame-to-frame step (pure noise: 2-5x)
    spike_systematic_share: float = 0.5  # a channel hit in more episodes than this is reported once, not per episode
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

    def __post_init__(self):
        self.message = _singular(self.message)

    def to_dict(self) -> dict:
        return asdict(self)


_SINGULAR = {"are": "is", "have": "has", "were": "was", "move": "moves", "spend": "spends", "point": "points",
             "barely move": "barely moves"}


def _singular(msg: str) -> str:
    """Fix verb agreement after "1 episode" ("1 episode are" -> "1 episode is")."""
    def fix(m):
        return f"{m.group(1)}1 episode {_SINGULAR[m.group(2)]}"
    return re.sub(r"(^|\bin )1 episode (barely move|are|have|were|move|spend|point)\b", fix, msg)


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
    raw_fps = ds.info.get("fps")
    if isinstance(raw_fps, (int, float)) and raw_fps <= 0:
        out.append(Finding("info.bad_fps", CRITICAL, f"fps is {raw_fps}, which is impossible"))
    elif ds.fps is None and "fps" in ds.info:
        out.append(Finding("info.bad_fps", CRITICAL, "fps is empty, so timestamps cannot be checked or replayed"))
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
    if getattr(ds, "truncated", None):
        out.append(Finding("info.partial_read", INFO, ds.truncated + "; counts and stats were not compared with the metadata"))
    skipped = getattr(ds, "skipped_features", None) or {}
    if skipped:
        listed = "; ".join(f"{k} ({why})" for k, why in list(skipped.items())[:8])
        more = f" and {len(skipped) - 8} more" if len(skipped) > 8 else ""
        out.append(
            Finding(
                "schema.features_not_checked",
                INFO,
                f"{len(skipped)} feature(s) are not joint signals and were not checked: {listed}{more}",
                details={"features": skipped},
            )
        )
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
    skipped = getattr(ds, "skipped_features", None) or {}
    if not a and "action" in skipped:
        out.append(Finding("schema.action_not_numeric", CRITICAL,
                           f"'action' is declared but is not a numeric signal ({skipped['action']}), so a policy cannot learn from it"))
    elif not a:
        out.append(Finding("schema.no_action", CRITICAL, "no 'action' feature, so a policy cannot learn from it"))
    state_skipped = [k for k in skipped if k.startswith("observation.state")]
    if not s and state_skipped:
        out.append(Finding("schema.state_not_numeric", WARNING,
                           f"'{state_skipped[0]}' is declared but is not a numeric signal ({skipped[state_skipped[0]]})"))
    elif not s:
        out.append(Finding("schema.no_state", WARNING, "no observation.state feature"))
    return out


def check_counts(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    actual_eps = len(ds.episodes)
    actual_frames = sum(ep.length for ep in ds.episodes.values())
    declared_eps = ds.info.get("total_episodes")
    declared_frames = ds.info.get("total_frames")
    truncated = bool(getattr(ds, "truncated", None))
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
    if ds.episodes_meta is not None and "episode_index" in ds.episodes_meta.columns and ds.episodes:
        meta_ids = {int(x) for x in ds.episodes_meta["episode_index"]}
        data_ids = set(ds.episodes)
        not_in_meta = sorted(data_ids - meta_ids)
        not_in_data = sorted(meta_ids - data_ids)
        if not_in_meta:
            out.append(
                Finding(
                    "counts.episodes_not_in_metadata",
                    CRITICAL,
                    f"{_eps(len(not_in_meta))} in the data files are not listed in the episode metadata, "
                    "so LeRobot cannot load them",
                    episodes=not_in_meta,
                )
            )
        if not_in_data and not truncated:
            out.append(
                Finding(
                    "counts.metadata_episodes_missing",
                    CRITICAL,
                    f"the episode metadata lists {len(not_in_data)} episode(s) that are not in the data files",
                    details={"missing": not_in_data[:50]},
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


def dead_nan_channels(ds: Dataset) -> dict[str, list[int]]:
    """Channels that are NaN or infinite in every frame of every episode (unused sensors, padding)."""
    out: dict[str, list[int]] = {}
    for key in ds.numeric_keys:
        arrays = [ep.features[key] for ep in ds.episodes.values() if key in ep.features]
        if not arrays or len({a.shape[1] for a in arrays}) != 1:
            continue
        any_finite = np.zeros(arrays[0].shape[1], bool)
        for a in arrays:
            any_finite |= np.isfinite(a).any(0)
        dead = [int(d) for d in np.where(~any_finite)[0]]
        if dead:
            out[key] = dead
    return out


def _paired_dead(ds: Dataset, dead: dict[str, list[int]]) -> dict[str, list[int]]:
    """Action and state of the same width are compared channel by channel (sync, relative
    actions), so a channel dead in either is dropped from both to keep them aligned."""
    a, s = ds.action_key, ds.state_key
    if not a or not s or not (dead.get(a) or dead.get(s)):
        return dead
    widths = {(ep.features[a].shape[1], ep.features[s].shape[1]) for ep in ds.episodes.values()
              if a in ep.features and s in ep.features}
    if len(widths) != 1 or len({w for pair in widths for w in pair}) != 1:
        return dead
    both = sorted(set(dead.get(a, [])) | set(dead.get(s, [])))
    return {**dead, a: both, s: both}


def without_dead_channels(ds: Dataset, dead: dict[str, list[int]]) -> Dataset:
    """A view of the dataset with never-finite channels removed, for the signal checks."""
    if not dead:
        return ds
    from dataclasses import replace

    eps = {}
    for i, ep in ds.episodes.items():
        feats = dict(ep.features)
        for key, dims in dead.items():
            if key in feats:
                keep = np.setdiff1d(np.arange(feats[key].shape[1]), dims)
                if len(keep):
                    feats[key] = feats[key][:, keep]
                else:
                    feats.pop(key)
        eps[i] = replace(ep, features=feats)
    gone = {k for k, dims in dead.items() if all(k not in e.features for e in eps.values())}
    index = dict(ds.channel_index)
    for key, dims in dead.items():
        width = next((e.features[key].shape[1] for e in ds.episodes.values() if key in e.features), 0)
        index[key] = [int(d) for d in np.setdiff1d(np.arange(width), dims)]
    return replace(ds, episodes=eps, numeric_keys=[k for k in ds.numeric_keys if k not in gone], channel_index=index)


def action_state_pairs(ds: Dataset) -> tuple[list[int], list[int]] | None:
    """Columns of action and state that measure the same thing.

    Same width: every column, in order. Different widths (a gripper only in the action,
    velocities appended to the state, ...): the channels whose declared names match, if at
    least three do. Otherwise None, and channel-by-channel comparisons are skipped.
    """
    a, s = ds.action_key, ds.state_key
    if not a or not s:
        return None
    widths = {(ep.features[a].shape[1], ep.features[s].shape[1]) for ep in ds.episodes.values()
              if a in ep.features and s in ep.features}
    if len(widths) != 1:
        return None
    wa, ws = next(iter(widths))
    if wa == ws:
        return list(range(wa)), list(range(ws))
    na, ns = _names_list(ds.features.get(a, {})), _names_list(ds.features.get(s, {}))
    ia = ds.channel_index.get(a, list(range(wa)))
    is_ = ds.channel_index.get(s, list(range(ws)))
    if not na or not ns or len(ia) != wa or len(is_) != ws:
        return None
    pos_s = {}
    for col, d in enumerate(is_):
        if d < len(ns):
            pos_s.setdefault(str(ns[d]), col)
    pairs = [(col, pos_s[str(na[d])]) for col, d in enumerate(ia) if d < len(na) and str(na[d]) in pos_s]
    if len(pairs) < 3:
        return None
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _sync_episodes(ds: Dataset, th: Thresholds) -> tuple[list, tuple | None]:
    """(episode, action columns, state columns) for episodes long and finite enough to align."""
    a, s = ds.action_key, ds.state_key
    pairs = action_state_pairs(ds)
    if pairs is None:
        return [], None
    ia, is_ = pairs
    out = []
    for ep in ds.episodes.values():
        if a not in ep.features or s not in ep.features or ep.length <= th.lag_max + 10:
            continue
        A, S = ep.features[a][:, ia], ep.features[s][:, is_]
        if np.all(np.isfinite(A)) and np.all(np.isfinite(S)):
            out.append((ep, A, S))
    return out, pairs


def check_values(ds: Dataset, th: Thresholds) -> list[Finding]:
    out = []
    dead = dead_nan_channels(ds)
    for key, dims in dead.items():
        names = _names_list(ds.features.get(key, {}))
        labels = [names[d] if names and d < len(names) else f"dim {d}" for d in dims]
        width = next((ep.features[key].shape[1] for ep in ds.episodes.values() if key in ep.features), 0)
        if len(dims) == width and key in (ds.action_key, ds.state_key):
            out.append(
                Finding(
                    "values.empty_feature",
                    CRITICAL,
                    f"'{key}' has no finite value anywhere in the dataset, so a policy has nothing to learn from or condition on",
                    details={"feature": key},
                )
            )
            continue
        out.append(
            Finding(
                "values.empty_channel",
                WARNING,
                f"'{key}' has {len(dims)} channel(s) with no finite value anywhere in the dataset "
                f"({', '.join(map(str, labels[:6]))}): an unused sensor or padding; drop them before training",
                details={"feature": key, "dims": dims},
            )
        )
    clean = without_dead_channels(ds, dead)
    nan_eps: dict[str, list[int]] = {}
    for ep in clean.episodes.values():
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
        live = np.isfinite(stacked).any(0)  # never-finite channels are reported above
        with np.errstate(all="ignore"):
            spread = np.nanmax(np.where(np.isfinite(stacked), stacked, np.nan)[:, live], axis=0) - \
                np.nanmin(np.where(np.isfinite(stacked), stacked, np.nan)[:, live], axis=0)
        if len(stacked) < 2 or not live.any():
            continue
        constant = [int(d) for d, sp in zip(np.where(live)[0], spread) if sp < 1e-9]
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
    truncated = bool(getattr(ds, "truncated", None))
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
        data = np.where(np.isfinite(data), data, np.nan)
        live = ~np.all(np.isnan(data), axis=0)
        if not len(data) or not live.any():
            continue
        entry = ds.stats[key]
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            # Column-wise, ignoring non-finite values, as NaN-aware tools compute them.
            recomputed = {"min": np.nanmin(data, 0), "max": np.nanmax(data, 0), "mean": np.nanmean(data, 0),
                          "std": np.nanstd(data, 0)}
        span = recomputed["max"] - recomputed["min"]
        span = np.where(~np.isfinite(span) | (span < 1e-9), 1.0, span)
        problems = {}
        for stat, value in recomputed.items():
            if stat not in entry:
                continue
            stored = np.asarray(entry[stat], dtype=np.float64).reshape(-1)
            if stored.shape != value.shape:
                problems[stat] = f"stored {stored.shape[0]} values, data has {value.shape[0]}"
                continue
            rel = np.abs(stored - value) / span
            rel = np.where(live, rel, np.nan)  # never-finite channels are reported by check_values
            if np.isfinite(rel).any() and np.nanmax(rel) > th.stats_rel_tolerance:
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
        longer: dict[int, float] = {}
        shorter: dict[int, float] = {}
        for ep in ds.episodes.values():
            start, end = ep.meta.get(start_col), ep.meta.get(end_col)
            if start is None or end is None:
                continue
            delta = (float(end) - float(start)) - ep.length / ds.fps   # > 0: more video than data
            if abs(delta) > 1.5 * frame:
                (longer if delta > 0 else shorter)[ep.index] = round(abs(delta), 2)
        if longer or shorter:
            parts = []
            if shorter:
                parts.append(f"the video is shorter than the data in {_eps(len(shorter))} (by up to "
                             f"{max(shorter.values()):.2f}s: the last frames have no image, and loaders can fail there)")
            if longer:
                parts.append(f"the video is longer than the data in {_eps(len(longer))} (by up to "
                             f"{max(longer.values()):.2f}s: extra footage, often a reset left in, or data frames "
                             "dropped, so check that frames still line up with actions)")
            out.append(
                Finding(
                    "video.span_mismatch",
                    WARNING,
                    f"camera '{vkey}': " + "; ".join(parts),
                    episodes=sorted(set(longer) | set(shorter)),
                    details={"video_key": vkey, "worst_seconds": max([*longer.values(), *shorter.values()]),
                             "video_longer_by_s": {str(k): v for k, v in sorted(longer.items())},
                             "video_shorter_by_s": {str(k): v for k, v in sorted(shorter.items())}},
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


def _unwrap_angles(x: np.ndarray) -> np.ndarray:
    """Unwrap channels that look like angles wrapping at +-pi (values within +-pi, jumps above pi)."""
    if len(x) < 2:
        return x
    within = np.all(np.abs(x) <= np.pi + 1e-3, axis=0)
    jumps = np.any(np.abs(np.diff(x, axis=0)) > np.pi, axis=0)
    cols = np.where(within & jumps)[0]
    if not len(cols):
        return x
    x = x.copy()
    x[:, cols] = np.unwrap(x[:, cols], axis=0)
    return x


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
        ds.not_checked.append("idle episodes: needs at least 3 episodes to compare")
        return []
    # Spatial extent, not path length: sensor jitter adds path length without moving anywhere.
    motions = np.array([_extent(ep.features[key]) for ep in eps])
    median = float(np.median(motions))
    out = []
    if median <= 1e-9:
        return out  # nothing moves at all: check_no_motion reports it
    # Compare each episode with others of the same task: some tasks legitimately move less.
    def task_of(ep):
        if ep.task_index is None or not len(ep.task_index):
            return None
        vals, counts = np.unique(ep.task_index, return_counts=True)
        return int(vals[counts.argmax()])

    tasks = [task_of(ep) for ep in eps]
    by_task: dict = {}
    for t, m in zip(tasks, motions):
        by_task.setdefault(t, []).append(m)
    task_med = {t: float(np.median(v)) for t, v in by_task.items() if len(v) >= 3}
    ref_of = {}
    for t, med in task_med.items():
        others = [m for u, m in task_med.items() if u != t]
        other_med = float(np.median(others)) if others else median
        # A whole task that barely moves compared with the other tasks is suspicious too
        # (for example a session recorded with the follower arm unplugged).
        ref_of[t] = other_med if med < th.idle_motion_ratio * other_med else med
    ref = [ref_of.get(t, median) for t in tasks]
    idle = [ep.index for ep, m, r in zip(eps, motions, ref) if r > 0 and m < th.idle_motion_ratio * r]
    if idle:
        out.append(
            Finding(
                "episodes.idle",
                CRITICAL,
                f"{_eps(len(idle))} barely move (less than {th.idle_motion_ratio:.0%} of a "
                "typical episode of the same task); the robot was probably not doing the task",
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


def check_no_motion(ds: Dataset, th: Thresholds) -> list[Finding]:
    """Streams that never move: the range of motion is no larger than frame-to-frame jitter.

    Scale-free, so sensor noise on a frozen stream does not count as motion. Checked for the
    action and the state separately: a frozen state with moving actions means the follower
    arm was not recorded (or not connected).
    """
    out = []
    moving = {}
    # Delta or velocity actions look like noise by design, so actions are only judged when
    # they are absolute positions (or when there is no state to judge instead).
    keys = [ds.state_key]
    if ds.action_key and (not ds.state_key or _action_tracks_state(ds, th)):
        keys.append(ds.action_key)
    for key in keys:
        if not key:
            continue
        eps = [ep for ep in ds.episodes.values() if key in ep.features and ep.length > 5]
        if not eps:
            continue
        ratios = []
        for ep in eps:
            x = ep.features[key]
            x = x[np.all(np.isfinite(x), axis=1)]
            if len(x) < 6:
                ratios.append(np.nan)
                continue
            # Per channel: a stream moves if enough of its channels range further than their own
            # frame-to-frame noise. (One norm over all channels lets noisy velocity or effort
            # channels in a wide state vector drown out the joints that do move.)
            x = _unwrap_angles(x)
            filt = _medfilt(x, 5) if len(x) >= 5 else x
            ext = np.ptp(filt, axis=0)
            jit = np.median(np.abs(np.diff(x, axis=0)), axis=0)
            live = ext > 1e-9
            if not live.any():
                ratios.append(0.0)
                continue
            per = ext[live] / np.maximum(jit[live], 1e-12)
            share = float(np.mean(per >= th.no_motion_ratio))
            ratios.append(1.0 if share >= 0.1 else 0.0)
        r = np.array(ratios, float)
        if not np.isfinite(r).any():
            continue
        frozen = [ep.index for ep, v in zip(eps, r) if np.isfinite(v) and v < 0.5]
        moving[key] = len(frozen) < 0.5 * np.isfinite(r).sum()
        if not moving[key]:
            out.append(
                Finding(
                    "episodes.no_motion",
                    CRITICAL,
                    f"'{key}' does not move in {_eps(len(frozen))}: its range of motion is no larger than its "
                    "frame-to-frame noise (a frozen stream or a disconnected arm)",
                    episodes=frozen,
                    details={"feature": key},
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
    eps, _ = _sync_episodes(ds, th)
    eps = eps[:sample]
    if not eps:
        return False
    corrs = []
    for ep, A, S in eps:
        errors, _ = _lag_errors(A, S, th.lag_max)
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
        # Angles that wrap at +-pi (Euler angles, yaw) jump by 2*pi and back; unwrap them first.
        unwrapped = {ep.index: _unwrap_angles(ep.features[key]) for ep in eps}
        residuals = {i: np.abs(x - _medfilt(x, th.spike_window)) for i, x in unwrapped.items()}
        # Normal jitter: the median over episodes of each episode's 99th-percentile residual.
        # Pooling all frames would let the glitches themselves set the bar once they make up
        # more than 1% of frames (a few bad episodes with 3-frame dropouts is enough).
        # The 90th percentile (scaled to where the 99th would be for Gaussian jitter, x1.6) stays
        # clear of glitch frames even when most episodes contain a few of them.
        typical = 1.6 * np.median(np.stack([np.percentile(r, 90, axis=0) for r in residuals.values()]), axis=0)
        limit = np.maximum(th.spike_fraction * span, 6.0 * typical)
        hits = {idx: np.any(resid > limit, axis=0) for idx, resid in residuals.items()}  # channels hit per episode
        share = np.mean(np.stack(list(hits.values())), axis=0)
        # A channel that "glitches" in most episodes is behaving as designed (fast gripper pulses,
        # quantised encoders, angle wrap) or is broken everywhere; either way it is a property of
        # the dataset, not of individual episodes, so report it once and keep it out of the filter.
        systematic = share > th.spike_systematic_share if len(eps) >= 8 else np.zeros_like(share, bool)
        names = _names_list(ds.features.get(key, {}))

        def label(d):
            return str(names[d]) if names and d < len(names) else f"dim {d}"

        if systematic.any():
            dims = [int(d) for d in np.where(systematic)[0]]
            out.append(
                Finding(
                    "episodes.spikes_systematic",
                    WARNING,
                    f"'{key}' channel(s) {', '.join(label(d) for d in dims[:6])} jump out and back within a few "
                    f"frames in most episodes ({share[dims].max():.0%}). That is either how this signal behaves "
                    "(fast gripper pulses, quantised encoders) or a sensor fault in every session; check which before "
                    "training. Episodes are not excluded for it",
                    details={"feature": key, "dims": dims, "episode_share": [round(float(share[d]), 3) for d in dims]},
                )
            )
        flagged, channels = {}, {}
        for idx, resid in residuals.items():
            over = (resid > limit) & ~systematic
            n = int(np.sum(np.any(over, axis=1)))
            if n:
                flagged[idx] = n
                channels[idx] = [label(int(d)) for d in np.where(over.any(0))[0]][:4]
        if flagged:
            out.append(
                Finding(
                    "episodes.spikes",
                    WARNING,
                    f"'{key}' has short glitches in {_eps(len(flagged))}: a channel jumps far outside its "
                    "normal motion for a few frames and comes back; usually a sensor glitch or a teleop dropout",
                    episodes=sorted(flagged),
                    details={
                        "feature": key,
                        "glitch_frames_per_episode": {str(k): v for k, v in list(flagged.items())[:100]},
                        "channels_per_episode": {str(k): v for k, v in list(channels.items())[:100]},
                    },
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
        ds.not_checked.append("action/state sync: needs both an action and an observation.state")
        return []
    eps, pairs = _sync_episodes(ds, th)
    if len(eps) < th.min_episodes_for_outliers:
        why = ("action and state have different widths and fewer than three channel names in common"
               if pairs is None else f"fewer than {th.min_episodes_for_outliers} long, fully finite episodes")
        ds.not_checked.append(f"action/state sync: {why}")
        return []
    curves = {ep.index: _lag_errors(A, S, th.lag_max) for ep, A, S in eps}
    curves = {i: c for i, c in curves.items() if len(c[0])}
    if len(curves) < th.min_episodes_for_outliers:
        return []
    corrs = np.array([c for _, c in curves.values()], float)
    if not np.isfinite(corrs).any():
        ds.not_checked.append("action/state sync: the state does not vary, so lag cannot be measured")
        return []
    if np.nanmedian(corrs) < 0.6:
        # Actions do not track state in this dataset (e.g. velocity control), so lag is meaningless.
        ds.not_checked.append("action/state sync: actions do not track the state (delta or velocity control?)")
        return []
    m = th.lag_max
    best = {i: int(np.argmin(err)) - m for i, (err, _) in curves.items()}  # lag in frames
    typical = int(round(float(np.median(list(best.values())))))
    shifted = []
    for i, (err, _) in curves.items():
        k = best[i]
        if abs(k - typical) <= th.lag_deviation:
            continue
        # Only flag when the shifted alignment is clearly better than the typical one, not a
        # flat error curve. Velocity errors have a noise floor, so 30% better is already clear.
        if err[typical + m] > 1.3 * err[k + m] + 1e-9:
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
        ds.not_checked.append(f"trajectory outliers: needs at least {th.min_episodes_for_outliers} fully finite episodes of one width")
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
    check_no_motion,
    check_idle,
    check_spikes,
    check_action_state_sync,
    check_trajectory_outliers,
]


# Checks that only make sense for LeRobot's on-disk format (metadata files, stats, video index).
FORMAT_CHECKS = {check_info, check_counts, check_files, check_tasks, check_stats, check_video_alignment}


def run_checks(ds: Dataset, th: Thresholds | None = None, signal_only: bool = False) -> list[Finding]:
    """Run every check. signal_only=True skips the LeRobot-format checks (for MCAP and other sources)."""
    th = th or Thresholds()
    findings: list[Finding] = []
    ds.not_checked.clear()
    # Signal checks run on a view without never-finite channels, so one unused sensor does not
    # exclude every episode from them (check_values reports those channels once).
    clean = without_dead_channels(ds, _paired_dead(ds, dead_nan_channels(ds)))
    raw_checks = {check_info, check_feature_shapes, check_counts, check_files, check_tasks, check_values, check_stats,
                  check_video_alignment}
    for check in DATASET_CHECKS + EPISODE_CHECKS:
        if signal_only and check in FORMAT_CHECKS:
            continue
        try:
            findings.extend(check(ds if check in raw_checks else clean, th))
        except Exception as exc:  # one broken check should not hide the others
            findings.append(
                Finding(f"internal.{check.__name__}", INFO, f"check could not run: {type(exc).__name__}: {exc}")
            )
    if len(ds.episodes) == 1:
        ds.not_checked.append("cross-episode checks (idle, duplicates, outliers, sync): only one episode")
    return findings
