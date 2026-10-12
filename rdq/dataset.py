"""Load a local LeRobot dataset (format v3.0 or v2.x) into plain numpy arrays.

Only the tabular part is loaded (actions, states, timestamps, indices). Image
columns stored inside parquet are skipped and videos are never decoded, so a
dataset can be audited without a GPU and without downloading the videos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Columns LeRobot fills in automatically for every frame.
INDEX_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
NON_NUMERIC_DTYPES = {"image", "video", "string", "language"}
# Only these dtypes are loaded as numeric signals; anything else (depth, depth_video, pcd,
# audio, str, ...) is listed in Dataset.skipped_features with the reason.
NUMERIC_DTYPES = {
    "float16", "float32", "float64", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "bool",
}
# Wider arrays are images, depth maps, point clouds or tactile grids, not joint signals.
MAX_SIGNAL_WIDTH = 1024


def classify_feature(key: str, ft: dict | None) -> str | None:
    """None if the feature is a numeric signal rdq should load, else the reason it is skipped."""
    ft = ft or {}
    dtype = str(ft.get("dtype"))
    if dtype in NON_NUMERIC_DTYPES or dtype.endswith("video"):
        return f"{dtype} feature"
    if dtype not in NUMERIC_DTYPES:
        return f"dtype '{dtype}' is not a numeric signal"
    shape = ft.get("shape")
    if shape is None:
        return None
    if key == "action" or key.startswith("observation.state"):
        # The policy's own input and output are always loaded, whatever their declared shape:
        # the shape checks then report variable or unusual widths from the data itself.
        return None
    dims = []
    for d in shape if isinstance(shape, (list, tuple)) else [shape]:
        try:
            dims.append(int(d))
        except (TypeError, ValueError):
            return f"shape {list(shape)} has a variable dimension"
    if any(d < 0 for d in dims):
        return f"shape {list(shape)} has a variable dimension"
    if len(dims) >= 3 and sum(d > 1 for d in dims) >= 3:
        return f"shape {list(shape)} is an image-like array"
    width = int(np.prod(dims)) if dims else 1
    if width > MAX_SIGNAL_WIDTH:
        return f"shape {list(shape)} is too wide for a joint signal ({width} values per frame)"
    return None


class DatasetLoadError(Exception):
    """Raised when a folder cannot be read as a LeRobot dataset at all."""


@dataclass
class Episode:
    index: int
    length: int
    timestamps: np.ndarray | None
    frame_index: np.ndarray | None
    task_index: np.ndarray | None
    features: dict[str, np.ndarray]  # feature name -> (T, D) float array
    meta: dict = field(default_factory=dict)  # row from the episodes metadata, if any


@dataclass
class Dataset:
    root: Path
    info: dict
    version: str
    fps: float | None
    features: dict[str, dict]  # declared features from info.json
    numeric_keys: list[str]
    video_keys: list[str]
    tasks: dict[int, str]  # task_index -> task text
    episodes_meta: pd.DataFrame | None  # one row per episode, from meta/episodes*
    episodes: dict[int, Episode]
    stats: dict | None
    load_warnings: list[str] = field(default_factory=list)
    data_columns: list[str] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)  # data files no metadata points to
    has_episode_stats: bool = False  # per-episode stats (v2.1 episodes_stats.jsonl or v3 stats/ columns)
    skipped_features: dict[str, str] = field(default_factory=dict)  # feature -> why it was not loaded
    not_checked: list[str] = field(default_factory=list)  # checks that could not run, with the reason
    channel_index: dict[str, list[int]] = field(default_factory=dict)  # kept columns -> declared positions (after dropping dead channels)

    @property
    def action_key(self) -> str | None:
        return "action" if "action" in self.numeric_keys else None

    @property
    def state_key(self) -> str | None:
        if "observation.state" in self.numeric_keys:
            return "observation.state"
        for key in self.numeric_keys:
            if key.startswith("observation.state"):
                return key
        return None


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet without restoring pandas metadata (which can produce Arrow-backed
    list dtypes numpy cannot handle)."""
    return pq.read_table(path, columns=columns).to_pandas(ignore_metadata=True)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_version(raw: str | None) -> str:
    if not raw:
        return "unknown"
    return str(raw).lstrip("v")


def _load_tasks(root: Path, warnings: list[str]) -> dict[int, str]:
    tasks: dict[int, str] = {}
    v3_path = root / "meta" / "tasks.parquet"
    v2_path = root / "meta" / "tasks.jsonl"
    if v3_path.exists():
        df = pd.read_parquet(v3_path)  # task text lives in the pandas index, so keep metadata here
        if "task_index" in df.columns:
            # LeRobot stores the task text as the dataframe index.
            texts = df["task"] if "task" in df.columns else df.index.to_series()
            for text, idx in zip(texts, df["task_index"]):
                tasks[int(idx)] = "" if text is None else str(text)
        else:
            warnings.append("meta/tasks.parquet has no task_index column")
    elif v2_path.exists():
        for row in _read_jsonl(v2_path):
            if "task_index" in row:
                tasks[int(row["task_index"])] = str(row.get("task", "") or "")
    return tasks


def _load_episodes_meta(root: Path) -> pd.DataFrame | None:
    v3_dir = root / "meta" / "episodes"
    v2_path = root / "meta" / "episodes.jsonl"
    if v3_dir.exists():
        files = sorted(v3_dir.glob("**/*.parquet"))
        if files:
            frames = []
            for path in files:
                cols = [c for c in pq.read_schema(path).names if not c.startswith("stats/")]
                frames.append(_read_parquet(path, columns=cols))
            return pd.concat(frames, ignore_index=True)
    if v2_path.exists():
        return pd.DataFrame(_read_jsonl(v2_path))
    return None


def _to_matrix(values) -> np.ndarray:
    """Turn a parquet list column (or scalar column) into a (T, D) float array."""
    arr = np.asarray(values)
    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in values])
        except (ValueError, TypeError):
            # Ragged rows: pad with NaN so the shape problem shows up in the checks.
            width = max(len(np.asarray(v).reshape(-1)) for v in values)
            out = np.full((len(values), width), np.nan)
            for i, v in enumerate(values):
                flat = np.asarray(v, dtype=np.float64).reshape(-1)
                out[i, : len(flat)] = flat
            arr = out
    arr = arr.astype(np.float64, copy=False)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _referenced_data_files(root: Path, info: dict, episodes_meta: pd.DataFrame | None) -> list[Path] | None:
    """The data files LeRobot itself would read, or None if the metadata doesn't say."""
    template = info.get("data_path")
    if not template or episodes_meta is None or "episode_index" not in episodes_meta.columns:
        return None
    paths = set()
    try:
        if "data/chunk_index" in episodes_meta.columns and "data/file_index" in episodes_meta.columns:
            for c, f in zip(episodes_meta["data/chunk_index"], episodes_meta["data/file_index"]):
                paths.add(root / template.format(chunk_index=int(c), file_index=int(f)))
        elif "{episode_index" in template:
            chunk = int(info.get("chunks_size") or 1000)
            for ep in episodes_meta["episode_index"]:
                paths.add(root / template.format(episode_chunk=int(ep) // chunk, episode_index=int(ep)))
        else:
            return None
    except (KeyError, ValueError, IndexError):
        return None
    return sorted(paths)


def load_dataset(root: str | Path, max_frames: int | None = None) -> Dataset:
    root = Path(root)
    warnings: list[str] = []
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise DatasetLoadError(f"{root} has no meta/info.json, so it is not a LeRobot dataset")
    info = _read_json(info_path)
    features = info.get("features") or {}
    version = _parse_version(info.get("codebase_version"))
    fps = info.get("fps")

    numeric_keys = []
    skipped: dict[str, str] = {}
    for k, ft in features.items():
        if k in INDEX_COLUMNS:
            continue
        reason = classify_feature(k, ft if isinstance(ft, dict) else None)
        if reason is None:
            numeric_keys.append(k)
        elif str((ft or {}).get("dtype")) not in NON_NUMERIC_DTYPES or k == "action" or k.startswith("observation.state"):
            skipped[k] = reason
    video_keys = [k for k, ft in features.items() if (ft or {}).get("dtype") == "video"]

    tasks = _load_tasks(root, warnings)
    episodes_meta = _load_episodes_meta(root)

    stats = None
    if (root / "meta" / "stats.json").exists():
        try:
            stats = _read_json(root / "meta" / "stats.json")
        except json.JSONDecodeError:
            warnings.append("meta/stats.json is not valid JSON")

    all_files = sorted((root / "data").glob("**/*.parquet")) if (root / "data").exists() else []
    if not all_files:
        raise DatasetLoadError(f"{root} has no parquet files under data/")
    referenced = _referenced_data_files(root, info, episodes_meta)
    orphan_files: list[str] = []
    if referenced:
        existing = [p for p in referenced if p.exists()]
        missing_files = [p for p in referenced if not p.exists()]
        for p in missing_files[:20]:
            warnings.append(f"metadata points to {p.relative_to(root)}, which does not exist")
        ref_set = set(existing)
        orphan_files = [str(p.relative_to(root)) for p in all_files if p not in ref_set]
        data_files = existing or all_files
    else:
        data_files = all_files

    has_episode_stats = (root / "meta" / "episodes_stats.jsonl").exists()
    ep_dir = root / "meta" / "episodes"
    if ep_dir.exists():
        for path in ep_dir.glob("**/*.parquet"):
            if any(c.startswith("stats/") for c in pq.read_schema(path).names):
                has_episode_stats = True
                break

    frames = []
    data_columns: set[str] = set()
    total = 0
    for path in data_files:
        available = pq.read_schema(path).names
        data_columns.update(available)
        wanted = [c for c in (*INDEX_COLUMNS, *numeric_keys) if c in available]
        frames.append(_read_parquet(path, columns=wanted))
        total += len(frames[-1])
        if max_frames and total >= max_frames:
            warnings.append(f"stopped reading after {total} frames (max_frames={max_frames})")
            break
    data = pd.concat(frames, ignore_index=True)

    missing = [k for k in numeric_keys if k not in data.columns]
    for key in missing:
        warnings.append(f"declared feature '{key}' has no column in the data files")
    numeric_keys = [k for k in numeric_keys if k in data.columns]

    if "episode_index" not in data.columns:
        raise DatasetLoadError("data files have no episode_index column")

    meta_by_ep: dict[int, dict] = {}
    if episodes_meta is not None and "episode_index" in episodes_meta.columns:
        for row in episodes_meta.to_dict("records"):
            meta_by_ep[int(row["episode_index"])] = row

    # A column that will not convert to numbers is skipped with a warning, not fatal.
    bad: set[str] = set()

    def convert(group: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for key in numeric_keys:
            if key in bad:
                continue
            try:
                out[key] = _to_matrix(group[key].to_numpy())
            except (ValueError, TypeError) as exc:
                bad.add(key)
                skipped[key] = f"values could not be read as numbers ({type(exc).__name__})"
                warnings.append(f"feature '{key}' could not be read as numbers and was skipped")
        return out

    episodes: dict[int, Episode] = {}
    sort_cols = [c for c in ("episode_index", "frame_index") if c in data.columns]
    for ep_idx, group in data.sort_values(sort_cols, kind="stable").groupby("episode_index", sort=True):
        ep_idx = int(ep_idx)
        meta = meta_by_ep.get(ep_idx, {})
        start, end = meta.get("dataset_from_index"), meta.get("dataset_to_index")
        if start is not None and end is not None and "index" in group.columns:
            # Read exactly the rows LeRobot would use for this episode.
            sliced = group[(group["index"] >= int(start)) & (group["index"] < int(end))]
            if len(sliced):
                group = sliced
        episodes[ep_idx] = Episode(
            index=ep_idx,
            length=len(group),
            timestamps=group["timestamp"].to_numpy(np.float64) if "timestamp" in group else None,
            frame_index=group["frame_index"].to_numpy(np.int64) if "frame_index" in group else None,
            task_index=group["task_index"].to_numpy(np.int64) if "task_index" in group else None,
            features=convert(group),
            meta=meta,
        )

    if bad:
        numeric_keys = [k for k in numeric_keys if k not in bad]
        for ep in episodes.values():
            for key in bad:
                ep.features.pop(key, None)

    return Dataset(
        root=root,
        info=info,
        version=version,
        fps=float(fps) if fps else None,
        features=features,
        numeric_keys=numeric_keys,
        video_keys=video_keys,
        tasks=tasks,
        episodes_meta=episodes_meta,
        episodes=episodes,
        stats=stats,
        load_warnings=warnings,
        data_columns=sorted(data_columns),
        orphan_files=orphan_files,
        has_episode_stats=has_episode_stats,
        skipped_features=skipped,
    )
