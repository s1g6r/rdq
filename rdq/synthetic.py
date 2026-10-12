"""Write synthetic LeRobot v3.0 datasets with known, injected faults.

Used by the tests (every check must catch the fault it is designed for and
stay quiet on clean data) and by the experiment (corrupt a clean dataset in
controlled ways and see whether the score predicts the damage).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _reach_trajectory(rng: np.random.Generator, length: int, dims: int) -> np.ndarray:
    """Smooth motion through a few random waypoints, like a pick-and-place."""
    n_way = 4
    way = rng.uniform(-60, 60, size=(n_way, dims))
    way[0] = rng.normal(0, 3, size=dims)  # start near home
    t = np.linspace(0, n_way - 1, length)
    out = np.empty((length, dims))
    for d in range(dims):
        out[:, d] = np.interp(t, np.arange(n_way), way[:, d])
    # Smooth the corners.
    kernel = np.ones(9) / 9
    for d in range(dims):
        padded = np.pad(out[:, d], 4, mode="edge")
        out[:, d] = np.convolve(padded, kernel, mode="valid")
    return out


def make_episodes(
    n_episodes: int = 30,
    length: int = 200,
    fps: int = 30,
    seed: int = 0,
    lag: int = 2,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_episodes):
        ep_len = int(length * rng.uniform(0.8, 1.2))
        action = _reach_trajectory(rng, ep_len, len(MOTORS))
        state = np.vstack([np.repeat(action[:1], lag, axis=0), action[:-lag]])
        state = state + rng.normal(0, 0.3, size=state.shape)
        episodes.append(
            {
                "action": action,
                "state": state,
                "timestamp": np.arange(ep_len) / fps,
                "task_index": 0,
            }
        )
    return episodes


def write_dataset(
    root: str | Path,
    episodes: list[dict],
    fps: int = 30,
    tasks: list[str] | None = None,
    names: list[str] | None = None,
    stats_override: dict | None = None,
    with_video: bool = True,
    video_span_error: dict[int, float] | None = None,
    meta_length_error: dict[int, int] | None = None,
    robot_type: str | None = "so101_follower",
) -> Path:
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    tasks = tasks or ["Pick up the red cube and place it in the bowl"]
    names = names or MOTORS
    dims = episodes[0]["action"].shape[1]

    rows, ep_rows = [], []
    global_idx = 0
    video_t = 0.0
    for i, ep in enumerate(episodes):
        n = len(ep["action"])
        task_idx = ep.get("task_index", 0)
        for t in range(n):
            rows.append(
                {
                    "action": ep["action"][t].astype(np.float32).tolist(),
                    "observation.state": ep["state"][t].astype(np.float32).tolist(),
                    "timestamp": float(ep["timestamp"][t]),
                    "frame_index": int(ep.get("frame_index", np.arange(n))[t]),
                    "episode_index": i,
                    "index": global_idx,
                    "task_index": int(task_idx),
                }
            )
            global_idx += 1
        span = n / fps + (video_span_error or {}).get(i, 0.0)
        ep_row = {
            "episode_index": i,
            "tasks": [tasks[task_idx] if task_idx < len(tasks) else "missing"],
            "length": n + (meta_length_error or {}).get(i, 0),
            "dataset_from_index": global_idx - n,
            "dataset_to_index": global_idx,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        if with_video:
            ep_row["videos/observation.images.front/chunk_index"] = 0
            ep_row["videos/observation.images.front/file_index"] = 0
            ep_row["videos/observation.images.front/from_timestamp"] = video_t
            ep_row["videos/observation.images.front/to_timestamp"] = video_t + span
            video_t += span
        ep_rows.append(ep_row)

    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(ep_rows).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")).to_parquet(
        root / "meta" / "tasks.parquet"
    )

    features = {
        "action": {"dtype": "float32", "shape": [dims], "names": names},
        "observation.state": {"dtype": "float32", "shape": [dims], "names": names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if with_video:
        features["observation.images.front"] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        }
    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": global_idx,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))

    all_action = np.concatenate([e["action"] for e in episodes])
    all_state = np.concatenate([e["state"] for e in episodes])
    stats = {}
    for key, arr in (("action", all_action), ("observation.state", all_state)):
        finite = arr[np.all(np.isfinite(arr), axis=1)]
        stats[key] = {
            "min": finite.min(0).tolist(),
            "max": finite.max(0).tolist(),
            "mean": finite.mean(0).tolist(),
            "std": finite.std(0).tolist(),
            "count": [len(finite)],
        }
    if stats_override:
        for key, entry in stats_override.items():
            stats.setdefault(key, {}).update(entry)
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    return root


# --------------------------------------------------------------------------- #
# Fault injection. Each function edits an episode dict in place.
# --------------------------------------------------------------------------- #


def make_idle(ep: dict) -> None:
    ep["action"] = np.repeat(ep["action"][:1], len(ep["action"]), axis=0)
    ep["state"] = ep["action"] + np.random.default_rng(1).normal(0, 0.3, size=ep["action"].shape)


def drop_frames(ep: dict, n: int = 15, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(np.arange(1, len(ep["action"]) - 1), size=len(ep["action"]) - 2 - n, replace=False))
    keep = np.concatenate([[0], keep, [len(ep["action"]) - 1]])
    for key in ("action", "state", "timestamp"):
        ep[key] = ep[key][keep]


def reverse_time(ep: dict) -> None:
    ts = ep["timestamp"].copy()
    mid = len(ts) // 2
    ts[mid], ts[mid + 1] = ts[mid + 1], ts[mid]
    ep["timestamp"] = ts


def add_spike(ep: dict, size: float = 400.0) -> None:
    t = len(ep["state"]) // 3
    ep["state"] = ep["state"].copy()
    ep["state"][t, 1] += size


def shift_action(ep: dict, frames: int = 12) -> None:
    a = ep["action"]
    ep["action"] = np.vstack([a[frames:], np.repeat(a[-1:], frames, axis=0)])


def add_nan(ep: dict) -> None:
    ep["action"] = ep["action"].copy()
    ep["action"][5, 2] = np.nan


def truncate(ep: dict, keep: int = 6) -> None:
    for key in ("action", "state", "timestamp"):
        ep[key] = ep[key][:keep]


def pad_idle(ep: dict, ratio: float = 1.0) -> None:
    """Append a long still segment, as if the operator forgot to stop recording."""
    n = int(len(ep["action"]) * ratio)
    fps_dt = ep["timestamp"][1] - ep["timestamp"][0]
    ep["action"] = np.vstack([ep["action"], np.repeat(ep["action"][-1:], n, axis=0)])
    ep["state"] = np.vstack([ep["state"], np.repeat(ep["state"][-1:], n, axis=0)])
    ep["timestamp"] = np.arange(len(ep["action"])) * fps_dt


def kill_dimension(episodes: list[dict], dim: int = 4) -> None:
    for ep in episodes:
        ep["action"] = ep["action"].copy()
        ep["state"] = ep["state"].copy()
        ep["action"][:, dim] = 0.0
        ep["state"][:, dim] = 0.0


def delay_action(ep: dict, frames: int = 10) -> None:
    """Action stream written late: each recorded action belongs to `frames` steps earlier."""
    a = ep["action"]
    ep["action"] = np.vstack([np.repeat(a[:1], frames, axis=0), a[:-frames]])


def add_gripper_steps(episodes: list[dict], dim: int = 5) -> None:
    """Make one channel a binary gripper that snaps between open and closed (a legitimate step)."""
    for ep in episodes:
        n = len(ep["action"])
        g = np.zeros(n)
        g[n // 3 : 2 * n // 3] = 100.0
        ep["action"] = ep["action"].copy()
        ep["state"] = ep["state"].copy()
        ep["action"][:, dim] = g
        ep["state"][:, dim] = np.concatenate([[0, 0], g[:-2]])


def add_glitch(ep: dict, frames: int = 3) -> None:
    """A short dropout: state jumps far away for a few frames and returns."""
    t = len(ep["state"]) // 2
    ep["state"] = ep["state"].copy()
    ep["state"][t : t + frames, 0] += 300.0


def write_dataset_v21(root, episodes: list[dict], fps: int = 30, tasks: list[str] | None = None):
    """LeRobot v2.1 layout: one parquet per episode, jsonl metadata, per-episode stats, no stats.json."""
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    tasks = tasks or ["Pick up the red cube and place it in the bowl"]
    dims = episodes[0]["action"].shape[1]
    idx = 0
    ep_lines, stat_lines = [], []
    for i, ep in enumerate(episodes):
        n = len(ep["action"])
        rows = {
            "action": [r.astype(np.float32).tolist() for r in ep["action"]],
            "observation.state": [r.astype(np.float32).tolist() for r in ep["state"]],
            "timestamp": ep["timestamp"].astype(np.float32),
            "frame_index": np.arange(n),
            "episode_index": np.full(n, i),
            "index": np.arange(idx, idx + n),
            "task_index": np.zeros(n, dtype=int),
        }
        idx += n
        pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
        ep_lines.append(json.dumps({"episode_index": i, "tasks": [tasks[0]], "length": n}))
        stat_lines.append(json.dumps({"episode_index": i, "stats": {"action": {"mean": ep["action"].mean(0).tolist()}}}))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(ep_lines) + "\n")
    (root / "meta" / "episodes_stats.jsonl").write_text("\n".join(stat_lines) + "\n")
    (root / "meta" / "tasks.jsonl").write_text("\n".join(json.dumps({"task_index": j, "task": t}) for j, t in enumerate(tasks)) + "\n")
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so100",
        "total_episodes": len(episodes),
        "total_frames": idx,
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {"dtype": "float32", "shape": [dims], "names": [MOTORS[:dims]]},
            "observation.state": {"dtype": "float32", "shape": [dims], "names": [MOTORS[:dims]]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return root
