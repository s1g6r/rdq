"""Read robot recordings in MCAP (ROS 2 or JSON-encoded) so they can be audited before conversion.

Data vendors often deliver MCAP rather than LeRobot. This reader maps one or two topics
onto rdq's in-memory dataset so the signal-level checks (timing, idle, glitches,
action/state lag, duplicates, NaNs, dead channels, outliers) and the episode scores
run unchanged:

  state topic   sensor_msgs/msg/JointState (``position``), std_msgs Float*MultiArray
                (``data``), or a JSON message with a numeric array field
  action topic  optional, same message kinds; each action is held until the next one
                and sampled at the state timestamps

One MCAP file is one episode. Needs ``pip install mcap mcap-ros2-support``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import Dataset, DatasetLoadError, Episode

ARRAY_FIELDS = ("position", "data", "values", "q", "joints")


def _vector(msg) -> np.ndarray | None:
    """Pull the numeric vector out of a decoded message (ROS 2 object or JSON dict)."""
    for field in ARRAY_FIELDS:
        v = msg.get(field) if isinstance(msg, dict) else getattr(msg, field, None)
        if v is not None:
            try:
                arr = np.asarray(list(v), dtype=float)
            except (TypeError, ValueError):
                continue
            if arr.ndim == 1 and arr.size:
                return arr
    return None


def _read_topics(path: Path, topics: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:  # pragma: no cover
        raise DatasetLoadError("reading MCAP needs: pip install mcap mcap-ros2-support") from exc
    decoders = []
    try:
        from mcap_ros2.decoder import DecoderFactory as Ros2Factory

        decoders.append(Ros2Factory())
    except ImportError:
        pass
    out: dict[str, list] = {t: [] for t in topics}
    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=decoders)
        for schema, channel, message in reader.iter_messages(topics=topics):
            if channel.message_encoding == "json":
                decoded = json.loads(message.data)
            else:
                decoded = None
                for factory in decoders:
                    dec = factory.decoder_for(channel.message_encoding, schema)
                    if dec is not None:
                        decoded = dec(message.data)
                        break
                if decoded is None:
                    raise DatasetLoadError(f"{path.name}: no decoder for encoding '{channel.message_encoding}' on {channel.topic}")
            vec = _vector(decoded)
            if vec is not None:
                out[channel.topic].append((message.log_time, vec))
    result = {}
    for t, rows in out.items():
        if rows:
            dims = {len(v) for _, v in rows}
            if len(dims) != 1:
                raise DatasetLoadError(f"{path.name}: topic {t} changes width ({sorted(dims)})")
            result[t] = (np.array([r[0] for r in rows], dtype=np.int64), np.stack([r[1] for r in rows]))
    return result


def load_mcap(paths: list[str | Path], state_topic: str, action_topic: str | None = None) -> Dataset:
    import re as _re

    def natural(p: Path):
        return [int(t) if t.isdigit() else t for t in _re.split(r"(\d+)", p.name)]

    paths = sorted((Path(p) for p in paths), key=natural)
    if not paths:
        raise DatasetLoadError("no MCAP files given")
    topics = [state_topic] + ([action_topic] if action_topic else [])
    episodes: dict[int, Episode] = {}
    warnings: list[str] = []
    dts = []
    for i, path in enumerate(paths):
        data = _read_topics(path, topics)
        if state_topic not in data:
            warnings.append(f"{path.name}: no messages on {state_topic}")
            continue
        t_ns, state = data[state_topic]
        order = np.argsort(t_ns, kind="stable")
        t_ns, state = t_ns[order], state[order]
        features = {"observation.state": state}
        if action_topic:
            if action_topic in data:
                at, act = data[action_topic]
                o = np.argsort(at, kind="stable")
                at, act = at[o], act[o]
                idx = np.searchsorted(at, t_ns, side="right") - 1
                held = act[np.clip(idx, 0, len(act) - 1)].copy()
                held[idx < 0] = act[0]
                features["action"] = held
            else:
                warnings.append(f"{path.name}: no messages on {action_topic}")
        ts = (t_ns - t_ns[0]) / 1e9
        if len(ts) > 1:
            dts.append(np.median(np.diff(ts)))
        episodes[i] = Episode(index=i, length=len(ts), timestamps=ts, frame_index=np.arange(len(ts)),
                              task_index=np.zeros(len(ts), dtype=int), features=features, meta={"file": path.name})
    if not episodes:
        raise DatasetLoadError("no episode had messages on the state topic: " + "; ".join(warnings))
    fps = float(round(1.0 / np.median(dts), 3)) if dts else None
    numeric = ["observation.state"] + (["action"] if any("action" in e.features for e in episodes.values()) else [])
    dims = {k: next(e.features[k].shape[1] for e in episodes.values() if k in e.features) for k in numeric}
    info = {
        "codebase_version": "mcap",
        "fps": fps,
        "robot_type": None,
        "total_episodes": len(episodes),
        "total_frames": int(sum(e.length for e in episodes.values())),
        "features": {k: {"dtype": "float64", "shape": [dims[k]], "names": None} for k in numeric},
        "source": {"format": "mcap", "state_topic": state_topic, "action_topic": action_topic},
    }
    return Dataset(
        root=paths[0].parent,
        info=info,
        version="mcap",
        fps=fps,
        features=info["features"],
        numeric_keys=numeric,
        video_keys=[],
        tasks={0: "unknown"},
        episodes_meta=None,
        episodes=episodes,
        stats=None,
        load_warnings=warnings,
        data_columns=numeric,
        has_episode_stats=True,
    )
