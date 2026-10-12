"""Run every cell of the Kaggle notebook locally on CPU with tiny settings.

Uses a synthetic dataset shaped like lerobot/pusht_keypoints (state 2, environment
state 16, action 2) in place of the download, and the system's LeRobot install in
place of the Python 3.12 venv. Training is a handful of steps and evaluation a few
episodes in the real Push-T simulator, so this checks the plumbing, not the numbers.

    python experiments/kaggle/smoke_test.py experiments/kaggle/rdq_pusht_experiment.ipynb
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rdq import synthetic as syn  # noqa: E402


def pusht_like(root: Path, n_episodes: int = 24) -> None:
    rng = np.random.default_rng(0)
    eps = []
    for _ in range(n_episodes):
        n = int(rng.integers(80, 140))
        t = np.linspace(0, 1, n)[:, None]
        start, goal = rng.uniform(50, 450, 2), rng.uniform(50, 450, 2)
        action = start + (goal - start) * (3 * t**2 - 2 * t**3)
        state = np.vstack([action[:1], action[:-1]]) + rng.normal(0, 0.5, action.shape)
        eps.append({"action": action, "state": state, "timestamp": np.arange(n) / 10.0, "task_index": 0})
    syn.write_dataset(root, eps, fps=10, names=["x", "y"], with_video=False, robot_type=None,
                      tasks=["Push the T-shaped block onto the target."])
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(data_path)
    env = rng.uniform(0, 512, size=(len(df), 16)).astype(np.float32)
    df["observation.environment_state"] = [row.tolist() for row in env]
    df.to_parquet(data_path)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.environment_state"] = {"dtype": "float32", "shape": [16], "names": None}
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    stats = json.loads((root / "meta" / "stats.json").read_text())
    stats["observation.environment_state"] = {
        "min": env.min(0).tolist(), "max": env.max(0).tolist(), "mean": env.mean(0).tolist(),
        "std": env.std(0).tolist(), "count": [len(env)],
    }
    (root / "meta" / "stats.json").write_text(json.dumps(stats))


def main(nb_path: str, no_train: bool = False) -> None:
    nb = json.loads(Path(nb_path).read_text())
    cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    config, unpack, _env, data, corrupt, robomimic, train, evaluate, results = cells

    tmp = Path(tempfile.mkdtemp(prefix="rdq_smoke_"))
    os.environ["RDQ_SCRATCH"] = str(tmp / "scratch")
    os.chdir(tmp)
    g: dict = {}
    exec(config, g)
    g.update(STEPS=20, BATCH=8, EVAL_EPISODES=4, DEVICE="cpu")
    exec(unpack, g)

    # Stand-in for the Python 3.12 venv: point at the system LeRobot install.
    venv = g["SCRATCH"] / "venv312" / "bin"
    venv.mkdir(parents=True, exist_ok=True)
    for name, target in (("python", sys.executable), ("lerobot-train", shutil.which("lerobot-train")),
                         ("lerobot-eval", shutil.which("lerobot-eval"))):
        assert target, f"{name} not found"
        (venv / name).symlink_to(target)
    g["VENV"], g["PY"] = venv.parent, venv / "python"

    pusht_like(g["SCRATCH"] / "pusht_clean")
    steps = [("data", data), ("corrupt", corrupt), ("robomimic", robomimic), ("train", train),
             ("evaluate", evaluate), ("results", results)]
    if no_train:  # small machines: check everything up to training
        steps = steps[:3]
    for name, src in steps:
        print(f"\n===== cell: {name} =====", flush=True)
        exec(src, g)
    if no_train:
        print("\nSMOKE TEST PASSED (up to training)")
        return
    print("\nSMOKE TEST PASSED:", json.load(open(g["WORK"] / "results.json"))["policies"])


if __name__ == "__main__":
    main(sys.argv[1], no_train="--no-train" in sys.argv)
