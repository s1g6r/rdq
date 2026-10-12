"""Build a self-contained Kaggle/Colab notebook for the Push-T filtering experiment.

    python experiments/kaggle/build_notebook.py OUT.ipynb

The rdq source and the corruption script are embedded in the notebook, so the
user only uploads one file and clicks Run.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted((ROOT / "rdq").glob("*.py")):
            tar.add(path, arcname=f"rdq/{path.name}")
        for path in sorted((ROOT / "rdq" / "data").glob("*.json")):  # licence knowledge base, needed at import
            tar.add(path, arcname=f"rdq/data/{path.name}")
        for name in ("corrupt.py", "robomimic_mh.py"):
            tar.add(ROOT / "experiments" / name, arcname=f"experiments/{name}")
    return base64.b64encode(buf.getvalue()).decode()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")}


INTRO = """
# rdq · Push-T filtering experiment

**Question:** if some robot demonstrations are broken, does training only on the episodes rdq approves give a better robot policy than training on everything, or on a random subset of the same size?

**Setup:**
- Data: LeRobot's Push-T demonstrations, the keypoint version with no camera video. That's 206 human demos of pushing a T-shaped block into a target.
- Corruption: 30% of the episodes get one of three realistic recording faults: a frozen log, actions written 8 frames late, or a short teleop glitch.
- Training: three diffusion policies, identical except for which episodes they train on.
  - **raw**: every episode
  - **filtered**: only the episodes rdq marks usable
  - **random**: a random subset the same size as filtered
- Evaluation: each policy runs in the Push-T simulator, and the notebook reports the success rate with 95% confidence intervals.

**How to run on Kaggle (free):**
1. Create a new notebook, then choose **File → Import notebook** and upload this file.
2. In the right panel under **Settings**:
   - Set **Accelerator** to **GPU T4** (or T4 x2).
   - Turn **Internet** on. This needs a phone-verified account.
3. Click **Save Version → Save & Run All (Commit)**. It runs in the background, so you can close the tab. It takes about 4–6 hours with the default settings.
4. When it's done, open the version's **Output** tab and download `results.json` and `rdq_report.html`.

Every step skips work that's already done, so if a run is cut short, start it again.
"""

CONFIG = """
import os, sys, json, subprocess, pathlib, shutil, random, base64, io, tarfile, math, time

STEPS = 30_000          # training steps per policy (lower to 5_000 for a quick, rough run)
BATCH = 64
CORRUPT_RATE = 0.30
SEED = 0
EVAL_EPISODES = 200     # simulator episodes per policy

WORK = pathlib.Path("/kaggle/working") if pathlib.Path("/kaggle/working").exists() else pathlib.Path.cwd() / "rdq_experiment"
SCRATCH = pathlib.Path(os.environ.get("RDQ_SCRATCH", "/tmp/rdq"))   # big files (venv, data, checkpoints) stay out of saved outputs
WORK.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)

def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    result = subprocess.run([str(c) for c in cmd], **kw)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}")
    return result

try:
    run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    DEVICE = "cuda"
except Exception:
    DEVICE = "cpu"
    print("No GPU found. Turn on a GPU accelerator, or expect training to be very slow.")
"""

UNPACK = """
BUNDLE = "__BUNDLE__"
SRC = SCRATCH / "rdq_src"
if not (SRC / "rdq" / "cli.py").exists():
    SRC.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz") as tar:
        tar.extractall(SRC)
print("rdq source ready in", SRC)
"""

ENV = """
# LeRobot 0.6.1 needs Python 3.12. Build a separate environment so the notebook's own Python version doesn't matter.
VENV = SCRATCH / "venv312"
PY = VENV / "bin" / "python"
import re

def torch_index_for_driver():
    # Pick a PyTorch build the GPU driver can run (default PyPI wheels need a CUDA 13 driver).
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        m = re.search(r"CUDA Version: *([0-9]+)[.]([0-9]+)", out)
        version = (int(m.group(1)), int(m.group(2))) if m else None
    except FileNotFoundError:
        version = None
    if version is None or version >= (13, 0):
        return None
    if version >= (12, 8):
        return "https://download.pytorch.org/whl/cu128"
    if version >= (12, 6):
        return "https://download.pytorch.org/whl/cu126"
    return "https://download.pytorch.org/whl/cu118"

if not (VENV / "bin" / "lerobot-train").exists():
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run([sys.executable, "-m", "uv", "venv", "--python", "3.12", VENV])
    index = torch_index_for_driver() if DEVICE == "cuda" else None
    if index:
        print("installing PyTorch from", index)
        run([sys.executable, "-m", "uv", "pip", "install", "--python", PY,
             "torch>=2.7,<2.12", "torchvision>=0.22,<0.27", "--index-url", index])
    run([sys.executable, "-m", "uv", "pip", "install", "--python", PY, "lerobot[pusht,training,diffusion]==0.6.1", "pyyaml"])
run([PY, "-c", "import torch, lerobot; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| lerobot', lerobot.__version__)"])
if DEVICE == "cuda":
    ok = subprocess.run([PY, "-c", "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"]).returncode == 0
    assert ok, "PyTorch cannot see the GPU. Delete /tmp/rdq/venv312 and rerun this cell, or report the nvidia-smi output."
"""

DATA = """
CLEAN = SCRATCH / "pusht_clean"
if not (CLEAN / "meta" / "info.json").exists():
    run([PY, "-c",
         "from huggingface_hub import snapshot_download; "
         f"snapshot_download('lerobot/pusht_keypoints', repo_type='dataset', local_dir=r'{CLEAN}', "
         "allow_patterns=['meta/**', 'data/**', 'README.md'])"])
info = json.load(open(CLEAN / "meta" / "info.json"))
print(info["codebase_version"], info["total_episodes"], "episodes |", {k: v["shape"] for k, v in info["features"].items()})
assert not any(v.get("dtype") in ("video", "image") for v in info["features"].values()), "expected the keypoint dataset with no cameras"
"""

CORRUPT = """
CORRUPT = SCRATCH / "pusht_corrupt"
if not (CORRUPT / "corruption.json").exists():
    if CORRUPT.exists():
        shutil.rmtree(CORRUPT)
    run([PY, SRC / "experiments" / "corrupt.py", CLEAN, CORRUPT, "--rate", CORRUPT_RATE, "--seed", SEED])

# Run rdq on the corrupted copy. It sees only the data, never the answer key.
env = dict(os.environ, PYTHONPATH=str(SRC))
run([PY, "-m", "rdq.cli", "check", CORRUPT, "--json", WORK / "rdq_report.json", "--html", WORK / "rdq_report.html",
     "--filter", WORK / "keep.json", "--no-color"], env=env)

truth = {int(k): v for k, v in json.load(open(CORRUPT / "corruption.json"))["corrupted"].items()}
keep = json.load(open(WORK / "keep.json"))["episodes"]
n_eps = info["total_episodes"]
kept = set(keep)
detection = {kind: f"{sum(1 for e, k in truth.items() if k == kind and e not in kept)}/{sum(1 for k in truth.values() if k == kind)}"
             for kind in sorted(set(truth.values()))}
clean_dropped = sum(1 for e in range(n_eps) if e not in truth and e not in kept)
print("faults caught by rdq:", detection)
print(f"clean episodes wrongly dropped: {clean_dropped}/{n_eps - len(truth)}")

rng = random.Random(SEED)
SUBSETS = {
    "raw": list(range(n_eps)),
    "filtered": sorted(keep),
    "random": sorted(rng.sample(range(n_eps), len(keep))),
}
print({k: len(v) for k, v in SUBSETS.items()}, "episodes per policy")
"""

ROBOMIMIC = """
# Natural-noise benchmark on RoboMimic multi-human data (CPU, a few minutes). No faults are injected:
# six operators are labelled better / okay / worse, and each method keeps 50% of the demos.
# Wrapped so that a failure here never stops the Push-T experiment below.
try:
    run([sys.executable, "-m", "uv", "pip", "install", "--python", PY, "-q", "h5py", "scipy"])
    env = dict(os.environ, PYTHONPATH=str(SRC))
    run([PY, SRC / "experiments" / "robomimic_mh.py", "--tasks", "can", "square", "--out", WORK / "robomimic",
         "--cache", SCRATCH / "robomimic_cache"], env=env)
    print(open(WORK / "robomimic" / "robomimic_mh.md").read())
except Exception as exc:
    print("RoboMimic benchmark skipped:", exc)
"""

TRAIN = """
RUNS = SCRATCH / "runs"

def train(name, episodes):
    out = RUNS / name
    model = out / "checkpoints" / "last" / "pretrained_model"
    if model.exists():
        print(name, "already trained")
        return model
    if out.exists():
        shutil.rmtree(out)
    run([VENV / "bin" / "lerobot-train",
         "--dataset.repo_id=local/pusht_corrupt", f"--dataset.root={CORRUPT}",
         "--dataset.episodes=[" + ",".join(map(str, episodes)) + "]",
         "--policy.type=diffusion", f"--policy.device={DEVICE}", "--policy.push_to_hub=false",
         "--policy.horizon=16", "--policy.n_action_steps=8", "--policy.drop_n_last_frames=7",
         "--policy.noise_scheduler_type=DDIM", "--policy.num_inference_steps=10",
         f"--steps={STEPS}", f"--batch_size={BATCH}", f"--save_freq={STEPS}", "--env_eval_freq=0",
         "--log_freq=1000", f"--seed={SEED}", "--wandb.enable=false", "--num_workers=2",
         f"--output_dir={out}", f"--job_name=pusht_{name}"])
    # The optimizer state is only needed to resume training; drop it to save several GB.
    for state in out.glob("checkpoints/*/training_state"):
        shutil.rmtree(state, ignore_errors=True)
    return model

MODELS = {}
for name, episodes in SUBSETS.items():
    t0 = time.time()
    MODELS[name] = train(name, episodes)
    print(f"{name}: done in {(time.time() - t0) / 60:.0f} min", flush=True)
"""

EVAL = """
def evaluate(name, model):
    out = WORK / "eval" / name
    info_path = out / "eval_info.json"
    if not info_path.exists():
        if out.exists():
            shutil.rmtree(out)
        run([VENV / "bin" / "lerobot-eval", f"--policy.path={model}", f"--policy.device={DEVICE}",
             "--env.type=pusht", "--env.obs_type=environment_state_agent_pos",
             f"--eval.n_episodes={EVAL_EPISODES}", f"--eval.batch_size={min(50, EVAL_EPISODES)}",
             "--eval.use_async_envs=false",  # worker processes would not have the Push-T env registered
             f"--seed={SEED + 1000}", f"--output_dir={out}"])
    data = json.load(open(info_path))
    agg = data.get("overall") or data.get("aggregated")
    return agg["pc_success"] / 100, int(agg.get("n_episodes", EVAL_EPISODES))

SUCCESS = {name: evaluate(name, model) for name, model in MODELS.items()}
SUCCESS
"""

RESULTS = """
def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def two_prop_p(p1, n1, p2, n2):
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) or 1e-12
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))  # two-sided

rows = []
for name in ("raw", "random", "filtered"):
    p, n = SUCCESS[name]
    lo, hi = wilson(p, n)
    rows.append({"policy": name, "episodes_trained_on": len(SUBSETS[name]), "success": round(p, 3),
                 "ci95": [round(lo, 3), round(hi, 3)], "eval_episodes": n})
    print(f"{name:9s} trained on {len(SUBSETS[name]):3d} episodes  success {p:.1%}  (95% CI {lo:.1%}–{hi:.1%})")

pf, nf = SUCCESS["filtered"]; pr, nr = SUCCESS["random"]; pw, nw = SUCCESS["raw"]
results = {
    "dataset": "lerobot/pusht_keypoints", "corruption_rate": CORRUPT_RATE, "seed": SEED, "steps": STEPS,
    "batch_size": BATCH, "policy": "diffusion (horizon 16, 8 action steps, DDIM 10)",
    "faults_caught": detection, "clean_wrongly_dropped": f"{clean_dropped}/{n_eps - len(truth)}",
    "policies": rows,
    "p_filtered_vs_random": two_prop_p(pf, nf, pr, nr),
    "p_filtered_vs_raw": two_prop_p(pf, nf, pw, nw),
}
json.dump(results, open(WORK / "results.json", "w"), indent=2)
print()
print(f"filtered vs random: {pf - pr:+.1%}  (p = {results['p_filtered_vs_random']:.3g})")
print(f"filtered vs raw:    {pf - pw:+.1%}  (p = {results['p_filtered_vs_raw']:.3g})")
print("saved", WORK / "results.json")
"""


def build(out: Path) -> None:
    cells = [
        md(INTRO),
        md("## 1 · Settings and GPU check"),
        code(CONFIG),
        md("## 2 · Unpack rdq (embedded in this notebook)"),
        code(UNPACK.replace("__BUNDLE__", bundle())),
        md("## 3 · Install LeRobot in a Python 3.12 environment (about 5 minutes)"),
        code(ENV),
        md("## 4 · Download Push-T (keypoint version, no video)"),
        code(DATA),
        md("## 5 · Corrupt 30% of episodes, then let rdq find them"),
        code(CORRUPT),
        md("## 5b · Natural-noise check on RoboMimic multi-human data (CPU, a few minutes)"),
        code(ROBOMIMIC),
        md("## 6 · Train three policies (the long part)"),
        code(TRAIN),
        md("## 7 · Evaluate each policy in the Push-T simulator"),
        code(EVAL),
        md("## 8 · Results"),
        code(RESULTS),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.write_text(json.dumps(nb, indent=1))


if __name__ == "__main__":
    build(Path(sys.argv[1]))
    print("wrote", sys.argv[1])
