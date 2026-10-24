"""Does an episode whose metadata names the wrong data file (lerobot#2401) load when you ask LeRobot for it?

    python experiments/repro_2401.py owner/name BAD_EP GOOD_EP
Installs lerobot 0.6.1, then loads LeRobotDataset(repo, episodes=[ep]) from the Hub for each episode in a fresh
cache and reports frames loaded vs the metadata's length, and whether reading the first frame works.
Also runs rdq on the tables to show the finding.
"""
import json
import subprocess
import sys
import tempfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lerobot[dataset]==0.6.1"], check=True)
repo, eps = sys.argv[1], [int(x) for x in sys.argv[2:]]
code = r'''
import json, sys, traceback
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo, ep, root = sys.argv[1], int(sys.argv[2]), sys.argv[3]
out = {"episode": ep}
try:
    ds = LeRobotDataset(repo, episodes=[ep], root=root, download_videos=False)
    out["meta_length"] = int(ds.meta.episodes[ep]["length"])
    out["frames_loaded"] = len(ds.hf_dataset)
    try:
        item = ds[0]
        out["first_item_episode"] = int(item["episode_index"])
    except Exception as e:
        out["getitem_error"] = f"{type(e).__name__}: {str(e)[:200]}"
except Exception as e:
    out["load_error"] = f"{type(e).__name__}: {str(e)[:300]}"
print("RESULT " + json.dumps(out))
'''
for ep in eps:
    with tempfile.TemporaryDirectory() as root:
        p = subprocess.run([sys.executable, "-c", code, repo, str(ep), root + "/ds"], capture_output=True, text=True)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), "RESULT " + json.dumps({"episode": ep, "crash": (p.stderr or "")[-400:]}))
        print(f"::notice title=r2401-{ep}::" + line[7:][:3500], flush=True)
p = subprocess.run([sys.executable, "-m", "rdq.cli", "check", repo, "--json", "/tmp/r.json", "--no-color"], capture_output=True, text=True)
try:
    f = [x for x in json.load(open("/tmp/r.json"))["findings"] if x["check"] == "meta.data_file"]
    print("::notice title=r2401-rdq::" + json.dumps(f)[:3500])
except Exception as e:  # noqa: BLE001
    print("::notice title=r2401-rdq::error " + str(e) + (p.stderr or "")[-300:])
