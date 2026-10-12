"""Build a realistic *synthetic* vendor delivery and every rdq output for it.

This is the sample report to show a prospective customer before they share real data.
Three contractors recorded 150 SO-101 pick-and-place demonstrations. Contractor C's rig
has a laggy action logger and a camera-side recorder that sometimes freezes, contractor B
has a shaky leader arm, and one episode was uploaded twice. The card is LeRobot's untouched
template with the default licence. Everything here is generated; no real data is used.

    python examples/sample_delivery.py --out examples/sample_delivery_output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq import synthetic as syn  # noqa: E402
from rdq.dataset import load_dataset  # noqa: E402
from rdq.passport import build_passport, to_datasheet, to_spdx  # noqa: E402
from rdq.report import audit, write_html, write_json  # noqa: E402
from rdq.scorecard import episode_groups, format_scorecard, scorecard  # noqa: E402
from rdq.scores import score_episodes  # noqa: E402

CARD = """---
license: apache-2.0
tags:
- LeRobot
task_categories:
- robotics
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0
"""


def build(out: Path) -> dict:
    rng = np.random.default_rng(7)
    eps = syn.make_episodes(n_episodes=150, length=240, fps=30, seed=7, lag=2)
    collectors = ["contractor_A"] * 50 + ["contractor_B"] * 50 + ["contractor_C"] * 50
    truth: dict[int, str] = {}
    for i in range(50, 100):  # B: shaky leader arm
        shake = rng.normal(0, 2.5, eps[i]["action"].shape)
        eps[i]["action"] = eps[i]["action"] + shake
        eps[i]["state"] = eps[i]["state"] + shake * 0.8
        truth[i] = "shaky"
    for j, i in enumerate(range(100, 150)):  # C: logger and recorder problems
        if j % 5 == 0:
            syn.delay_action(eps[i], frames=10)
            truth[i] = "desync"
        elif j % 5 == 1:
            syn.make_idle(eps[i])
            truth[i] = "frozen_log"
        elif j % 5 == 2:
            syn.add_glitch(eps[i], frames=3)
            truth[i] = "glitch"
    eps[149] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in eps[10].items()}  # double upload
    truth[149] = "duplicate"
    root = syn.write_dataset(out / "dataset", eps, fps=30, tasks=["Pick up the red cube and place it in the bowl"])
    (root / "README.md").write_text(CARD)
    (out / "collectors.csv").write_text("episode_index,group\n" + "".join(f"{i},{c}\n" for i, c in enumerate(collectors)))

    result = audit(root, repo_id="acme-robotics/so101_pick_place_delivery_03")
    result["dataset"]["path"] = "acme-robotics/so101_pick_place_delivery_03"
    write_json(result, out / "report.json")
    write_html(result, out / "report.html")
    scores = score_episodes(load_dataset(root))
    rows = scorecard(result, scores, episode_groups(load_dataset(root), mapping_csv=out / "collectors.csv"))
    (out / "scorecard.txt").write_text(format_scorecard(rows) + "\n")
    passport = build_passport(result, CARD, root, repo_id="acme-robotics/so101_pick_place_delivery_03", revision="1.0.0",
                              answers={"collection_method": "Leader-arm teleoperation (SO-101 leader to follower), 30 Hz",
                                       "operators": "3 contractors (A, B, C), operator id recorded per episode in collectors.csv",
                                       "intended_use": "Fine-tuning pick-and-place policies"})
    (out / "passport.json").write_text(json.dumps(passport, indent=2, default=str))
    (out / "passport.spdx.json").write_text(json.dumps(to_spdx(passport), indent=2, default=str))
    (out / "DATASHEET.md").write_text(to_datasheet(passport))
    usable = set(result["score"]["usable_episodes"])
    caught = {k: sum(1 for i, t in truth.items() if t == k and i not in usable) for k in set(truth.values())}
    planted = {k: sum(1 for t in truth.values() if t == k) for k in set(truth.values())}
    clean_dropped = sum(1 for i in range(150) if i not in truth and i not in usable)
    summary = {"planted": planted, "dropped_by_rules": caught, "clean_dropped": clean_dropped,
               "score": result["score"]["dataset_score"], "usable": len(usable)}
    (out / "ground_truth.json").write_text(json.dumps({"truth": {str(k): v for k, v in truth.items()}, "summary": summary}, indent=1))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="examples/sample_delivery_output")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(out), indent=1))
    print((out / "scorecard.txt").read_text())
