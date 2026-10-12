# Experiments

The question is whether rdq's verdicts lead to better policies. Every experiment compares rdq against controls that would explain a gain without rdq: a random subset of the same size, keep-the-shortest (length is a well-known confound in curation metrics), the published spectral metric, a loss-based filter and an oracle.

| Experiment | Data | Noise | Needs | Status |
|---|---|---|---|---|
| `toy_reach.py` | 2D reaching, 10 fps | injected faults | CPU, minutes | done ([results](results/toy_results.json)) |
| `curation_benchmark.py` | 2D reaching, 30 fps | injected faults **and** imperfect operators | CPU, ~15 min | done ([results](results/curation_benchmark.md)) |
| `robomimic_mh.py` | RoboMimic Can and Square, multi-human | **natural**: six operators labelled better, okay or worse | CPU, minutes; downloads ~230 MB | ready, needs internet to the Hub |
| `kaggle/rdq_pusht_experiment.ipynb` | LeRobot Push-T (keypoints) | injected faults | one GPU, 4–6 h | run in progress on Kaggle |

## 1. Toy reaching (`toy_reach.py`)

This is the first mechanism test: 200 expert demos, a share of them corrupted (idle, 8-frame desync, 3-frame glitch, wrong goal), and behaviour cloning trained on raw, rdq-filtered and random subsets. At 30% corruption the success rates were raw 0.24, random 0.27 and filtered 0.92. The wrong-goal episodes look like normal motion and were never caught, which is an honest limit of rule-based checks.

## 2. Curation benchmark (`curation_benchmark.py`)

This is the version reviewers would ask for. Three scenarios are each run with 5 seeds and 400 rollouts:

- **structural**: 30% recording faults.
- **skill**: 40% imperfect operators (shaky hands, hesitation with pauses and back-and-forth, detours, failed attempts that stop short).
- **mixed**: 15% faults plus 20% imperfect operators.

Every subset method keeps the number of good episodes. Rules choose their own size and are compared with a random subset of that size. Detection AUROC per fault type is reported next to success.

The main result: recording faults are devastating (0.16 success with all data) and rdq's rules recover the oracle exactly. Operator skill hurts much less (0.88 with all data), and no training-free score reliably improves on using all the data. Length alone is as good as any of them. See [results/curation_benchmark.md](results/curation_benchmark.md).

## 3. RoboMimic MH (`robomimic_mh.py`)

This benchmark has no injected faults. Each method keeps 50% of the 300 demonstrations, and the score is the mean operator tier of what it kept (better = 3, okay = 2, worse = 1), following Sojib et al. (2026), Table I. The published Can-MH numbers at 50% are random 1.99, DemInf 2.37, CUPID 2.32, PSD 2.53 and oracle 2.67. The script reports rdq's scores, the replicated PSD and keep-the-shortest on the same footing.

```bash
pip install -e . h5py
python experiments/robomimic_mh.py --tasks can square --out experiments/results
```

You can also run it from the repo's **Actions → benchmarks** workflow, which has internet access.

## 4. Push-T on a GPU (`kaggle/`)

The setup is LeRobot diffusion policy on `lerobot/pusht_keypoints` (206 human demos). 30% of episodes are corrupted with the same three recording faults, and three policies (raw, rdq-filtered, random of the same size) are each evaluated on 200 simulator episodes. `build_notebook.py` embeds the current rdq source in a single notebook. `smoke_test.py` runs every cell on CPU against a synthetic Push-T-shaped dataset.

**Read the result honestly.** An independent replication of heuristic curation on Push-T (LorenzoDolfi/robot-data-quality) found that heuristics did worse than a random subset (24% vs 27%), and both did worse than the full dataset (41%). Diffusion policy is known to be fairly robust to noisy demos, and removing 30% of 206 demos also removes coverage. A null result here would be informative, not embarrassing.

## Kill criterion

If rdq's selection doesn't beat a same-size random subset on real data, its value is in licence and provenance and in catching broken recordings, not in curation. The roadmap already reflects that.
