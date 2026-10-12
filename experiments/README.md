# Experiments

The claim to test is: **training on the episodes rdq marks usable produces better policies than training on everything, and better than training on a random subset of the same size.**

The random-subset control is what makes this a test of rdq. Without it, any gain could just come from training on less data.

## 1. Toy experiment (done, CPU only)

`toy_reach.py` uses a 2D position-controlled robot reaching random goals.

- Expert demos are written in LeRobot v3.0 format.
- A fraction of episodes get realistic faults: idle (the log froze), desync (actions written 8 frames late), glitch (3-frame teleop dropout) and wrong goal (a failed attempt).
- rdq has no way to detect a wrong-goal episode. It is included on purpose, as an honest limit.
- The policy is a numpy MLP trained by behaviour cloning, evaluated on 400 rollouts per seed with 3 seeds.

```bash
python experiments/toy_reach.py --episodes 200 --rates 0 0.1 0.2 0.3 0.4 --seeds 3 --rollouts 400
```

Results are in `results/toy_results.json`. Success rate averaged over 3 seeds × 400 rollouts:

| Corrupted episodes | Raw | Random subset (same size) | rdq-filtered | Dataset score |
|---|---|---|---|---|
| 0% | 1.00 | 1.00 | 1.00 | 100 |
| 10% | 0.77 | 0.81 | 0.98 | 96 |
| 20% | 0.49 | 0.48 | 0.94 | 93 |
| 30% | 0.24 | 0.27 | 0.92 | 90 |
| 40% | 0.22 | 0.21 | 0.84 | 87 |

Detection:

- Idle, desync and glitch: 150/150 each.
- Wrong goal: 0/150, as expected.
- Clean episodes wrongly dropped: 10/2,400 (0.4%). These are demos whose goal was already within reach, so they barely move.
- Correlation between the dataset score and raw-training success across all 15 runs: 0.97.

**What this shows and doesn't.** The mechanism works: the faults rdq detects really do damage behaviour cloning, and removing exactly those episodes recovers most of the performance. But this is a toy task, a toy policy and faults we designed ourselves. The 0.97 correlation is inflated because corruption rate drives both the score and the success. Nothing here proves the effect on real robot data.

## 2. Real experiment (next, needs one GPU)

**Setup.**

- Policy: SmolVLA (`lerobot/smolvla_base`, 450M parameters), fine-tuned with LeRobot.
- Data: a LIBERO dataset in LeRobot v3.0 format, with simulation evaluation through `lerobot-eval --env.type=libero`.

**Protocol.**

1. Take a clean LIBERO training set, for example the libero_spatial portion.
2. `python experiments/corrupt.py CLEAN CORRUPT --rate 0.3 --seed 0`. Ground truth goes to `corruption.json`.
3. `rdq check CORRUPT --filter keep.json`, then record detection against the ground truth.
4. Train three SmolVLA policies, identical except for `--dataset.episodes`:
   - raw: all episodes
   - filtered: the episodes in `keep.json`
   - random: a random subset the same size as `keep.json`
5. Evaluate each on libero_spatial: 10 tasks × 50 episodes = 500 rollouts per policy. Report success with Wilson 95% intervals, and test filtered vs. random with a paired two-proportion test.
6. Repeat with a second seed if the budget allows.

**Commands** (one A100 or L4; the dataset repo name is filled in from the hub scan):

```bash
pip install "lerobot[smolvla,libero]"
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=LOCAL/libero_corrupt \
  --dataset.root=CORRUPT --dataset.episodes="[$(jq -r '.episodes|join(",")' keep.json)]" \
  --steps=20000 --batch_size=32 --output_dir=runs/filtered --job_name=filtered
lerobot-eval --policy.path=runs/filtered/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_spatial --eval.n_episodes=50 --eval.batch_size=10
```

**Budget.** A SmolVLA fine-tune of about 20k steps takes a few GPU-hours. Three policies plus evaluation comes to roughly 10–15 GPU-hours, about $15–40 on a rented A100 or L4. Check current prices before renting.

**Kill criterion.** If filtered is not clearly better than random on real data, rdq is a linter, not a product. In that case, pivot toward provenance/licence auditing or evaluation statistics, which reuse this code.
