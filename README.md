# rdq: quality and provenance checks for robot-learning data

`rdq` audits robot datasets in the [LeRobot](https://github.com/huggingface/lerobot) format (v3.0 and v2.x) **before** you spend GPU hours training on them. It reads only the tables (actions, states, timestamps, metadata), never decodes video, and runs on a laptop.

```bash
pip install -e .
rdq check path/to/dataset --html report.html --filter keep.json
```

`keep.json` lists the episodes that passed. It can be passed straight to LeRobot's `episodes=` argument to train on the clean subset.

## What it checks

| Check | Severity | Why it matters for training |
|---|---|---|
| `schema.shape_mismatch`, `schema.names_shape_mismatch` | critical / warning | The declared feature shape doesn't match the data or its joint names. |
| `counts.*` | warning / critical | info.json and episode metadata disagree with the data. |
| `tasks.placeholder`, `tasks.unknown_index` | warning / critical | Language-conditioned policies get a useless or missing instruction. |
| `values.non_finite`, `values.dead_dimension` | critical / warning | NaNs, or a motor or sensor channel that never changes (disconnected). |
| `stats.stale` | warning | stats.json doesn't match the data, so normalization is silently wrong. |
| `episodes.duplicate` | warning | Exact copies inflate the dataset and bias training. |
| `video.span_mismatch` | critical | A video segment and the recorded actions have different lengths, so frames drift away from the actions. |
| `time.non_monotonic`, `time.fps_mismatch`, `time.dropped_frames` | critical / warning | Broken or irregular timing. |
| `episodes.idle`, `episodes.idle_padding` | critical / warning | The robot never really moves, or the recording kept running long after the task. |
| `episodes.too_short`, `episodes.too_long` | warning / info | Aborted or runaway recordings. |
| `episodes.spikes` | warning | One-frame jumps from sensor glitches or teleop dropouts. |
| `sync.action_state_lag` | warning | Actions shifted against observations in some episodes. |
| `episodes.trajectory_outlier` | info | An episode moves unlike the rest: maybe a different task, a different setup, or a failed attempt. |

Licence and provenance come from the dataset card. LeRobot's upload tool defaults to `license: apache-2.0` and an unedited template card, so a template card plus apache-2.0 is reported as **"default licence, unconfirmed"**. The tool also flags non-commercial licences and mentions of restricted upstream sources such as EPIC-KITCHENS or Ego4D.

Every point taken off the score traces back to a named finding. Thresholds are listed in the JSON report.

## Validation

- `tests/` plants each fault in synthetic datasets and checks that the right check flags the right episode.
- On 120 clean synthetic datasets, there were zero critical or warning false alarms.
- Whether the score predicts how well a policy trains is the open question. `experiments/` tests it.

## Hub audit

```bash
rdq scan --limit 3000 --out results        # licence + metadata, two small files per dataset
rdq audit --from-scan results --n 150 --out results   # full checks on a sample, no videos
```
