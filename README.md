# rdq: know what's in your robot data before you train on it

`rdq` audits robot-learning datasets in the [LeRobot](https://github.com/huggingface/lerobot) format (v3.0 and v2.x). It answers three questions:

1. **Can I legally use this?** It checks the licence and provenance, including whether the declared licence contradicts the dataset it was converted from.
2. **Is anything broken?** It finds desynced streams, frozen logs, glitches, idle padding, duplicates, stale normalisation stats, schema errors and (optionally) bad camera footage.
3. **Which episodes are worth training on?** It ranks episodes with training-free quality scores from the curation literature, and it says honestly where those scores help and where they don't.

It reads tables, not videos (unless you ask), and runs on a laptop.

```bash
pip install -e .                       # Python 3.10+
rdq check owner/name --html report.html --filter keep.json      # a Hub dataset id (fetches tables + card only)
rdq check path/to/dataset --repo-id owner/name --video          # or a local folder, with camera checks
rdq rank path/to/dataset --keep-top 0.9 --filter keep_ranked.json
rdq passport path/to/dataset --repo-id owner/name --spdx passport.spdx.json --datasheet DATASHEET.md
rdq scorecard path/to/dataset --groups collectors.csv     # quality per collector, operator or task
```

`keep.json` lists the episodes that passed, in the form LeRobot's `--dataset.episodes` accepts.

## Licence and provenance

LeRobot's `push_to_hub()` defaults to `license="apache-2.0"` and writes a template card. In a scan of the 3,000 most-downloaded LeRobot datasets on the Hub (October 2026):

- **56.5%** (51.9% of downloads) carry `apache-2.0` on an unedited template card. That is most likely the tool's default rather than a licence the owner chose.
- **1,167** (70.6% of downloads) can be traced to one of 65 public robot datasets whose terms we checked against the primary source (26 of them at high confidence). **201 of these (32% of downloads) declare a licence that differs from the original's terms**, 144 of them (21.5% of downloads) against a high-confidence source. Most are CC BY 4.0 data relabelled as apache-2.0 or MIT. At least one is a non-commercial, share-alike dataset tagged apache-2.0.
- **45 more** (8.4% of downloads) declare a licence on data whose original release states no data licence at all, only a code licence. ALOHA sim, Push-T and xArm are examples.

`rdq` reports a verdict for every dataset and marks how strong each upstream match is: the card links to the source, the repo name matches, or the card only mentions it. The upstream table lives in [`rdq/data/upstream_licences.json`](rdq/data/upstream_licences.json). Every entry keeps the statement, its URL, the date it was checked and a confidence level. For example, an Open X-Embodiment component that only has the umbrella CC BY 4.0 line counts as medium confidence and never produces a critical finding. A tag on a mirror cannot grant rights the uploader doesn't hold. This is a diligence aid, not legal advice.

### Data passport

`rdq passport` writes a machine-readable record of a dataset in three forms:

- [Croissant 1.1](https://docs.mlcommons.org/croissant/) JSON-LD with the RAI vocabulary and PROV-O. It validates with `mlcroissant`.
- An SPDX 3.0 dataset package, which records the declared and concluded licences separately.
- A Markdown datasheet.

Facts computed from the data are kept separate from statements by the owner. rdq never makes up how data was collected. See [docs/passport.md](docs/passport.md).

## Integrity checks

| Check | Severity | Why it matters |
|---|---|---|
| `sync.action_state_lag` | warning | Actions shifted against observations. Detected from velocities, so steady motion doesn't hide it. |
| `episodes.idle`, `episodes.idle_padding` | critical / warning | The robot never really moves, or the recording kept running long after the task. |
| `episodes.spikes` | warning | Short out-and-back glitches from sensor dropouts. The threshold stays robust when glitches are common. |
| `episodes.duplicate` | warning | Exact copies inflate the dataset. |
| `values.non_finite`, `values.dead_dimension` | critical / info | NaNs, or a channel that never changes. |
| `stats.stale` | warning | `stats.json` doesn't match the data, so normalisation is silently wrong. |
| `schema.*`, `counts.*`, `time.*`, `files.*` | various | Metadata that disagrees with the data, broken timing, orphaned files. |
| `tasks.placeholder` | warning | Language-conditioned policies get a useless instruction. |
| `video.span_mismatch` | warning | A video segment and its data differ in length. |
| `video.frozen`, `video.black`, `video.decode_error`, `video.blur` | with `--video` | Six frames per episode per camera are decoded to catch stalled, covered or broken cameras. |

On 120 clean synthetic datasets the checks raise no warnings or critical findings. CI runs this test on every commit (`tests/false_alarms.py`).

## Episode quality scores (`rdq rank`)

These scores need no training, are cheap to compute, and come from published methods:

- **DemInf-lite**: each episode's contribution to state-action mutual information, using a KSG estimator. Based on [Hejna et al., RSS 2025](https://arxiv.org/abs/2502.08623), without the VAE.
- **Consistency**: a leave-one-episode-out kNN behaviour-cloning residual. Frames in multimodal regions are reported as a *minority strategy* rather than dropped (see [Belkhale et al. 2023](https://arxiv.org/abs/2306.02437) and [CUPID](https://arxiv.org/abs/2506.19121)).
- **Smoothness** (high-frequency ratio, log dimensionless jerk), hesitation, gripper cycles, unusual final states, and calibrated near-duplicate detection.
- **The spectral metric of [Sojib et al. 2026](https://arxiv.org/abs/2605.01544)**, reproduced as a baseline. By Parseval's theorem it scales with episode length, so rdq always reports length alongside it.
- **Idle trimming** following the openpi DROID recipe (`--keep-ranges`).

The composite score is the mean rank of DemInf-lite and consistency. If it correlates strongly with episode length, rdq warns you.

## Evidence

`experiments/curation_benchmark.py` is a controlled CPU benchmark: 200 demonstrations at 30 fps, 5 seeds, 400 evaluation rollouts each. It compares selection methods against an oracle, random subsets, keep-the-shortest, the published spectral metric and a loss-based filter.

| Data (200 episodes) | All data | Random subset | Keep shortest | Spectral (PSD) | DemInf-lite | rdq composite | rdq rules | Oracle |
|---|---|---|---|---|---|---|---|---|
| 30% recording faults (desync, frozen log, glitch) | 0.16 | 0.19 | 0.21 | 0.21 | 0.93 | 0.92 | **1.00** | 1.00 |
| 40% imperfect operators (shaky, hesitant, detours, failed) | 0.88 | 0.74 | 0.89 | 0.88 | 0.70 | 0.84 | 0.88 | 0.99 |
| Both (15% faults + 20% imperfect operators) | 0.34 | 0.37 | 0.24 | 0.27 | 0.88 | **0.93** | 0.92 | 1.00 |

*Success rates. Full tables, confidence intervals and detection AUROCs are in [`experiments/results/curation_benchmark.md`](experiments/results/curation_benchmark.md).*

What this shows:

- **Recording faults are the expensive problem, and rules fix them.** Thirty percent faulty episodes cut success from 1.00 to 0.16. rdq's rules recovered the oracle exactly and dropped no clean episodes. Length and smoothness metrics did no better than random.
- **Operator skill is a different problem.** Behaviour cloning was fairly robust to imperfect operators (0.88 with everything), and no training-free score reliably beat training on all the data. Keep-the-shortest was as good as any. rdq therefore reports skill scores for review and does not drop episodes based on them.
- **The honest limit:** this is a toy task. The real tests are `experiments/kaggle/` (diffusion policy on Push-T with injected faults) and `experiments/robomimic_mh.py`. The latter runs on RoboMimic's multi-operator data, which has natural quality labels, and is directly comparable with published DemInf, CUPID and spectral numbers.

## Hub audit

```bash
rdq scan --limit 3000 --out results                      # licence + metadata, two small files per dataset
rdq audit --from-scan results --n 300 --out results      # full checks on a sample, no videos
python experiments/analyze_hub.py results --out summary.json
```

## Hugging Face Space

`space/app.py` is a Gradio app. You paste a dataset id and get the licence and provenance verdict in seconds, plus optional full checks, a report and a passport.

## Licence

Apache-2.0 for the code. Hub figures describe public dataset cards at scan time.
