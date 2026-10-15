# rdq: know what's in your robot data before you train on it

`rdq` audits robot-learning datasets in the [LeRobot](https://github.com/huggingface/lerobot) format (v3.0 and v2.x). It answers three questions:

1. **Can I legally use this?** It checks the licence and provenance, including whether the declared licence contradicts the dataset it was converted from.
2. **Is anything broken?** It finds desynced streams, frozen logs, glitches, idle padding, duplicates, stale normalisation stats, schema errors and (optionally) bad camera footage.
3. **Which episodes are worth training on?** It ranks episodes with training-free quality scores from the curation literature, and it says honestly where those scores help and where they don't.

**Try the licence check in your browser, no install:** [huggingface.co/spaces/rs1g6r/rdq](https://huggingface.co/spaces/rs1g6r/rdq)

It reads tables, not videos (unless you ask), and runs on a laptop for typical datasets: about 1.4 GB of memory for 600,000 frames. For DROID-scale corpora, audit a sample of episodes.

```bash
pip install git+https://github.com/s1g6r/rdq    # Python 3.10+
rdq check owner/name --html report.html --filter keep.json      # a Hub dataset id (fetches tables + card only)
rdq check path/to/dataset --repo-id owner/name --video          # or a local folder, with camera checks
rdq check owner/name --video-full                               # decode every frame: frozen/black/blurry stretches, duplicated episodes
rdq rank path/to/dataset --json scores.json                     # rank episodes for review
rdq passport path/to/dataset --repo-id owner/name --spdx passport.spdx.json --datasheet DATASHEET.md
rdq scorecard path/to/dataset --groups collectors.csv     # quality per collector, operator or task
rdq check-mcap delivery/*.mcap --state-topic /joint_states --action-topic /leader/joint_states   # audit MCAP before converting
rdq check-video deliveries/oct/ --by-folder --html video.html   # raw video: usable hours, defects, copies, per contributor
```

`keep.json` is `{"episodes": [...]}`, the episodes with no critical or warning finding. Pass the list to LeRobot with `--dataset.episodes="$(jq -c .episodes keep.json)"`.

## Licence and provenance

LeRobot's `push_to_hub()` defaults to `license="apache-2.0"` and writes a template card. In a scan of the 3,000 most-downloaded LeRobot datasets on the Hub (October 2026):

- **56.5%** (51.9% of downloads) carry `apache-2.0` on an unedited template card. That is most likely the tool's default rather than a licence the owner chose.
- **304** (51.8% of downloads) can be matched to one of 65 public robot datasets whose terms we checked against the primary source (26 of them at high confidence). Uploads by an original publisher's own organisation (862: RoboCOIN, Unitree and TacVerse) are counted separately, because their tag is theirs to choose.
- **200 of those 304 (32% of downloads) declare a licence that differs from the original's terms**, 143 of them (21.5% of downloads) against a high-confidence source. 190 are CC BY 4.0 data relabelled as apache-2.0 or MIT. One, matched by name, is a sample of a non-commercial, share-alike dataset tagged apache-2.0.
- How strong the matches are: 189 of the 200 are matched by repository name alone (such as `droid_lerobot` or `bridge_v2_lerobot`), 10 by a link to the original in a derivation context and 1 by a mention. For the 17 most-downloaded name matches we compared robot type and episode count with the original release: the robot matches in all 17, 13 have the original's episode count, 3 are subsets and 1 is a 6-episode sample. In a random sample of 30 of the other name matches, 26 are copies or subsets of the original (same episode counts or splits, same robot); the other 4 are simulation data that may have been newly generated in the same simulator, which the original's licence would not cover. Because no single signal is treated as proof, none of these is reported as critical.
- **45 more** (8.4% of downloads) declare a licence on data whose original release states no data licence at all, only a code licence. ALOHA sim, Push-T and xArm are examples.

`rdq` reports a verdict for every dataset and marks how strong each upstream match is, from strongest to weakest: the card's `source_datasets` names the source, the card links to it in a derivation context (a "Homepage" field, "converted from"), the repo name matches, or the card only mentions or cites it. Each source is judged on its own evidence and confidence, and the most restrictive term wins. A finding is critical only when the card declares the source, or when two independent signals agree (for example a derivation link and the repo name). A single signal, such as a repo name alone, is at most a warning. A mention in a citation context (a pretrained encoder, a benchmark, a format) is not a match, and uploads by the original publisher are informational. The upstream table lives in [`rdq/data/upstream_licences.json`](rdq/data/upstream_licences.json). Every entry keeps the statement, its URL, the date it was checked and a confidence level. For example, an Open X-Embodiment component that only has the umbrella CC BY 4.0 line counts as medium confidence and never produces a critical finding. A tag on a mirror cannot grant rights the uploader doesn't hold. This is a diligence aid, not legal advice. The full rules, with how each verdict is reached and how it was validated, are in [docs/licence-methodology.md](docs/licence-methodology.md).

### Data passport

`rdq passport` writes a machine-readable record of a dataset in three forms:

- [Croissant 1.1](https://docs.mlcommons.org/croissant/) JSON-LD with the RAI vocabulary and PROV-O. It validates with `mlcroissant`, and its record set lets Croissant loaders enumerate frames by episode, frame index and timestamp. Action and state vectors are described in the passport but not yet as Croissant fields.
- An SPDX 3.0 dataset package, which records the declared and concluded licences separately.
- A Markdown datasheet.

Facts computed from the data are kept separate from statements by the owner. rdq never makes up how data was collected. See [docs/passport.md](docs/passport.md).

## Raw video deliveries (`rdq check-video`)

Egocentric and teleop footage is bought by the hour, and not every hour is usable. `rdq check-video` reads a folder of video files (one sub-folder per contributor, or a CSV) and reports:

- **usable hours**, per file and per contributor: footage minus black, covered, washed-out, frozen, idle and blurry stretches, recording gaps, unreadable or truncated files, and copies;
- **where each problem is**, with times, so a reviewer can jump straight to it;
- **copies**: re-uploads, re-encodes, mirrored or cut copies, the same stream in another container, footage looped within a file, and (with `--store`) copies of earlier deliveries. Takes of the same task from a fixed camera look alike frame by frame; a copy also repeats the same movements at the same moments, and that is what rdq checks before calling anything a copy;
- a **per-contributor scorecard** and an optional delivery spec (resolution, fps, codec, duration, maximum idle and blur share).

On synthetic deliveries with known answers it finds every planted defect class and every copy (re-encoded, cut, mirrored, 25 fps) with no false copies, and estimates usable hours within a few percent. On 60 real episodes from ten public LeRobot datasets (many takes of one task from one fixed camera), and on 72 held-out episodes from twelve more, it reported nothing false and found every planted copy. Details, thresholds and limits: [docs/raw-video.md](docs/raw-video.md). Needs `pip install "rdq[video]"`.

## Integrity checks

| Check | Severity | Why it matters |
|---|---|---|
| `sync.action_state_lag` | warning | Actions shifted against observations. Detected from velocities, so steady motion doesn't hide it. |
| `episodes.idle`, `episodes.idle_padding` | critical / warning | The robot barely moves compared with other episodes of the same task, or the recording kept running long after the task. |
| `episodes.spikes` | warning | Short out-and-back glitches from sensor dropouts. In datasets of 8 or more episodes, a channel that does this in most episodes (fast gripper pulses, quantised encoders, or a fault in every session) is reported once as a dataset-level warning, `episodes.spikes_systematic`, instead of excluding every episode. |
| `episodes.duplicate` | warning | Exact copies inflate the dataset. |
| `values.non_finite`, `values.empty_channel`, `values.empty_feature`, `values.dead_dimension` | critical / warning / critical / info | NaNs in some frames, a channel that is never finite (reported once, not per episode, and dropped from both action and state so the sync check still runs), an action or state with no finite value at all, or a channel that never changes. |
| `episodes.no_motion` | critical | Most episodes' state (or absolute-position action) moves no further than its frame-to-frame noise: a frozen stream or a disconnected arm. |
| `counts.episodes_not_in_metadata`, `counts.metadata_episodes_missing` | critical | Episodes in the data files that LeRobot's metadata doesn't list, or the reverse. |
| `stats.stale` | warning | `stats.json` doesn't match the data, so normalisation is silently wrong. |
| `schema.*`, `counts.*`, `time.*`, `files.*` | various | Metadata that disagrees with the data, broken timing, orphaned files. |
| `tasks.placeholder` | warning | Language-conditioned policies get a useless instruction. |
| `video.span_mismatch` | warning | A video segment and its data differ in length. |
| `video.frozen`, `video.black`, `video.decode_error`, `video.blur` | with `--video` | Six frames per episode per camera are decoded to catch stalled, covered or broken cameras. |

Features that are not joint signals (depth maps, point clouds, audio, tactile grids wider than 1,024 values) are listed as not checked rather than loaded. Every report ends with a "Not checked" list, for example when the sync check can't run because action and state have different sizes and share fewer than three channel names. (When the sizes differ but the names line up, for example a state that also carries velocities, the sync check compares the matching channels.)

On 120 clean synthetic datasets the checks raise no warnings or critical findings, and CI repeats this on 40 datasets on every commit (`tests/false_alarms.py`). The clean data comes from the same generator the thresholds were tuned on, so this shows the checks are quiet on well-behaved data, not that they are quiet on yours. On 210 real Hub datasets, an earlier version of the checks raised a warning or critical finding on 58%; a re-audit with the current version is pending.

## Episode quality scores (`rdq rank`)

These scores need no training, are cheap to compute, and come from published methods:

- **DemInf-lite**: each episode's contribution to state-action mutual information, using a KSG estimator. Based on [Hejna et al., RSS 2025](https://arxiv.org/abs/2502.08623), without the VAE.
- **Consistency**: a leave-one-episode-out kNN behaviour-cloning residual. Frames in multimodal regions are reported as a *minority strategy* rather than dropped (see [Belkhale et al. 2023](https://arxiv.org/abs/2306.02437) and [CUPID](https://arxiv.org/abs/2506.19121)).
- **Smoothness** (high-frequency ratio, log dimensionless jerk), hesitation, gripper cycles, unusual final states, and calibrated near-duplicate detection.
- **The spectral metric of [Sojib et al. 2026](https://arxiv.org/abs/2605.01544)**, reproduced as a baseline. By Parseval's theorem it scales with episode length, so rdq always reports length alongside it.
- **Idle trimming** following the openpi DROID recipe (`--keep-ranges`).

The composite score is the mean rank of DemInf-lite and consistency. If it correlates strongly with episode length, rdq warns you. These two components were chosen after seeing the benchmark below (a first version that also averaged smoothness and hesitation scored 0.82 on the mixed scenario), so the benchmark is not an independent test of the composite. The independent test is RoboMimic (below), where the composite did better than random but worse than simply keeping the shortest demonstrations. Without an `observation.state`, DemInf-lite and consistency are only computed when other observations exist to condition on.

`rdq rank --keep-top` writes the best-scoring fraction of episodes. Treat it as experimental: in the benchmark, dropping episodes on soft scores never reliably beat training on all clean data.

## Evidence

`experiments/curation_benchmark.py` is a controlled CPU benchmark: 200 demonstrations at 30 fps, 5 seeds, 400 evaluation rollouts each. It compares selection methods against an oracle, random subsets, keep-the-shortest, the published spectral metric and a loss-based filter.

| Data (200 episodes) | All data | Random subset | Keep shortest | Spectral (PSD) | DemInf-lite | rdq composite | rdq rules | Oracle |
|---|---|---|---|---|---|---|---|---|
| 30% recording faults (desync, frozen log, glitch) | 0.16 | 0.19 | 0.21 | 0.21 | 0.89 | 0.92 | **1.00** | 1.00 |
| 40% imperfect operators (shaky, hesitant, detours, failed) | 0.88 | 0.74 | 0.89 | 0.88 | 0.73 | 0.83 | 0.88 | 0.99 |
| Both (15% faults + 20% imperfect operators) | 0.34 | 0.37 | 0.24 | 0.27 | 0.87 | 0.93 | 0.92 | 1.00 |

*Success rates, mean of 5 seeds. Full tables, 95% confidence intervals over seeds and detection AUROCs are in [`experiments/results/curation_benchmark.md`](experiments/results/curation_benchmark.md). Seed-to-seed variation is large for the soft scores: on "Both", the composite's interval is 0.87–0.99 and the rules' is 0.90–0.94, so the two are not distinguishable.*

What this shows:

- **Recording faults are the expensive problem, and rules fix them.** Thirty percent faulty episodes cut success from 1.00 to 0.16. rdq's rules recovered the oracle (0.997, interval 0.992–1.000) and dropped no clean episodes. Length and smoothness metrics did no better than random.
- **Operator skill is a different problem.** Behaviour cloning was fairly robust to imperfect operators (0.88 with everything), and no training-free score reliably beat training on all the data. Keep-the-shortest (0.89) and all data (0.88) were statistically tied, and every soft score did worse. rdq therefore reports skill scores for review and does not drop episodes based on them.
- **The honest limit:** this is a toy task, so we ran two real tests.

### Real data

**RoboMimic multi-operator demos** (`experiments/robomimic_mh.py`, results in [`experiments/results/robomimic/`](experiments/results/robomimic/robomimic_mh.md)). Each method keeps 50% of 300 demonstrations, scored by the operators' published skill tier (3 = better, 1 = worse). Published numbers on Can are from Sojib et al. 2026.

| Method | Can | Square |
|---|---|---|
| Random | 2.00 | 2.00 |
| rdq composite | 2.16 | 2.15 |
| rdq DemInf-lite | 2.19 | 2.13 |
| Published DemInf (Hejna et al., with a trained VAE) | 2.37 | – |
| Published CUPID (needs policy rollouts) | 2.32 | – |
| Keep the shortest demos | 2.53 | 2.47 |
| Spectral metric (our replication; published 2.53) | 2.53 | 2.47 |
| Oracle | 2.67 | 2.67 |

Our replication of the spectral metric matches its published number exactly, so the setup is sound. On this data, demonstration length is the strongest training-free skill signal, and the spectral metric is length in disguise (Spearman 1.00 with length). rdq's lightweight DemInf and composite beat random but not the published methods or length. So `rdq rank` reports length next to every score, and rdq does not claim its skill scores beat the state of the art. The rules dropped 1 of 600 demos, as they should on clean data.

**Push-T with a diffusion policy** (`experiments/kaggle/`, results in [`experiments/results/pusht_kaggle_v1.json`](experiments/results/pusht_kaggle_v1.json)). rdq caught all 62 faults injected into real Push-T demos (21 desync, 20 glitch, 21 idle) and dropped none of the 144 clean ones. The policies were undertrained (30k steps on a free GPU), so success rates are at the floor (2.0% filtered, 0.5% all data, 0.0% random) and do not separate the methods. Average best coverage orders them filtered 0.44, all data 0.34, random 0.26, from one seed.

## Hub audit

```bash
rdq scan --limit 3000 --out results                      # licence + metadata, two small files per dataset
rdq audit --from-scan results --n 300 --out results      # full checks on a sample, no videos
python experiments/analyze_hub.py results --out summary.json
```

## Hugging Face Space

[huggingface.co/spaces/rs1g6r/rdq](https://huggingface.co/spaces/rs1g6r/rdq) runs the licence and provenance check in the browser with Pyodide (`space_static/`, built by `python space_static/build.py`). `space/app.py` is a Gradio version that also runs the full quality checks and writes a passport, for hosts with compute.

## Licence

Apache-2.0 for the code. Hub figures describe public dataset cards at scan time.
