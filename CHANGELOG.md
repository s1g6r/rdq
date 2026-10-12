# Changelog

## 0.2.1 (unreleased)

Fixes from a pre-release review against real Hub dataset schemas.

**Loading and checks**
- Only numeric joint signals are loaded. Depth, depth video, point clouds, audio and arrays wider than 1,024 values are listed under `schema.features_not_checked` instead of crashing the load or causing false critical findings (about 4% of Hub datasets have such features).
- An empty `fps` is critical, and reports no longer crash when it is missing.
- New: `counts.episodes_not_in_metadata` / `counts.metadata_episodes_missing`, `values.empty_channel` (a never-finite channel is reported once instead of making every episode critical), `episodes.no_motion` (a dataset that never moves), and `episodes.spikes_systematic` (a channel that "glitches" in most episodes is reported once and no longer empties the usable-episode filter).
- Idle episodes are compared with episodes of the same task.
- Reports list what was not checked and why.

**Scores**
- Without `observation.state`, actions are no longer compared with themselves; DemInf-lite and consistency need another observation to condition on.
- Never-finite channels, episodes with an unusual action/state width, and datasets with fewer than 20 scoreable frames no longer crash `rank` or `scorecard`.
- KSG counts marginal neighbours strictly, as in Kraskov et al.'s algorithm 1.
- `rdq rank` shows check failures instead of a "minority strategy" note for broken episodes, writes strict JSON, and labels `--keep-top` experimental.

**Licences and passport**
- The most restrictive upstream term wins across multiple sources, and each source's severity is capped by its own confidence and evidence.
- Evidence kinds: `card-source`, `card-link` (only in a derivation context), `repo-id`, `card-mention`, `publisher`. Generic repo-name matches are capped at warning; publisher uploads are informational.
- New status `stricter-than-upstream`; the passport keeps restrictions a publisher added (the concluded licence is the conjunction).
- Fixed patterns that matched unrelated ids (`cambridge_v2` as BridgeData V2, Unitree Z1 as G1).
- Passport: a Croissant record set over the frame index columns, no local paths, valid SPDX LicenseRef ids, more HF licence tags mapped to SPDX.

**After a second review**
- A never-finite channel is dropped from action and state together, so the sync check keeps running.
- `episodes.no_motion` is scale-free and looks at the state (and absolute-position actions), so a frozen stream with sensor noise is caught; a whole task that barely moves compared with the other tasks is flagged as idle.
- `episodes.spikes_systematic` needs 8 or more episodes and is a dataset-level warning.
- `values.empty_feature`: an action or state with no finite values at all is critical.
- Licence findings are critical only when the card declares the source (`source_datasets`, `license_link`) or two independent signals agree; a single signal is at most a warning. Mentions in citation context (pretrained encoders, benchmarks, formats) are ignored, and only derivation evidence sets the concluded terms. New statuses `publisher-upload` and verdict "no upstream data licence".

- The sync check also runs when action and state have different widths, on the channels whose declared names match (at least three). About 10% of Hub datasets have different widths with matching names.

**Evidence**
- Hub summary separates publisher uploads from traced conversions, reports match evidence, and no longer lists dataset ids.
- Benchmark confidence intervals are computed over seeds.

## 0.2.0 (2026-10-12)

**Licence and provenance**
- Upstream licence knowledge base (`rdq/data/upstream_licences.json`): 65 public robot datasets with terms checked against primary sources, and a confidence level.
- Declared-vs-upstream assessment: `consistent`, `relabelled`, `upstream-conflict`, `restricted-redistribution`, `no-upstream-grant`, `review`, `missing`. Each match records its evidence: card link, repo name or card mention.
- `rdq passport`: Croissant 1.1 + RAI + PROV-O JSON-LD (validates with `mlcroissant`), an SPDX 3.0 export and a Markdown datasheet. Owner-only facts come from an answers file.

**Integrity checks**
- Action/state lag is now estimated from velocities, so desync is detectable in steady motion. The check runs on episodes longer than `lag_max + 10` frames.
- The glitch threshold is robust when glitches exceed 1% of frames.
- Optional camera checks with `--video`: frozen, black, blurry and undecodable footage.
- `rdq check owner/name` fetches a Hub dataset's tables and card.
- `rdq check-mcap` audits MCAP recordings (ROS 2 JointState, Float*MultiArray or JSON), one file per episode, with the signal-level checks, so vendor deliveries can be checked before conversion.

**Episode scores**
- `rdq rank`: DemInf-lite, kNN consistency with minority-strategy detection, smoothness (HFR, LDLJ), the published spectral metric as a baseline, hesitation, gripper cycles, final-state outliers, near-duplicates and openpi-style idle trimming. The composite is the mean rank of DemInf-lite and consistency, with a length-confound warning.
- `rdq scorecard`: quality per collector, operator, session or task.

**Evidence and tooling**
- `experiments/curation_benchmark.py` covers structural faults vs operator skill against oracle, random, length, spectral, loss-based and rule-based selection. `experiments/robomimic_mh.py` is a natural-noise benchmark.
- CI with a false-alarm test, a manual benchmarks workflow, a Hugging Face Space app, and a synthetic sample vendor delivery.

## 0.1.0 (2026-10-11)
- First release: rule-based integrity checks, licence verdict for the default-template card, hub scan and audit, HTML/JSON reports, toy experiment.
