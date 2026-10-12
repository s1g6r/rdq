# Changelog

## 0.2.0 (2026-10-12)

**Licence and provenance**
- Upstream licence knowledge base (`rdq/data/upstream_licences.json`): 65 public robot datasets with terms checked against primary sources, a confidence level, and repository-name patterns validated against the Hub.
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
