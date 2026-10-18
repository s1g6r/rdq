# Changelog

## Unreleased

- Fix: `video.duplicate_episode` now checks the robot data too. A copy from a merge or upload mistake duplicates the data with the footage; a precise robot replaying one motion films near-identical takes whose joint data still differs as much as any two recordings. Those are now leads (`video.possible_duplicate`, with `data_agreement` in the details). Found on lerobot/dlr_sara_grid_clamp while comparing rdq with LeBoard; checked by eye and in the data.
- Fix: `video.cameras_swapped` no longer fires on a stereo pair (two cameras that see nearly the same picture, such as a humanoid's head cameras): a swap between them can't be seen. Found on lerobot/unitreeh1_rearrange_objects while comparing rdq with LeBoard; checked by eye.

## 0.3.0 (2026-10-18)

- Docs: [docs/json-report.md](docs/json-report.md) documents the JSON report for tools that build on it; reports now carry `report_version: 1`.
- `video.span_mismatch` now says whether the video is shorter than the data (frames without an image; loaders can fail) or longer (extra footage, often a reset left in), by how much, per episode.
- Fix: `--video-full` crashed (KeyError) on a dataset whose camera was moved and which had an episode too short to measure (found on a Hub dataset).
- New: `video.cameras_swapped` (with `--video-full`): two cameras whose streams trade places in some episodes, as when camera devices change order between recording sessions. Found on a public dataset where the gripper and overhead feeds are swapped in the first 20 episodes.
- Fix: in simulated datasets (card tags such as robosuite, mujoco, isaac, robotwin, libero, or a sim robot type), `video.frozen_span` is info, not a warning: rendered frames have no sensor noise, so a still scene repeats exactly. Black renders and other problems still warn.
- New: a GitHub Action (`uses: s1g6r/rdq@main`) runs `rdq check` on a folder or Hub dataset, writes the report to the job summary and fails below a score; `rdq check --markdown FILE` writes the same Markdown summary anywhere.
- Fix: with a single camera, `video.duplicate_episode` now needs the movements to agree almost exactly (0.97) unless the files are identical; other takes of one grasp from a wrist camera scored 0.80-0.93 and are now leads (`video.possible_duplicate`). Found on two Open X-Embodiment conversions.
- Fix: `video.viewpoint_change` no longer fires on hand-held or wrist cameras: a camera with less than 40% of its picture still within episodes is not judged (found on a UMI gripper dataset; checked against a real moved camera, which is still found).
- New: episodes that are unusual on purpose can be listed in `meta/rdq_expected.yaml` (or passed with `rdq check --expected FILE`). Matching findings are reported as info with the owner's reason and no longer drop those episodes from the keep list. Requested by the owners of a Hub dataset whose deliberate no-object episodes rdq flagged.
- New: `rdq check --video-full` decodes every frame of every episode (each shared LeRobot v3 video file once) and reports frozen, black, covered and blurry stretches, timing gaps, decode problems and **duplicated episodes**: episodes whose footage repeats another episode's movements at the same moments, which takes of one task from a fixed camera do not.
- New: `rdq check-video` audits raw video deliveries (folders of mp4/mov/mkv, as bought from collection vendors or contributors): usable hours, black/covered/washed-out, frozen, idle and blurry stretches with their times, truncated and corrupt files, timing problems, spec violations, copies (re-encoded, cut, mirrored, remuxed, looped, or from an earlier batch via `--store`) and a per-contributor scorecard. JSON, HTML and CSV output; install with `pip install "rdq[video]"`. Method and validation: [docs/raw-video.md](docs/raw-video.md).
- New: when a copy's tag differs from a clearly identified original, rdq suggests the card fix (`license:` and `source_datasets:` lines to paste into the YAML header), in the CLI report, the HTML report, the JSON (`card_fix`) and the browser Space. rdq reads a fixed card back as consistent: this holds for 192 of the 196 Hub datasets that get a suggestion. The other 4 are custom or non-commercial terms, which still say to read the terms.
- Licence check: a card that names a source only to say it was *not* used ("we do not use BridgeData V2") no longer counts as evidence. One Hub dataset was wrongly matched this way; the scan figures are now 304 matched and 200 differing (were 305 and 201).
- When the only evidence is a mention on the card, the bottom line says "check first" and gives the original's licence, instead of "usable" with an empty licence.
- Odd `source_datasets` values on a card (a number, a mapping) no longer crash the check.
- Grammar: "restrict" for several restricted sources.

## 0.2.1 (2026-10-12)

Fixes from a pre-release review against real Hub dataset schemas.

**Loading and checks**
- Only numeric joint signals are loaded. Depth, depth video, point clouds, audio and arrays wider than 1,024 values are listed under `schema.features_not_checked` instead of crashing the load or causing false critical findings (about 4% of Hub datasets have such features).
- An empty `fps` is critical, and reports no longer crash when it is missing.
- New: `counts.episodes_not_in_metadata` / `counts.metadata_episodes_missing`, `values.empty_channel` (a never-finite channel is reported once instead of making every episode critical), `episodes.no_motion` (a dataset that never moves), and `episodes.spikes_systematic` (a channel that "glitches" in most episodes is reported once and no longer empties the usable-episode filter).
- Idle episodes are compared with episodes of the same task.
- Reports list what was not checked and why.
- `--max-frames` reads whole episodes up to the budget, even from a single large file, and reports it as an INFO note instead of a warning.

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

- From the first real-data validation run: angles that wrap at +-pi (Euler angles) are unwrapped before the glitch check, which removed most glitch flags on RoboCOIN dual-arm data; no-motion is judged per channel, so noisy velocity or effort channels in a wide state no longer mark a moving robot as frozen.

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
