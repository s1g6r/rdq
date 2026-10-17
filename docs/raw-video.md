# `rdq check-video`: raw video deliveries

Teams that buy or collect robot and egocentric footage pay per hour, and only some of those hours are
usable. `rdq check-video` answers the questions a buyer asks of a delivery before paying for it or
training on it:

- **How many usable hours are in it?** Footage minus black, covered, washed-out, frozen, idle and
  blurry stretches, minus recording gaps, copies and unreadable files.
- **What is wrong, where?** Every finding has the times it covers, so a reviewer can jump to it.
- **Is any of it a copy?** Re-uploads, re-encodes, mirrored or cut copies, the same stream in a
  different container, footage repeated within a file, and copies of earlier deliveries.
- **Who delivered what?** A per-contributor scorecard (usable share, defects, copies of other
  people's footage), so payment and feedback can follow the footage.

```bash
pip install "rdq[video]"            # PyAV + OpenCV; ffprobe is used when installed
rdq check-video deliveries/2026-10/ --by-folder --html report.html --csv files.csv
rdq check-video batch.zip-extracted/ --contributors who.csv --spec spec.yaml --store hashes.json
```

`--by-folder` takes the first sub-folder as the contributor; `--contributors` reads a CSV with a `file`
column and a `contributor` (or collector, operator, worker, partner, device) column. `--store` keeps
compact fingerprints of every checked file, so the next delivery is also compared with this one. The
exit code is 1 when any file is critical (unreadable, truncated, corrupt, or an exact copy).

## What it checks

| Check | Severity | What it means |
|---|---|---|
| `video.unreadable` | critical | the file cannot be opened or decoded |
| `video.truncated` | critical | it decodes to less than the header says (cut off, e.g. battery or upload failure) |
| `video.corrupt_frames` | critical above 1% | frames the decoder rejects |
| `video.identical_file`, `video.same_stream_remuxed`, `video.near_duplicate` | critical | a copy of another file in the batch or an earlier batch |
| `video.trimmed_copy`, `video.partial_overlap`, `video.loop_within_file` | warning | part of the footage is someone else's or repeats |
| `video.black`, `video.covered`, `video.overexposed` | warning | lens cap, hand over the lens, washed out |
| `video.frozen` | warning | the same frame repeated: a stalled camera |
| `video.idle` | warning above the spec's share | nothing moving at all for 5 s or more |
| `video.blur` | warning above the spec's share | blurry for 3 s or more without camera motion |
| `video.recording_gap`, `video.dropped_frames`, `video.variable_frame_rate`, `video.bad_timestamps` | warning | timing problems that break loaders |
| `video.spec_*` | warning | resolution, fps, codec or duration outside the delivery spec |
| `video.low_bitrate`, `video.unusual_fps`, `video.unusual_resolution`, `video.low_light`, `video.rotation`, `video.hdr`, `video.interlaced` | info | hints (for example, a re-compressed download among camera originals) |

The delivery spec is a small YAML file:

```yaml
fps: 30
fps_tolerance: 0.5
min_width: 1280
min_height: 720
codecs: [h264, hevc]
min_duration_s: 60
max_idle_share: 0.15    # idle above this share of a file is a warning
max_blur_share: 0.10
```

## How it works

Each file is read once. `ffprobe` (or PyAV) gives the container metadata; the packet timestamps give
gaps, variable frame rate and a hash of the coded stream; one decode pass at 320 px wide (never
upscaled) gives per-frame brightness, detail, sharpness and repeat flags, and, twice a second, a
perceptual hash, a mirrored-frame hash, a 32x18 thumbnail and optical flow. Files run in parallel.

- **Black / covered / washed out** come from per-frame brightness and detail.
- **Frozen**: runs of repeated frames. A stalled camera's frames come back bit-identical much of the
  time (the encoder only refines a few pixels); a still scene recorded live is near-identical after
  compression but almost never bit-identical, so it is reported as idle instead.
- **Idle**: how much the picture changes over one second, measured on the thumbnails (which average
  away sensor noise and compression artefacts), stays below 2 grey levels for at least 5 s.
- **Blur** is judged against the same camera's typical footage (same contributor, same resolution),
  never a fixed number: a white wall, a smooth towel or a soft low-resolution camera has little fine
  detail and is still in focus. A stretch is blurry when both the Laplacian energy and the
  texture-independent ratio of fine detail to edges fall well below that camera's norm, and the
  camera is not moving (motion blur is counted separately and not removed).
- **Copies**: candidate pairs come from a band index over the perceptual hashes; a pair is aligned by
  finding the longest run of matching frames at one time offset (mirrored copies through the
  mirrored hashes). Takes of the same task from a fixed camera share every background pixel, so
  frames alone are not evidence. The decisive test re-decodes the matched stretch of both files,
  finds the sub-second shift that lines them up, and requires the frame-to-frame changes to
  correlate (0.80 or more): a copy repeats the same movements at the same moments; another take
  does not. Copies are grouped, and each copy is reported once against one original (an earlier
  batch if there is one, else the longest file).
- **Usable time** is footage minus the union of all unusable spans; copies count as zero.

## Validation

**Synthetic deliveries with known answers** (`experiments/video_benchmark.py`): five contributors
with the failure patterns of paid collection (clean, idle padding, a bad lens, a flaky device with
stalls, black frames, gaps and a truncated file, and a re-uploader of other people's clips:
re-encoded, cut, mirrored and brightened, or converted to 25 fps).

| Seed | Files | Planted defects found (time within 1.5 s) | Files falsely flagged | Copies found | False copy pairs | Usable-hours error | Contributor ranking (Spearman) |
|---|---|---|---|---|---|---|---|
| 0 (20 s clips) | 41 | 25 of 25 | 0 | 8 of 8 | 0 | +3.1% | 1.0 |
| 1 (30 s clips) | 41 | 24 of 24 | 0 | 8 of 8 | 0 | +2.1% | 1.0 |
| 2 (25 s clips) | 51 | 33 of 33 | 0 | 10 of 10 | 0 | +2.4% | 1.0 |

Copies include every mode (re-encoded at lower resolution, cut, mirrored and brightened, 25 fps), and
the truncated file was caught in every seed. The usable-hours estimate is slightly optimistic because
span edges are found up to about a second late. Analysis took about 2.4 s per minute of video on 2
cores. Results: `experiments/results/video_benchmark_seed*.json`.

**Many takes of one scene** (`experiments/video_scale.py`): 80 takes of one task from one fixed
camera, so every frame hash matches every other take and the hash index saturates, plus three copies
(half-size re-encode, mirrored cut starting off the sampling grid, 25 fps). With moving objects of
moderate size, all 3 copies were found and nothing else; with objects so small they almost vanish
at thumbnail scale, 2 of 3 (the mirrored cut was missed) and nothing else.

**Real robot footage** (`experiments/video_real.py`, run in CI by `.github/workflows/video-real.yml`):
60 episodes from ten public LeRobot datasets (ALOHA static and mobile, ALOHA sim, DROID, SO-101,
Berkeley UR5 wrist camera, Push-T, UT Austin MUTEX, TACO Play, NYU Franka), many of them several takes
of one task from one fixed camera, plus two planted copies of one ALOHA episode (a half-size
re-encode and a mirrored, brightened 60% cut) filed under another contributor.

| Measure | Result |
|---|---|
| Genuine episodes with any finding (false alarms) | 0 of 60 |
| Planted copies found, with the right original and kind | 2 of 2 (near-duplicate; trimmed copy) |
| Look-alike candidate pairs that are not copies of each other (other takes of the same task, or a copy against a different take), rejected | 34 of 34 |
| Motion agreement: copies vs other takes | 0.90-0.94 vs at most 0.54 (threshold then 0.75, now 0.80) |
| Analysis speed | about 3.1 s per minute of video on 2 CI cores |

The per-file and per-pair measurements behind these numbers are in
`experiments/results/video_real_ci_annotations.txt`. "No false alarms" is the claim; these curated public
episodes may simply contain few real defects, so this run says little about recall on real footage.

**Held-out real footage**: 72 episodes from twelve other public LeRobot datasets that were never used
for tuning (ALOHA cups and Mobile ALOHA wipe-wine, Unitree H1, xArm, Stanford HYDRA, UT Austin BUDS,
TOTO, FMB, Berkeley cable routing, SO-100 stacking, IAMLab CMU, CMU Play Fusion), plus one planted
half-size re-encode of an FMB episode.

| Measure | Result |
|---|---|
| Genuine episodes with any finding (false alarms) | 0 of 72 |
| Planted copy found | 1 of 1 (motion agreement 0.92) |
| Look-alike candidate pairs rejected | 53 of 53, highest motion agreement 0.70 |

One change was made after this run: the highest score between two different takes (0.70, two ALOHA
episodes of the same cup task) was closer to the copy threshold than in the tuning set, so the threshold
was raised from 0.75 to 0.80. Every copy seen so far scores 0.86 or more. Measurements:
`experiments/results/video_real_heldout_ci_annotations.txt`.

Tuning history, kept honest: the first run on real footage flagged takes of the same task as copies
(their frames match) and soft, low-texture wrist-camera footage as blurry, and missed the planted
copies because a fixed threshold called them idle. The motion-agreement check, camera-relative blur
and the one-second change measure were added in response; the numbers above are after those fixes,
on the same files, so they are a check of the fixes, not an independent test. The next independent
test is a real delivery.

## Inside LeRobot datasets: `rdq check --video-full`

The same machinery runs on LeRobot datasets. Each video file is decoded once; each episode's stretch
(from `meta/episodes`) is checked as its own clip, and the findings go into the normal `rdq check`
report with episode numbers, so they also drop episodes from `--filter keep.json`:
`video.frozen_span`, `video.black_span`, `video.covered_span`, `video.overexposed_span`,
`video.blur_span`, `video.idle_span` (info), `video.recording_gap`, `video.decode_error`,
`video.duplicate_episode` (an episode whose footage repeats another's), `video.viewpoint_change` and
`video.cameras_swapped`. Every camera is checked.

Two rules specific to datasets:

- **A duplicated episode must match in every camera** (at least two, when the dataset has two or more) and over
  at least half the episode. With a single camera there is no second view, so the movements must agree almost
  exactly (0.97), unless the files are identical. Cameras record together, so a real copy shows in all of them; a match in one
  camera, or over a small part of an episode, is reported as `video.possible_duplicate` (info, not counted).
- **`video.cameras_swapped`**: two cameras whose streams trade places in some episodes (camera devices that
  changed order between sessions, a common LeRobot recording problem): in those episodes each camera's view looks
  like the other camera's view elsewhere, and neither looks like its own. A swap must cover at least three
  episodes in at most three runs of consecutive episodes (sessions); single or scattered matches come from a
  wrist camera that happens to point at the other camera's view. On 12 public multi-camera datasets it finds the
  one real swap (EPITECH-LILLE/eval_training, episodes 0-19, confirmed by eye) and nothing else.
- **Simulated footage**: when the card's tags (robosuite, mujoco, isaac, robotwin, libero, ...) or the robot type say
  the data is rendered, frozen stretches are info: a renderer has no sensor noise, so a still scene repeats frames
  exactly, which in real footage only a stalled camera does.
- **`video.viewpoint_change`**: when the camera was moved between sessions, episodes fall into contiguous
  groups with different backgrounds (the median of each episode's thumbnails). rdq reports the groups and
  judges sharpness within each viewpoint, so a new framing is not mistaken for blur. A hand-held or wrist
  camera, where most of the picture changes within an episode (less than 40% of it still), is not judged:
  its "background" is just wherever the operator pointed it.

**On the Hub (October 2026).** `--video-full` on 20 datasets (2 from the lerobot org and the 18
most-downloaded community datasets with videos under 600 MB, about 4,600 episodes). What held up after
looking at the frames:

| Dataset | Finding | How it was checked |
|---|---|---|
| IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot | the second camera repeats frames for 1.3-1.8 s at the start of 6 episodes while the main camera moves | frame differences recomputed in the browser for 4 of the 6; a control episode is clean |
| EPITECH-LILLE/eval_training | the two camera feeds are swapped in episodes 0-19 (gripper view stored as `front`, overhead view as `back`), and the overhead camera was moved after episode 39 | frames of both cameras seen side by side (episodes 5, 19, 20, 30, 39, 40, 50, 59) |
| cloudwalk-research/gr00t-g1-grab-bottle-right-hand-v11 | episodes 342, 350, 351 nearly repeat 343 and 349: 5 s clips of an empty scene with no task in them. **Not a defect:** the owners confirmed these are deliberate no-bottle examples (the robot should not grasp) | frames seen side by side; owners' reply on the dataset's discussion #1 |

Three false alarms surfaced on the way and were fixed before these results: duplicates matched in one camera
only (now leads), duplicates accepted on matching foregrounds in slow play data (CMU Play Fusion; now the
movement must match too), and a moved camera reported as blur (now a viewpoint change). The other datasets
had no warnings. Results: `experiments/video_full_hub.py` and the `video-full-hub` workflow.

A fourth false alarm came from the next batch: a hand-held UMI gripper camera (brandonyang/umi-fold-shirt) was
reported as moved between sessions because the operator held it differently. The viewpoint test now skips cameras whose
picture mostly changes within episodes (17% still for that camera, 1% for a Unitree wrist camera, 82% for the fixed
EPITECH camera that really was moved). A fifth, in the
same batch: two single-camera conversions (Berkeley MVP, NYU door opening; one wrist camera at 3-5 fps) where
other takes of the same grasp agreed at 0.80-0.85. With one camera a duplicate now needs 0.97: a third single-camera conversion (Berkeley RPT, 30 fps) had takes of the
same grasp at 0.90-0.93, while a copy from a merge mistake is near 1.0.

## Limits

- A contributor whose camera is blurry all the time is compared with its own norm, so that blur is
  not removed from usable time. Set `min_width`/`min_height` in the spec and review a sample per
  contributor.
- Copies shorter than 5 s, or with too little movement to compare, are not reported.
- Idle means nothing in view moves; a person standing still in front of a moving camera is not idle,
  and a slowly drifting static shot may not be either.
- Only the first video stream of a file is checked; audio is ignored.
- Thresholds were set on the footage above. They are constants at the top of `rdq/rawvideo.py`.
