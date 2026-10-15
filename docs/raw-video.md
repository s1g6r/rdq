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
  correlate (0.75 or more): a copy repeats the same movements at the same moments; another take
  does not. Copies are grouped, and each copy is reported once against one original (an earlier
  batch if there is one, else the longest file).
- **Usable time** is footage minus the union of all unusable spans; copies count as zero.

## Validation

**Synthetic deliveries with known answers** (`experiments/video_benchmark.py`): five contributors
with the failure patterns of paid collection (clean, idle padding, a bad lens, a flaky device with
stalls, black frames, gaps and a truncated file, and a re-uploader of other people's clips:
re-encoded, cut, mirrored and brightened, or converted to 25 fps).

BENCHMARK_TABLE

**Real robot footage** (`experiments/video_real.py`, run in CI by `.github/workflows/video-real.yml`):
60 episodes from ten public LeRobot datasets (ALOHA static and mobile, ALOHA sim, DROID, SO-101,
Berkeley UR5 wrist camera, Push-T, UT Austin MUTEX, TACO Play, NYU Franka), many of them several takes
of one task from one fixed camera, plus two planted copies of one ALOHA episode (a half-size
re-encode and a mirrored, brightened 60% cut) filed under another contributor.

REAL_TABLE

Tuning history, kept honest: the first run on real footage flagged takes of the same task as copies
(their frames match) and soft, low-texture wrist-camera footage as blurry, and missed the planted
copies because a fixed threshold called them idle. The motion-agreement check, camera-relative blur
and the one-second change measure were added in response; the numbers above are after those fixes,
on the same files, so they are a check of the fixes, not an independent test. The next independent
test is a real delivery.

## Limits

- A contributor whose camera is blurry all the time is compared with its own norm, so that blur is
  not removed from usable time. Set `min_width`/`min_height` in the spec and review a sample per
  contributor.
- Copies shorter than 5 s, or with too little movement to compare, are not reported.
- Idle means nothing in view moves; a person standing still in front of a moving camera is not idle,
  and a slowly drifting static shot may not be either.
- Only the first video stream of a file is checked; audio is ignored.
- Thresholds were set on the footage above. They are constants at the top of `rdq/rawvideo.py`.
