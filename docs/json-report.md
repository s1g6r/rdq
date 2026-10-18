# Reading rdq's JSON report

`rdq check owner/name --json report.json` (add `--video-full` for the frame-by-frame video check) writes one JSON object per dataset. This page is for people who build on it: dashboards, leaderboards, CI gates, data pipelines. Fields listed here keep their names and meaning within a `report_version`; new fields may be added at any time, so ignore keys you don't know.

From Python:

```python
from rdq.report import audit
result = audit("path/to/local/dataset", video_full=True, repo_id="owner/name")
```

For a Hub dataset, `rdq check` downloads the tables and card (and the videos with `--video` / `--video-full`) first; pin a commit with `--revision <sha>` so the result refers to an exact version.

## Top level

| Key | What it holds |
|---|---|
| `report_version` | `1`. Bumped only if a field below changes name or meaning. |
| `tool` | `{"name": "rdq", "version": "..."}` |
| `audited_at` | UTC time of the check (ISO 8601). |
| `dataset` | `path`, `revision` (when pinned), `codebase_version`, `robot_type`, `fps`, `episodes`, `frames`, `median_episode_frames`, `numeric_features`, `cameras`, `tasks`. |
| `score` | `dataset_score` (0-100), `usable_episodes` (list of episode ids with no critical or warning finding), `usable_fraction`, `flagged_episodes` (`{"<episode id>": ["check", ...]}`), `episode_scores`, `severity_counts`. |
| `findings` | List of findings (below), most severe first. |
| `provenance` | Licence and source result (below), or `null` when there is no dataset card. |
| `expected` | Present when an expected-findings file was used: `file`, `entries`, `findings_marked`. |
| `not_checked` | Plain-language list of what could not be checked (for example videos when they were not downloaded). Treat "not flagged" and "not checked" as different things. |
| `thresholds` | The data-check thresholds used. |

## A finding

```json
{
  "check": "video.black_span",
  "severity": "warning",
  "message": "1 episode has a black stretch in observation.images.front (...)",
  "episodes": [149],
  "details": {"camera": "observation.images.front", "...": "..."}
}
```

- `check` is a stable id. The full list with meanings is in the [README](../README.md#integrity-checks) and [raw-video.md](raw-video.md).
- `severity` is `critical`, `warning` or `info`. Critical and warning findings remove their episodes from `usable_episodes`; info never does.
- `episodes` is empty for dataset-level findings.
- `details` differs per check. Keys present in `details` of a given check stay stable within a `report_version`.
- A finding the owner marked as expected has `details.expected = true`, `details.expected_reason`, `details.original_severity` and `details.original_message`, and its severity is `info`.

## Video checks and their usual meaning

| Question | Checks |
|---|---|
| Blank or black footage | `video.black`, `video.black_span` |
| Lens covered | `video.covered`, `video.covered_span` |
| Blur | `video.blur`, `video.blur_span` (judged against the same camera's own footage) |
| Washed out | `video.overexposed`, `video.overexposed_span` |
| Frozen picture | `video.frozen`, `video.frozen_span` (info for simulated datasets, where a still scene renders identical frames) |
| Video and data lengths differ | `video.span_mismatch` (`details` says which is longer and by how much) |
| Footage that repeats another episode | `video.duplicate_episode`, `video.possible_duplicate`, `video.identical_file`, `video.same_stream_remuxed` |
| Camera streams swapped between episodes | `video.cameras_swapped` |
| Camera moved between sessions | `video.viewpoint_change` |
| File cannot be decoded | `video.decode_error`, `video.unreadable`, `video.truncated` |

## Licence

`provenance` holds the licence verdict:

- `license`: the tag on the card; `license_class`: `permissive`, `noncommercial`, `no-derivatives`, `missing` or `other`.
- `template_card` and `default_license_unconfirmed`: the card is LeRobot's unedited template, so the tag is probably the upload tool's default (apache-2.0).
- `upstream.status`: how the tag compares with the original dataset's terms (`consistent`, `relabelled`, `upstream-conflict`, `stricter-than-upstream`, `restricted-redistribution`, `no-upstream-grant`, `review`, `missing`, `publisher-upload` or `no-known-upstream`), with `upstream.concluded` (the terms that most likely apply), `upstream.upstream` (each matched source with its evidence and confidence) and `upstream.severity`.
- `bottom_line` and `bottom_level` (`go`, `check`, `stop`): one sentence and a traffic light for people who won't read the rest.
- `card_fix`: a suggested YAML change for the card, or `null`.

The rules behind these are in [licence-methodology.md](licence-methodology.md).

## Examples

```bash
# episodes to train on
jq -c .score.usable_episodes report.json
# one row per check: how many episodes it flags
jq -r '.findings[] | select(.severity != "info") | [.check, (.episodes | length)] | @tsv' report.json
# licence traffic light
jq -r '.provenance.bottom_level + "  " + .provenance.bottom_line' report.json
```
