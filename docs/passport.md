# rdq data passport (v0.1)

A data passport is a machine-readable record of where a robot dataset came from, what licence actually applies to it, and what state it is in. rdq writes it as a [Croissant 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html) JSON-LD document. The passport uses the [Croissant RAI](https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html) vocabulary and [W3C PROV-O](https://www.w3.org/TR/prov-o/), and adds the `rdq:` terms described on this page. The same record can also be written out as an SPDX 3.0 dataset package and as a Markdown datasheet.

```bash
rdq passport --init-answers answers.yaml        # questions only the data owner can answer
rdq passport path/to/dataset --repo-id owner/name --answers answers.yaml \
    --out passport.json --spdx passport.spdx.json --datasheet DATASHEET.md
```

The passport validates with `mlcroissant` (the MLCommons reference library). Its record set covers the frame tables' index columns (episode, frame, timestamp, task), so Croissant loaders can enumerate frames; action and state vectors are described in `rdq:structure` but are not yet Croissant fields.

## Where each field comes from

rdq never makes up facts about how data was collected. Every field comes from one of four sources:

| Source | Meaning | Examples |
|---|---|---|
| **data** | Computed from the dataset files | structure, counts, hours, content hash, quality findings |
| **card** | Parsed from the Hugging Face dataset card | declared licence, `license_link`, `source_datasets` |
| **rdq** | Derived by rdq | licence status, upstream match, concluded licence, restrictions |
| **owner** | Given by the owner in `answers.yaml` | collection method, operators, timeframe, personal data, intended use |

`rdq:fieldSources` lists which source each field came from. `rdq:unanswered` lists the owner questions that were left blank.

## `rdq:` terms

Namespace: `https://github.com/s1g6r/rdq/blob/main/docs/passport.md#`

### `rdq:license`

| Key | Meaning |
|---|---|
| `declared`, `declaredSpdx` | The licence tag on the card, as written and as an SPDX id |
| `status` | One of the statuses below |
| `concluded` | The licence rdq concludes applies: the upstream terms if the dataset is a known mirror or conversion, the declared tag if it was confirmed or explicitly chosen, and otherwise `NOASSERTION` |
| `commercialTraining`, `redistribution` | `true`, `false` or `null` (unknown), according to the upstream terms |
| `restrictions` | Obligations carried from upstream: `attribution`, `indicate-modifications`, `non-commercial`, `share-alike`, `no-derivatives`, `no-redistribution`, `signed-agreement` |
| `evidence` | The upstream statement, its URL, the date it was checked, and how the match was made (`card-link`, `repo-id` or `card-mention`) |
| `templateCard` | The card is LeRobot's unedited template |
| `confirmedBy` | The person who confirmed the licence, from the owner's answers |

Licence statuses:

- `explicit`: a licence was chosen and there is no known upstream to compare it against.
- `default-unconfirmed`: `apache-2.0` on an unedited LeRobot template card. This is most likely the upload tool's default (`push_to_hub(license="apache-2.0")`), not a choice the owner made.
- `owner-confirmed`: the owner confirmed the licence in the answers file.
- `consistent`: the declared tag matches the known upstream licence.
- `relabelled`: the upstream is CC BY, but the dataset declares a different permissive tag. Attribution and a notice of changes are still owed.
- `upstream-conflict`: the declared tag conflicts with the upstream terms, for example a permissive tag on non-commercial data.
- `restricted-redistribution`: the tag matches, but the upstream terms do not allow sharing adapted copies (no-derivatives or no-redistribution).
- `review`: a custom licence (`other`) on data with a known upstream. Check that the custom terms carry the upstream restrictions.
- `stricter-than-upstream`: the declared tag adds a restriction the upstream lacks (for example non-commercial on CC BY data). Both sets of terms apply, and the concluded licence is their conjunction.
- `no-upstream-grant`: the original release states no data licence (only a code licence), so the declared tag cannot be traced to a grant from the data's authors.
- `missing`: no licence declared.
- `unknown`: no licence assessment was available.

### Other terms

- `rdq:contentHash`: SHA-256 over the path and digest of every file in `meta/` and `data/`. Pass `--hash-videos` to include `videos/`. This ties the passport to one exact version of the files.
- `rdq:pinnedRevision`: the Hub commit or version the passport describes (`--revision`).
- `rdq:structure`: codebase version, robot type, fps, episodes, frames, hours, tasks, numeric features and cameras.
- `rdq:quality`: the rdq score, a summary of findings per check, and the thresholds used.
- `rdq:attestation`: who generated the passport and when. It also says plainly that owner answers were not verified by rdq.

## Upstream licence knowledge base

`rdq/licenses.py` records the terms of public robot datasets. A source is added only after its licence has been checked against a primary source, and each entry keeps the statement, URL and check date. It covers 65 sources (see [upstream_licences_sources.md](upstream_licences_sources.md)); 26 have an explicit data licence from the original authors (high confidence), and the others rest on an umbrella statement or an uncertain mapping, which caps how severe a finding can be. Matches have five kinds of evidence, strongest first:

- `card-source`: the card's `source_datasets` names the upstream.
- `card-link`: the card links to the upstream project in a derivation context, such as a "Homepage" field or "converted from". A link that only cites the project counts as a mention.
- `repo-id`: the repo name matches, for example `droid_1.0.1`. This needs confirming. A generic name match is never critical unless the name also marks a conversion (`lerobot`, `v30`, `subset`, ...).
- `card-mention`: the card names or cites the upstream, possibly only for comparison. This never makes a finding critical.
- `publisher`: the repo belongs to the upstream's own organisation, so its tag is the publisher's choice. This is informational only.

Each matched source is judged on its own evidence and confidence, and the most severe result is reported; when sources disagree, the most restrictive term (non-commercial, no redistribution) wins.

Data recorded *on* a robot is not the same as data from that robot maker's dataset. For example, `RoboCOIN/AgiBot-g1_*` is not AgiBot World, and rdq does not match it as such.

## Mapping to other frameworks

| Passport field | Croissant / RAI | SPDX 3.0 | EU AI Act |
|---|---|---|---|
| `rdq:structure`, `distribution` | `distribution`, `recordSet` | `dataset_datasetType`, `dataset_datasetSize` | Annex IV §2(d) main characteristics |
| `rdq:license.declared` | (none) | `hasDeclaredLicense` | GPAI training-content summary, licence information |
| `rdq:license.concluded` | `license` | `hasConcludedLicense` | Art. 53(1)(c), copyright policy evidence |
| `prov:wasDerivedFrom` | PROV-O | `descendantOf` | Art. 10(2)(b), data origin |
| `rai:dataCollection` and related fields | RAI lifecycle | `dataset_dataCollectionProcess` | Art. 10(2)(b), how data was obtained |
| `rai:dataPreprocessingProtocol` | RAI lifecycle | `dataset_dataPreprocessing` | Annex IV, preparation and cleaning |
| `rdq:quality` | `rai:dataLimitations` | `dataset_datasetNoise` | Art. 10(3), "free of errors and complete" |
| `rdq:contentHash`, `rdq:pinnedRevision` | `sdVersion` | `verifiedUsing` | Art. 18, keeping documentation reproducible |

The EU AI Act column shows where a field would be useful as evidence. A passport is not a conformity assessment, and nothing here is legal advice.
