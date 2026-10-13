# How rdq checks a dataset's licence

rdq answers one question for each dataset: **do the terms written on this copy match the terms of the dataset it was made from?** It reads only the public dataset card (`README.md`) and the repository id, and compares them with a table of 65 public robot datasets whose licences were checked by hand at the source.

It does not decide the legal question. A tag on a copy can't grant rights its uploader doesn't hold, so rdq reports where the tag and the original disagree and links to the original's own statement, so anyone can check it. This is a diligence aid, not legal advice.

Everything below is implemented in [`rdq/licenses.py`](../rdq/licenses.py) and [`rdq/provenance.py`](../rdq/provenance.py), and covered by tests in [`tests/test_licenses.py`](../tests/test_licenses.py).

## 1. The reference table: what each original allows

[`rdq/data/upstream_licences.json`](../rdq/data/upstream_licences.json) has one entry per original dataset (DROID, BridgeData V2, LIBERO, the Open X-Embodiment components, …). Each entry records:

| Field | Meaning |
|---|---|
| `license` | The original's data licence as an SPDX id, or `not stated` when the release gives no data licence |
| `statement`, `statement_url` | The sentence the licence comes from, and where it was found |
| `checked` | The date the statement was checked |
| `confidence` | How direct the statement is (see the list after this table) |
| `commercial_training`, `redistribution` | Whether the terms allow these (true, false or unknown) |
| `id_pattern`, `link_pattern`, `mention_pattern` | How a copy is recognised (section 2) |

**Where licences come from.** Only primary sources count: the authors' paper, project page, GitHub README or LICENSE file, or their own Hub or Kaggle page. A code licence is never read as a data licence. If a project only licenses its code, the entry says `not stated`.

**Confidence levels:**

- **High:** an explicit data licence from the original authors. 26 of the 65 entries.
- **Medium:** an umbrella or indirect statement, such as the Open X-Embodiment README's CC BY 4.0 line for its components. 29 entries.
- **Low:** the scope or mapping is uncertain. 10 entries.

Confidence limits how serious a finding can be (section 4).

## 2. Matching: is this copy made from a known original?

rdq looks for four kinds of evidence, from strongest to weakest:

| Evidence | Rule | Example |
|---|---|---|
| `card-source` | The card's YAML `source_datasets` (or `license_link`) names the original | `source_datasets: [droid]` |
| `card-link` | The card links to the original's site, either in its own Homepage field or within 120 characters of a derivation word ("converted from", "subset of", "ported", "mirror of", "→", …) | `Homepage: https://droid-dataset.github.io` |
| `repo-id` | The repository name matches the original's name pattern | `someone/droid_lerobot` |
| `card-mention` | The card names the original, but not in a citation context | "teleoperated like BridgeData V2" does **not** count |

**Mentions that don't count.** A mention is ignored when the text around it (80 characters before, 40 after) says the original is only cited. Examples: a pretrained encoder or backbone, a benchmark or baseline, a comparison ("similar to", "unlike", "inspired by"), a data format, or a negation ("we do **not** use BridgeData V2").

**The original's own uploads.** A repository owned by the original's organisation (for example `robocoin/...`) is the publisher's own upload. Its tag is theirs to choose, so it is reported as information only.

**Corroboration.** A match is *corroborated* when the card itself declares the source (`card-source`), or when two independent signals agree, such as a derivation link plus a matching repository name.

## 3. Comparing the tag with the original

Each matched original is compared with the card's `license:` tag. If the tag is a list, the most restrictive entry counts: NC, then ND, then SA.

| Status | Rule |
|---|---|
| `consistent` | The tag is the original's licence |
| `relabelled` | The original is CC BY (no NC) and the tag is a permissive licence such as apache-2.0 or mit. The data is still usable, but attribution and a notice of changes are owed |
| `upstream-conflict` | The original doesn't allow commercial use and the tag does, the original forbids redistribution, or the tag is otherwise different |
| `stricter-than-upstream` | The tag keeps the original's CC terms and adds NC, ND or SA. Both sets of terms apply |
| `restricted-redistribution` | The tag matches, but the original doesn't allow sharing the data or adapted copies |
| `no-upstream-grant` | The original states no data licence, so the tag can't be traced to permission from the data's authors |
| `review` | The tag is `other`/`unknown`. Read the custom terms |
| `missing` | The card has no licence |
| `publisher-upload` | Uploaded by the original's own organisation |
| `no-known-upstream` | No known original matched. rdq says nothing about provenance |

**Several originals.** When a copy matches several originals, each one is judged separately and the worst result decides.

**Concluded licence.** The concluded licence (the terms rdq says actually apply) is drawn only from matches that say the data came from the original: `card-source`, `card-link` and `repo-id`. A mention alone never changes the concluded licence.

## 4. Severity: how sure rdq is

A finding is **critical** only when three things hold:

1. the original's terms are high-confidence;
2. the match is corroborated (section 2);
3. the conflict involves commercial use or redistribution.

Each of these caps the severity:

- **Medium-confidence original:** at most a warning.
- **Low-confidence original:** at most information.
- **Single uncorroborated signal** (a repository name alone, a link alone, a mention): at most a warning.
- **Publisher upload:** information.

## 5. The one-line answer

Every result ends in **OK to use**, **Check first** or **Stop**, plus one sentence:

| Answer | When |
|---|---|
| Stop | A conflict with a non-commercial original, no licence on the card, or a non-commercial tag |
| OK to use, with conditions | `relabelled` with a concluded licence. The sentence names who to credit and says changes must be noted |
| Check first | Only a mention links the copy to the original; the original states no data licence; redistribution is restricted; a custom or no-derivatives licence; or an apache-2.0 tag on LeRobot's unedited template card, which is most likely the upload tool's default |
| OK to use | `consistent`, `publisher-upload` or `stricter-than-upstream`, or a permissive licence with no known original |

**Default-licence rule.** A card counts as unedited when three things hold:

- it has LeRobot's template marker ("This dataset was created using [LeRobot]");
- it has no description;
- its Homepage and Paper fields still read "[More Information Needed]".

`apache-2.0` on such a card is flagged as most likely the tool's default rather than a choice by the owner.

**Card fix.** When exactly one original is concluded, rdq also suggests a card fix: the `license:` and `source_datasets:` lines that state the original's terms. rdq reads a fixed card back as `consistent`.

## 6. Validation

**Name-only matches, checked by hand.** Most matches rest on the repository name, so these were checked against the original release's episode count, frame count, robot type and card text:

- **17 most-downloaded name matches:** the robot type matches in all 17. 13 have the original's episode count, 3 are subsets and 1 is a 6-episode sample.
- **30 random name matches:** 26 are copies or subsets (same counts or published splits, same robot). The other 4 are simulation data that may have been newly generated in the same simulator, which the original's licence would not cover. None was a wrong match.

**Hub run.** On the 3,000 most-downloaded LeRobot datasets, the check ran without errors.

**Card-fix round trip.** For 192 of the 196 datasets that get a card fix, applying it makes rdq read the card as consistent. The other 4 have custom or non-commercial terms, so rdq still says to read them.

**Corrections.** A false match found after publication (a card that named BridgeData V2 only to say it wasn't used) led to the negation rule in section 2. The published figures were corrected from 305/201 to 304/200.

## 7. Known limits

- **A repository name is a lead, not proof.** Name-only matches are never critical, and the report says when a name is the only evidence.
- **A rights holder may relicence on purpose.** If the uploader owns the data, a different tag can be deliberate. rdq notes this on every `relabelled` and `upstream-conflict` finding.
- **Self-generated simulation data.** Data newly generated in a public simulator (LIBERO, RoboCasa, MimicGen) may match by name but isn't covered by the original dataset's licence.
- **Only known originals are covered.** A dataset made from an original that isn't in the table gets `no-known-upstream`, not a clean bill.
- **Licences change.** Every entry has a `checked` date. Corrections are welcome as issues or pull requests that link the primary source.

## 8. Reproducing a result

```bash
pip install git+https://github.com/s1g6r/rdq
rdq check <dataset-id-or-path> --json report.json   # see "provenance" in the JSON
```

The browser version at [huggingface.co/spaces/rs1g6r/rdq](https://huggingface.co/spaces/rs1g6r/rdq) runs the same code. Each result lists the matched original, the evidence kind, the confidence, the original's statement with its URL and check date, and the notes that explain the verdict.
