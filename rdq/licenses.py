"""Upstream licence knowledge base and declared-vs-upstream conflict detection.

A licence tag on a Hugging Face mirror is not a grant: an uploader cannot relicense
data they do not own. Many LeRobot conversions of well-known datasets carry the
upload tool's default `apache-2.0` while the original release says something else
(CC BY 4.0 with attribution and change-notice duties, or a non-commercial licence).

This module records the upstream terms of public robot datasets whose licence we
have checked against a primary source, recognises mirrors and conversions of them
from the repo id or the card text, and compares the declared tag with the upstream
terms. Entries are deliberately few: an entry is only added once its terms are
verified, with the statement URL and the date it was checked.

This is a heuristic aid for diligence, not legal advice.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

# Restriction vocabulary used across rdq outputs.
ATTRIBUTION = "attribution"
CHANGE_NOTICE = "indicate-modifications"
NONCOMMERCIAL = "non-commercial"
SHAREALIKE = "share-alike"
NODERIVS = "no-derivatives"
NO_REDISTRIBUTION = "no-redistribution"
AGREEMENT = "signed-agreement"


@dataclass(frozen=True)
class UpstreamSource:
    key: str
    name: str
    license: str  # SPDX id where one exists, else "LicenseRef-<name>"
    commercial_training: bool | None  # None = unclear
    redistribution: bool | None
    restrictions: tuple[str, ...]
    statement_url: str
    statement: str  # short quote or paraphrase of the upstream statement
    checked: str  # ISO date the statement was checked
    id_pattern: str  # regex on the lower-cased repo id
    link_pattern: str | None = None  # regex for the upstream's URL in the card (strong evidence)
    mention_pattern: str | None = None  # regex for the upstream's name in the card (weak evidence)


# Checked 2026-10-12 against the primary sources listed. See research notes R4.
UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = (
    UpstreamSource(
        key="droid",
        name="DROID",
        license="CC-BY-4.0",
        commercial_training=True,
        redistribution=True,
        restrictions=(ATTRIBUTION, CHANGE_NOTICE),
        statement_url="https://arxiv.org/abs/2403.12945",
        statement="DROID is released under CC BY 4.0 (stated in the DROID paper).",
        checked="2026-10-12",
        id_pattern=r"(^|[/_\-.])droid([_\-.]|\d|$)",
        link_pattern=r"droid-dataset\.github\.io",
    ),
    UpstreamSource(
        key="bridge_v2",
        name="BridgeData V2",
        license="CC-BY-4.0",
        commercial_training=True,
        redistribution=True,
        restrictions=(ATTRIBUTION, CHANGE_NOTICE),
        statement_url="https://rail-berkeley.github.io/bridgedata/",
        statement="All data is provided under the Creative Commons Attribution 4.0 International License.",
        checked="2026-10-12",
        id_pattern=r"bridge[_\-]?(data)?[_\-]?(v2|orig)|bridgedata|bridgev2",
        link_pattern=r"rail-berkeley\.github\.io/bridgedata",
        mention_pattern=r"bridgedata v2|bridgedata",
    ),
    UpstreamSource(
        key="libero",
        name="LIBERO",
        license="CC-BY-4.0",
        commercial_training=True,
        redistribution=True,
        restrictions=(ATTRIBUTION, CHANGE_NOTICE),
        statement_url="https://github.com/Lifelong-Robot-Learning/LIBERO",
        statement="LIBERO code is MIT; the released datasets are CC BY 4.0.",
        checked="2026-10-12",
        id_pattern=r"(^|[/_\-])libero[_\-]?(10|90|goal|object|spatial|long)?([_\-]|$)",
        link_pattern=r"lifelong-robot-learning/libero|libero-project\.github\.io",
    ),
    UpstreamSource(
        key="agibot_world",
        name="AgiBot World",
        license="CC-BY-NC-SA-4.0",
        commercial_training=False,
        redistribution=True,
        restrictions=(ATTRIBUTION, NONCOMMERCIAL, SHAREALIKE, AGREEMENT),
        statement_url="https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha",
        statement="AgiBot World Community License Agreement, CC BY-NC-SA 4.0, gated access.",
        checked="2026-10-12",
        # Data recorded *on* an AgiBot robot (e.g. RoboCOIN/AgiBot-g1_*) is not AgiBot World.
        id_pattern=r"agibot[_\-]?world|agibot[_\-](alpha|beta)",
        link_pattern=r"agibot-world/agibotworld|agibot-world\.com",
        mention_pattern=r"agibotworld|agibot world|agibot-world",
    ),
    UpstreamSource(
        key="egodex",
        name="EgoDex",
        license="CC-BY-NC-ND-4.0",
        commercial_training=False,
        redistribution=False,  # ND: no adapted copies may be shared
        restrictions=(ATTRIBUTION, NONCOMMERCIAL, NODERIVS),
        statement_url="https://github.com/apple/ml-egodex",
        statement="EgoDex is available under CC-by-NC-ND terms.",
        checked="2026-10-12",
        id_pattern=r"egodex",
        link_pattern=r"apple/ml-egodex",
        mention_pattern=r"egodex",
    ),
    UpstreamSource(
        key="epic_kitchens",
        name="EPIC-KITCHENS",
        license="CC-BY-NC-4.0",
        commercial_training=False,
        redistribution=True,
        restrictions=(ATTRIBUTION, NONCOMMERCIAL),
        statement_url="https://epic-kitchens.github.io/2025",
        statement="EPIC-KITCHENS is CC BY-NC 4.0; commercial licences on request.",
        checked="2026-10-12",
        id_pattern=r"epic[_\-]?kitchens?",
        link_pattern=r"epic-kitchens\.github\.io",
        mention_pattern=r"epic[- ]kitchens",
    ),
    UpstreamSource(
        key="hoi4d",
        name="HOI4D",
        license="CC-BY-NC-4.0",
        commercial_training=False,
        redistribution=True,
        restrictions=(ATTRIBUTION, NONCOMMERCIAL),
        statement_url="https://hoi4d.github.io/",
        statement="HOI4D is released under CC BY-NC 4.0.",
        checked="2026-10-12",
        id_pattern=r"hoi4d",
        link_pattern=r"hoi4d\.github\.io",
        mention_pattern=r"hoi4d",
    ),
    UpstreamSource(
        key="ego4d",
        name="Ego4D",
        license="LicenseRef-Ego4D",
        commercial_training=True,
        redistribution=False,
        restrictions=(AGREEMENT, NO_REDISTRIBUTION),
        statement_url="https://ego4d-data.org/pdfs/Ego4D-Licenses-Draft.pdf",
        statement="Use requires a signed agreement; licensees may not give third parties access to the data.",
        checked="2026-10-12",
        id_pattern=r"ego4d",
        link_pattern=r"ego4d-data\.org",
        mention_pattern=r"ego4d",
    ),
)

_BY_KEY = {s.key: s for s in UPSTREAM_SOURCES}

# Declared tags that grant broad permissive rights. A CC BY upstream relabelled as one of
# these loses its attribution and change-notice duties on paper (not in law).
_PERMISSIVE_TAGS = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "pddl", "odc-by", "unlicense"}


@dataclass
class SourceMatch:
    key: str
    name: str
    evidence: str  # "card-link" (card links the upstream), "repo-id" (name), or "card-mention" (named in text)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LicenseAssessment:
    """Declared licence versus what the known upstream sources allow."""

    declared: str | None
    status: str  # consistent | relabelled | upstream-conflict | restricted-redistribution | review | no-known-upstream | missing
    severity: str  # critical | warning | info | ok
    concluded: str | None  # most restrictive upstream licence, if any upstream is known
    commercial_training: bool | None
    redistribution: bool | None
    restrictions: list[str] = field(default_factory=list)
    upstream: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def match_sources(repo_id: str | None, readme: str | None) -> list[SourceMatch]:
    """Known upstream datasets this repo is probably a mirror, conversion or subset of."""
    rid = (repo_id or "").lower()
    text = (readme or "").lower()
    out = []
    for src in UPSTREAM_SOURCES:
        if src.link_pattern and re.search(src.link_pattern, text):
            out.append(SourceMatch(src.key, src.name, "card-link"))
        elif rid and re.search(src.id_pattern, rid):
            out.append(SourceMatch(src.key, src.name, "repo-id"))
        elif src.mention_pattern and re.search(src.mention_pattern, text):
            out.append(SourceMatch(src.key, src.name, "card-mention"))
    return out


def normalise_tag(lic) -> str | None:
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    if lic is None:
        return None
    s = str(lic).strip().lower()
    return s or None


def _same_license(declared: str, spdx: str) -> bool:
    return declared.replace("_", "-") == spdx.lower()


def assess(declared, matches: list[SourceMatch]) -> LicenseAssessment:
    tag = normalise_tag(declared)
    if not matches:
        return LicenseAssessment(
            declared=tag,
            status="missing" if tag is None else "no-known-upstream",
            severity="warning" if tag is None else "ok",
            concluded=None,
            commercial_training=None,
            redistribution=None,
        )
    sources = [_BY_KEY[m.key] for m in matches]
    restrictions = sorted({r for s in sources for r in s.restrictions})
    commercial = None if any(s.commercial_training is None for s in sources) else all(s.commercial_training for s in sources)
    redistribution = None if any(s.redistribution is None for s in sources) else all(s.redistribution for s in sources)
    # Most restrictive upstream licence: non-commercial / no-redistribution first.
    ranked = sorted(sources, key=lambda s: (s.commercial_training is not False, s.redistribution is not False))
    concluded = ranked[0].license if len({s.license for s in sources}) == 1 else " AND ".join(sorted({s.license for s in sources}))
    upstream = [
        {
            "key": s.key,
            "name": s.name,
            "license": s.license,
            "evidence": m.evidence,
            "statement_url": s.statement_url,
            "statement": s.statement,
            "checked": s.checked,
        }
        for s, m in zip(sources, matches)
    ]
    notes: list[str] = []
    evidence = {m.evidence for m in matches}
    if "card-link" in evidence:
        basis = "the card links to"
    elif "repo-id" in evidence:
        basis = "the repo name suggests it is derived from"
    else:
        basis = "the card mentions"
    names = ", ".join(s.name for s in sources)

    if tag is None:
        status, severity = "missing", "warning"
        notes.append(f"no licence declared; {basis} {names} ({concluded})")
    elif tag in ("other", "unknown"):
        status, severity = "review", "warning"
        notes.append(f"custom licence; {basis} {names} ({concluded}), so check the custom terms carry those restrictions")
    elif commercial is False and (tag in _PERMISSIVE_TAGS or tag.startswith("cc-by") and "nc" not in tag):
        status, severity = "upstream-conflict", "critical"
        notes.append(
            f"declared {tag}, but {basis} {names}, which is {concluded}: commercial training is not "
            "permitted upstream, and a mirror's tag cannot grant rights its uploader does not hold"
        )
    elif redistribution is False:
        consistent = all(_same_license(tag, s.license) for s in sources)
        status = "restricted-redistribution" if consistent else "upstream-conflict"
        severity = "warning" if consistent else "critical"
        notes.append(f"{basis} {names}, whose terms do not allow redistributing the data or adapted copies")
    elif all(_same_license(tag, s.license) for s in sources):
        status, severity = "consistent", "ok"
    elif tag in _PERMISSIVE_TAGS and all(s.license.startswith("CC-BY-") and "NC" not in s.license for s in sources):
        status, severity = "relabelled", "warning"
        notes.append(
            f"declared {tag}, but {basis} {names}, released under {concluded}; attribution and a notice of "
            "changes are still owed to the original authors"
        )
    else:
        status, severity = "upstream-conflict", "warning"
        notes.append(f"declared {tag} differs from upstream {concluded} ({names})")
    if evidence == {"repo-id"}:
        notes.append("match is by repo name only; confirm the source before relying on it")
    elif evidence == {"card-mention"}:
        notes.append("the card only names the upstream; it may be cited for comparison rather than as a source")
        if severity == "critical":
            severity = "warning"
    return LicenseAssessment(
        declared=tag,
        status=status,
        severity=severity,
        concluded=concluded,
        commercial_training=commercial,
        redistribution=redistribution,
        restrictions=restrictions,
        upstream=upstream,
        notes=notes,
    )


def assess_card(repo_id: str | None, readme: str | None, declared) -> LicenseAssessment:
    return assess(declared, match_sources(repo_id, readme))
