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
    confidence: str = "high"  # high: explicit data licence; medium: umbrella/indirect; low: uncertain mapping
    notes: str = ""


NOT_STATED = "not stated"


def _load_sources() -> tuple[UpstreamSource, ...]:
    """Read the packaged knowledge base (rdq/data/upstream_licences.json)."""
    import json
    from importlib.resources import files

    data = json.loads(files("rdq").joinpath("data/upstream_licences.json").read_text())
    out = []
    for e in data["sources"]:
        out.append(
            UpstreamSource(
                key=e["key"],
                name=e["name"],
                license=e["license"],
                commercial_training=e.get("commercial_training"),
                redistribution=e.get("redistribution"),
                restrictions=tuple(e.get("restrictions") or ()),
                statement_url=e["statement_url"],
                statement=e["statement"],
                checked=e.get("checked", ""),
                id_pattern=e["id_pattern"],
                link_pattern=e.get("link_pattern"),
                mention_pattern=e.get("mention_pattern"),
                confidence=e.get("confidence", "high"),
                notes=e.get("notes", ""),
            )
        )
    return tuple(out)


# Each entry was checked against a primary source; see the JSON file for statements and dates.
UPSTREAM_SOURCES: tuple[UpstreamSource, ...] = _load_sources()

_BY_KEY = {s.key: s for s in UPSTREAM_SOURCES}

# Declared tags that grant broad permissive rights. A CC BY upstream relabelled as one of
# these loses its attribution and change-notice duties on paper (not in law).
_PERMISSIVE_TAGS = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc0-1.0", "pddl", "odc-by", "unlicense"}


@dataclass
class SourceMatch:
    key: str
    name: str
    evidence: str  # card-source | card-link | repo-id | card-mention | publisher (see match_sources)
    corroborated: bool = False  # two independent signals agree, or the card declares the source

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LicenseAssessment:
    """Declared licence versus what the known upstream sources allow."""

    declared: str | None
    status: str  # consistent | relabelled | upstream-conflict | restricted-redistribution | stricter-than-upstream | publisher-upload | review | no-upstream-grant | no-known-upstream | missing
    severity: str  # critical | warning | info | ok
    concluded: str | None  # most restrictive upstream licence, if any upstream is known
    commercial_training: bool | None
    redistribution: bool | None
    restrictions: list[str] = field(default_factory=list)
    upstream: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Words near an upstream link that say the data came from there (rather than citing it).
_DERIVATION = re.compile(
    r"\bconver(t|ted|ting|sion)s?\b|\bre-?conver|\bderived\b|\bderivative\b|\bsubset of\b|\bmirror of\b|"
    r"\bported\b|\bport of\b|\bre-?upload|\brepackag|\breformat|\brepublish|\bre-?render|"
    r"\bprocessed (version|copy)\b|\btaken from\b|\bcomes from\b|\bdownloaded from\b|\bcopy of\b|"
    r"\b(release|version|subset|split|sample) of\b|\boriginal (dataset|data|release)\b|\bsource dataset|→|->"
)
# Words around a mention that say the upstream is only cited: a pretrained model, a comparison,
# a benchmark or a format, not the source of the data.
_CITATION_CONTEXT = re.compile(
    r"pre-?train|encoder|backbone|representation|embedding|features from|\br3m\b|vc-1|\bvip\b|"
    r"\bcompar|in the style of|similar to|inspired by|evaluated on|\bbenchmark|\bbaseline|follows the|"
    r"\bformat\b|like the|unlike"
)


def _front_matter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    try:
        import yaml

        meta = yaml.safe_load(text[3:end]) or {}
    except Exception:  # noqa: BLE001 - a broken card is reported elsewhere
        return {}
    return meta if isinstance(meta, dict) else {}


def _front_matter_sources(text: str) -> list[str]:
    """source_datasets listed in the card's YAML front matter (lower-cased)."""
    src = _front_matter(text).get("source_datasets")
    if isinstance(src, str):
        src = [src]
    return [str(x).lower() for x in src or []]


def _link_is_derivation(pattern: str, text: str) -> bool:
    for m in re.finditer(pattern, text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        line = text[line_start : text.find("\n", m.end()) if "\n" in text[m.end():] else len(text)]
        if re.search(r"homepage\W{0,6}[:|]", line):  # the card's own Homepage field points there
            return True
        window = text[max(0, m.start() - 120) : m.end() + 120]
        if _DERIVATION.search(window):
            return True
    return False


def _mentioned_as_source(pattern: str, text: str) -> bool:
    """A name mention counts only if at least one occurrence is not in a citation context."""
    for m in re.finditer(pattern, text):
        window = text[max(0, m.start() - 80) : m.end() + 40]
        if not _CITATION_CONTEXT.search(window):
            return True
    return False


def match_sources(repo_id: str | None, readme: str | None) -> list[SourceMatch]:
    """Known upstream datasets this repo is probably a mirror, conversion or subset of.

    Evidence, strongest first: the card's ``source_datasets`` (or ``license_link``) names it
    (card-source); the card links to it in a derivation context such as its Homepage field or
    "converted from" (card-link); the repo name matches (repo-id); the card only names it
    (card-mention, ignored when it is cited as a model's training data, a benchmark or a
    format). A repo owned by the upstream's own organisation is the publisher's upload
    (publisher). A match is *corroborated* when two independent signals agree, or when the
    card itself declares the source.
    """
    rid = (repo_id or "").lower()
    text = (readme or "").lower()
    declared_sources = _front_matter_sources(text)
    license_link = str(_front_matter(text).get("license_link") or "").lower()
    out = []
    for src in UPSTREAM_SOURCES:
        org_anchored = bool(re.match(r"\^[\w.-]+/", src.id_pattern))  # e.g. ^robocoin/ or ^unitreerobotics/g1
        by_id = bool(rid) and bool(re.search(src.id_pattern, rid))
        if by_id and org_anchored:
            out.append(SourceMatch(src.key, src.name, "publisher", True))
            continue
        if org_anchored:
            by_id = False
        declared = any(re.search(src.id_pattern, d) for d in declared_sources) or bool(
            src.link_pattern and license_link and re.search(src.link_pattern, license_link))
        linked = bool(src.link_pattern and re.search(src.link_pattern, text))
        link_derives = linked and _link_is_derivation(src.link_pattern, text)
        mentioned = bool(src.mention_pattern and _mentioned_as_source(src.mention_pattern, text))
        if declared:
            out.append(SourceMatch(src.key, src.name, "card-source", True))
        elif link_derives:
            out.append(SourceMatch(src.key, src.name, "card-link", by_id))
        elif by_id:
            out.append(SourceMatch(src.key, src.name, "repo-id", linked or mentioned))
        elif mentioned or (linked and _mentioned_as_source(src.link_pattern, text)):
            out.append(SourceMatch(src.key, src.name, "card-mention", False))
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


_SEVERITY_RANK = {"ok": 0, "info": 1, "warning": 2, "critical": 3}
_STATUS_RANK = {
    "upstream-conflict": 6, "restricted-redistribution": 5, "relabelled": 4, "review": 3,
    "missing": 2, "stricter-than-upstream": 1, "consistent": 0, "publisher-upload": 0,
}
_BASIS = {
    "card-source": "the card's source_datasets names",
    "card-link": "the card links to",
    "repo-id": "the repo name suggests it is derived from",
    "card-mention": "the card mentions",
    "publisher": "it is uploaded by the publisher of",
}


def _tri(values) -> bool | None:
    """Most restrictive combination: any False wins, then any unknown, else True."""
    values = list(values)
    if any(v is False for v in values):
        return False
    if any(v is None for v in values):
        return None
    return True


def _is_nc(tag: str) -> bool:
    return bool(re.search(r"(^|-)nc(-|$)", tag))


def _adds_terms(tag: str, upstream: str) -> bool:
    """A Creative Commons tag that keeps the upstream's terms and adds NC, ND or SA."""
    t, u = tag.lower().replace("_", "-"), upstream.lower()
    if not (t.startswith("cc-by") and u.startswith("cc-by")):
        return False
    terms = lambda x: {m for m in ("nc", "nd", "sa") if re.search(rf"(^|-){m}(-|$)", x)}  # noqa: E731
    return terms(u) < terms(t)


def _assess_one(tag: str | None, s: UpstreamSource, evidence: str, rid: str) -> tuple[str, str, str]:
    """(status, severity, note) for one matched source, before any cap."""
    basis = f"{_BASIS[evidence]} {s.name}"
    if tag is None:
        return "missing", "warning", f"no licence declared; {basis} ({s.license})"
    if tag in ("other", "unknown"):
        return "review", "warning", f"custom licence; {basis} ({s.license}), so check the custom terms carry its restrictions"
    if s.commercial_training is False and not _is_nc(tag):
        return "upstream-conflict", "critical", (
            f"declared {tag}, but {basis}, which is {s.license}: commercial training is not permitted upstream, "
            "and a mirror's tag cannot grant rights its uploader does not hold")
    if s.redistribution is False:
        if _same_license(tag, s.license):
            return "restricted-redistribution", "warning", f"{basis}, whose terms do not allow redistributing the data or adapted copies"
        return "upstream-conflict", "critical", f"{basis}, whose terms do not allow redistributing the data or adapted copies"
    if _same_license(tag, s.license):
        return "consistent", "ok", ""
    if _adds_terms(tag, s.license):
        return "stricter-than-upstream", "info", (
            f"declared {tag} adds restrictions to {s.name} ({s.license}); both sets of terms apply to this copy")
    if tag in _PERMISSIVE_TAGS and s.license.startswith("CC-BY-") and "NC" not in s.license:
        return "relabelled", "warning", (
            f"declared {tag}, but {basis}, released under {s.license}; attribution and a notice of changes are "
            "still owed to the original authors")
    return "upstream-conflict", "warning", f"declared {tag} differs from upstream {s.license} ({basis})"


def _cap(severity: str, s: UpstreamSource, m: SourceMatch) -> str:
    """Limit severity by how sure we are of the upstream terms and of the match itself.

    A single weak signal (a repo name alone, a link alone, a mention) is never critical;
    the card's own source_datasets, or two independent signals that agree, can be.
    """
    cap = "critical"
    if s.confidence == "medium":
        cap = "warning"
    elif s.confidence == "low":
        cap = "info"
    if m.evidence == "publisher":
        cap = min(cap, "info", key=_SEVERITY_RANK.get)
    elif m.evidence == "card-mention" or not m.corroborated:
        cap = min(cap, "warning", key=_SEVERITY_RANK.get)
    return min(severity, cap, key=_SEVERITY_RANK.get)


def assess(declared, matches: list[SourceMatch], repo_id: str | None = None) -> LicenseAssessment:
    tag = normalise_tag(declared)
    rid = (repo_id or "").lower()
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

    def describe(src, m):
        return {"key": src.key, "name": src.name, "license": src.license, "evidence": m.evidence,
                "corroborated": m.corroborated, "statement_url": src.statement_url, "statement": src.statement,
                "checked": src.checked, "confidence": src.confidence}

    if all(m.evidence == "publisher" for m in matches):
        same = tag is not None and all(_same_license(tag, s.license) for s in sources if s.license != NOT_STATED)
        return LicenseAssessment(
            declared=tag,
            status="publisher-upload",
            severity="ok" if same else ("warning" if tag is None else "info"),
            concluded=None,
            commercial_training=None,
            redistribution=None,
            upstream=[describe(s, m) for s, m in zip(sources, matches)],
            notes=[] if same else [f"uploaded by the publisher of {', '.join(s.name for s in sources)}, whose tag on this "
                                   f"upload governs; it differs from their published terms ({', '.join(s.license for s in sources)})"],
        )

    if all(s.license == NOT_STATED for s in sources):
        names = ", ".join(s.name for s in sources)
        sev = "warning" if tag else "info"
        if all(m.evidence in ("publisher", "card-mention") for m in matches):
            sev = "info"
        return LicenseAssessment(
            declared=tag,
            status="no-upstream-grant",
            severity=sev,
            concluded=None,
            commercial_training=None,
            redistribution=None,
            upstream=[describe(s, m) for s, m in zip(sources, matches)],
            notes=[f"appears to be derived from {names}, whose original release states no data licence (only a code "
                   f"licence), so the declared '{tag or 'none'}' cannot be traced to a grant from the data's authors"],
        )
    pairs = [(s, m) for s, m in zip(sources, matches) if s.license != NOT_STATED]
    sources = [s for s, _ in pairs]
    matches = [m for _, m in pairs]
    # Only matches that say the data came from the upstream decide the concluded terms; a mere
    # mention (or the publisher's own upload) leaves the declared licence in force.
    strong = [s for s, m in pairs if m.evidence in ("card-source", "card-link", "repo-id")]
    restrictions = sorted({r for s in strong for r in s.restrictions})
    commercial = _tri(s.commercial_training for s in strong) if strong else None
    redistribution = _tri(s.redistribution for s in strong) if strong else None
    lics = sorted({s.license for s in strong})
    concluded = (lics[0] if len(lics) == 1 else " AND ".join(lics)) if lics else None

    # Judge each source on its own evidence and confidence; the worst one decides.
    per = []
    for s, m in pairs:
        status_s, sev_s, note_s = _assess_one(tag, s, m.evidence, rid)
        per.append((status_s, _cap(sev_s, s, m), note_s, s, m))
    worst = max(per, key=lambda p: (_SEVERITY_RANK[p[1]], _STATUS_RANK[p[0]]))
    status, severity = worst[0], worst[1]
    notes = [p[2] for p in sorted(per, key=lambda p: -_SEVERITY_RANK[p[1]]) if p[2]]
    notes = list(dict.fromkeys(notes))
    if any(s.confidence == "medium" for s in sources):
        notes.append("some upstream terms come from an umbrella or indirect statement (e.g. the Open X-Embodiment "
                     "README), not the component's own release")
    if any(s.confidence == "low" for s in sources):
        notes.append("some upstream licence mappings are uncertain; treat those as leads, not findings")
    if status in ("relabelled", "upstream-conflict"):
        notes.append("if the uploader is the original rights holder, the new tag may be a deliberate relicence")
    evidence = {m.evidence for m in matches}
    if evidence == {"repo-id"}:
        notes.append("match is by repo name only; confirm the source before relying on it")
    elif evidence == {"card-mention"}:
        notes.append("the card only names the upstream; it may be cited for comparison rather than as a source")
    if any(m.evidence in ("card-link", "repo-id") and not m.corroborated for m in matches) and severity == "warning" \
            and any(p[0] == "upstream-conflict" and _SEVERITY_RANK[_assess_one(tag, p[3], p[4].evidence, rid)[1]] == 3
                    for p in per):
        notes.append("this would be critical if a second signal confirmed the source (source_datasets on the card, "
                     "or both a derivation link and the repo name)")
    return LicenseAssessment(
        declared=tag,
        status=status,
        severity=severity,
        concluded=concluded,
        commercial_training=commercial,
        redistribution=redistribution,
        restrictions=restrictions,
        upstream=[describe(s, m) for s, m in pairs],
        notes=notes,
    )


def assess_card(repo_id: str | None, readme: str | None, declared) -> LicenseAssessment:
    return assess(declared, match_sources(repo_id, readme), repo_id=repo_id)
