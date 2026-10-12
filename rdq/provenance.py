"""License and provenance analysis of a Hugging Face dataset card.

LeRobot's push_to_hub() defaults to license="apache-2.0" and fills the README
from a template whose Homepage, Paper and Citation fields read
"[More Information Needed]". A dataset whose card is still that untouched
template has a license nobody actually chose and no record of where the data
came from.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import yaml

from .licenses import assess_card, most_restrictive

TEMPLATE_MARKER = "This dataset was created using [LeRobot]"
UNFILLED = "[More Information Needed]"

NONCOMMERCIAL_LICENSES = re.compile(r"(^|-)nc(-|$)|noncommercial|non-commercial", re.I)
PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "odc-by",
    "pddl",
    "cdla-permissive-2.0",
    "openrail",
}

# Public datasets whose licences forbid commercial use or require a signed agreement.
RESTRICTED_SOURCES = {
    "epic-kitchens": "non-commercial licence",
    "epic kitchens": "non-commercial licence",
    "ego4d": "signed agreement that forbids redistribution",
    "egodex": "non-commercial, no-derivatives licence",
    "agibot world": "non-commercial, share-alike licence",
    "agibotworld": "non-commercial, share-alike licence",
    "hoi4d": "non-commercial licence",
    "aria everyday": "non-commercial research licence",
    "aria digital twin": "non-commercial research licence",
    "something-something": "non-commercial licence",
    "kinetics": "built from YouTube videos with mixed rights",
}


@dataclass
class ProvenanceResult:
    license: str | None
    license_class: str  # permissive | noncommercial | other | missing
    template_card: bool  # README is the unedited LeRobot template
    unfilled_fields: list[str]
    default_license_unconfirmed: bool  # apache-2.0 + untouched template
    restricted_sources: list[str]
    verdict: str  # short label used in reports
    notes: list[str] = field(default_factory=list)
    license_link: str | None = None
    source_datasets: list[str] = field(default_factory=list)
    upstream: dict | None = None  # licenses.LicenseAssessment as a dict
    bottom_line: str = ""  # one plain-English sentence: can I use this, and on what condition?
    bottom_level: str = "check"  # go | check | stop

    def to_dict(self) -> dict:
        return asdict(self)


def split_card(readme: str) -> tuple[dict, str]:
    """Return (front-matter dict, body) of a Hugging Face README."""
    if readme.startswith("---"):
        parts = readme.split("---", 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            return (meta if isinstance(meta, dict) else {}), parts[2]
    return {}, readme


def classify_license(license_value) -> str:
    if not license_value:
        return "missing"
    if isinstance(license_value, list):
        license_value = most_restrictive(license_value)
        if not license_value:
            return "missing"
    lic = str(license_value).strip().lower()
    if NONCOMMERCIAL_LICENSES.search(lic):
        return "noncommercial"
    if re.search(r"(^|-)nd(-|$)", lic):
        return "no-derivatives"
    if lic in PERMISSIVE_LICENSES or lic.startswith("cc-by-") and "nc" not in lic:
        return "permissive"
    return "other"


_MODEL_CONTEXT = re.compile(r"pre-?train|encoder|backbone|representation|embedding|features from|r3m|vc-1|vip\b")


def _mentioned_as_data(name: str, text: str) -> bool:
    """True if the card names a restricted source other than as a pretrained model's training data."""
    for m in re.finditer(re.escape(name), text):
        window = text[max(0, m.start() - 80) : m.end() + 40]
        if not _MODEL_CONTEXT.search(window):
            return True
    return False


def bottom_line(verdict: str, lic, lic_class: str, up: dict, default_unconfirmed: bool) -> tuple[str, str]:
    """(go | check | stop, one sentence) answering: can I train on this, and on what condition?"""
    names = ", ".join(u["name"] for u in up.get("upstream") or []) or "the original dataset"
    status = up.get("status")
    if up.get("commercial_training") is False and status == "upstream-conflict":
        return "stop", (f"Do not use it commercially: it appears to come from {names}, which only allows non-commercial use, "
                        "whatever this copy's tag says. Confirm the source before relying on it.")
    if lic_class == "missing":
        return "stop", "No licence: nobody has given you permission to reuse this data. Ask the uploader before using it."
    if lic_class == "noncommercial":
        return "stop", f"Research and other non-commercial use only ({lic})."
    if status == "relabelled":
        return "go", (f"Usable, including commercially, as long as you credit {names} and say you changed it "
                      f"({up.get('concluded')}). The {lic} tag on this copy does not remove that duty.")
    if status == "no-upstream-grant":
        return "check", (f"Unclear: {names} never put a licence on the data itself. Ask its authors before commercial use; "
                         f"the {lic or 'missing'} tag on this copy is not their permission.")
    if status in ("restricted-redistribution",) or verdict == "restricted source":
        return "check", f"Read the original terms first: {names} restricts how the data and copies of it may be shared."
    if status == "upstream-conflict":
        return "check", f"The tag here ({lic}) differs from the original's terms ({up.get('concluded') or 'see notes'}); follow the original's terms."
    if status in ("consistent", "publisher-upload", "stricter-than-upstream"):
        return "go", f"Follow the {lic} licence{' and credit ' + names if str(lic).lower().startswith('cc-by') else ''}."
    if default_unconfirmed:
        return "check", ("The apache-2.0 tag looks like the upload tool's default, not a choice by the owner. Ask the "
                         "uploader to confirm it before relying on it, especially for commercial use.")
    if lic_class == "no-derivatives":
        return "check", f"You may share it unchanged, but the {lic} licence forbids sharing modified versions."
    if lic_class == "permissive":
        return "go", f"Follow the {lic} licence."
    return "check", f"Custom licence ({lic}): read its terms before use."


def analyse_card(readme: str | None, card_license=None, repo_id: str | None = None) -> ProvenanceResult:
    meta, body = split_card(readme or "")
    lic = card_license if card_license is not None else meta.get("license")
    if isinstance(lic, list):
        lic = most_restrictive(lic)
    lic_class = classify_license(lic)

    template = TEMPLATE_MARKER in body
    unfilled = []
    for label in ("Homepage", "Paper", "License", "BibTeX"):
        pattern = re.compile(rf"{label}[^\n]*\n?[^\n]*{re.escape(UNFILLED)}", re.I)
        if pattern.search(body):
            unfilled.append(label.lower())
    description = ""
    match = re.search(r"## Dataset Description\s*(.*?)\s*- \*\*Homepage", body, re.S)
    if match:
        description = match.group(1).strip()
    untouched = template and not description and {"homepage", "paper"} <= set(unfilled)

    lower = (readme or "").lower()
    restricted = sorted({name for name in RESTRICTED_SOURCES if _mentioned_as_data(name, lower)})

    default_unconfirmed = untouched and str(lic or "").lower() == "apache-2.0"
    notes = []
    if default_unconfirmed:
        notes.append(
            "licence is apache-2.0 but the card is LeRobot's unedited template, so the licence "
            "is most likely the upload tool's default rather than a choice by the data owner"
        )
    elif untouched:
        notes.append("card is LeRobot's unedited template: no description, homepage, paper or citation")
    for name in restricted:
        notes.append(f"card mentions {name}, whose data has a {RESTRICTED_SOURCES[name]}")

    upstream = assess_card(repo_id, readme, lic)
    notes.extend(upstream.notes)
    src = meta.get("source_datasets") or []
    if isinstance(src, str):
        src = [src]

    if upstream.status == "upstream-conflict" and upstream.severity == "critical":
        verdict = "licence conflicts with upstream"
    elif lic_class == "missing":
        verdict = "no licence"
    elif lic_class == "noncommercial":
        verdict = "non-commercial"
    elif lic_class == "no-derivatives":
        verdict = "no-derivatives licence"
    elif upstream.status == "relabelled" and upstream.severity == "warning":
        verdict = "relabelled (attribution owed)"
    elif restricted or upstream.status == "restricted-redistribution":
        verdict = "restricted source"
    elif upstream.status == "upstream-conflict" and upstream.severity == "warning":
        verdict = "differs from upstream"
    elif upstream.status == "no-upstream-grant" and upstream.severity == "warning":
        verdict = "no upstream data licence"
    elif default_unconfirmed:
        verdict = "default licence, unconfirmed"
    elif untouched:
        verdict = "licensed, no provenance"
    elif lic_class == "permissive":
        verdict = "documented"
    else:
        verdict = "custom licence"
    level, line = bottom_line(verdict, lic, lic_class, upstream.to_dict(), default_unconfirmed)
    return ProvenanceResult(
        bottom_line=line,
        bottom_level=level,
        license=str(lic) if lic else None,
        license_class=lic_class,
        template_card=template,
        unfilled_fields=unfilled,
        default_license_unconfirmed=default_unconfirmed,
        restricted_sources=restricted,
        verdict=verdict,
        notes=notes,
        license_link=str(meta["license_link"]) if meta.get("license_link") else None,
        source_datasets=[str(x) for x in src],
        upstream=upstream.to_dict(),
    )
