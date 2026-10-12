"""Data passport: a machine-readable provenance, licence and quality record for a dataset.

The passport is a Croissant 1.1 JSON-LD document with the Responsible-AI (RAI) vocabulary
and W3C PROV-O terms, plus an `rdq:` extension for what no standard covers yet: licence
evidence and conflict status, quality findings and a content hash. The same record can be
exported as an SPDX 3.0 dataset package (an "AI-BOM" entry) and as a human-readable
datasheet in Markdown.

Fields come from four places, and the passport says which:
  data   - computed from the dataset files (structure, counts, hashes, quality)
  card   - parsed from the Hugging Face dataset card (declared licence, links, sources)
  rdq    - derived by rdq (licence status, upstream conflicts, restrictions)
  owner  - supplied by the dataset owner in an answers file (collection method, operators,
           timeframe, personal data, intended use). rdq never invents these.

Nothing here is legal advice. A passport records evidence; it does not grant rights.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .provenance import split_card

RDQ_NS = "https://github.com/s1g6r/rdq/blob/main/docs/passport.md#"
PASSPORT_VERSION = "0.1"

# Hugging Face licence tags -> SPDX identifiers.
HF_TO_SPDX = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "cc0-1.0": "CC0-1.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc-by-nc-nd-4.0": "CC-BY-NC-ND-4.0",
    "cc-by-nd-4.0": "CC-BY-ND-4.0",
    "odc-by": "ODC-By-1.0",
    "pddl": "PDDL-1.0",
    "gpl-3.0": "GPL-3.0-only",
    "cdla-permissive-2.0": "CDLA-Permissive-2.0",
    "unlicense": "Unlicense",
    "cc-by-3.0": "CC-BY-3.0",
    "cc-by-sa-3.0": "CC-BY-SA-3.0",
    "cc-by-nc-3.0": "CC-BY-NC-3.0",
    "cc-by-nc-sa-3.0": "CC-BY-NC-SA-3.0",
    "cc-by-2.0": "CC-BY-2.0",
    "gpl-2.0": "GPL-2.0-only",
    "lgpl-3.0": "LGPL-3.0-only",
    "lgpl-2.1": "LGPL-2.1-only",
    "agpl-3.0": "AGPL-3.0-only",
    "afl-3.0": "AFL-3.0",
    "mpl-2.0": "MPL-2.0",
    "bsl-1.0": "BSL-1.0",
    "isc": "ISC",
    "wtfpl": "WTFPL",
    "openrail": "LicenseRef-openrail",
    "odbl": "ODbL-1.0",
}

# Questions only the data owner can answer. Written out by `rdq passport --init-answers`.
OWNER_QUESTIONS: dict[str, str] = {
    "creator": "Person or organisation that collected the data.",
    "description": "One paragraph: what the dataset contains and what it is for.",
    "collection_method": "How the demonstrations were produced: leader-arm teleop, VR, spacemouse, "
    "kinesthetic teaching, scripted policy, simulation, human video, ...",
    "collection_type": "One or more of: Experiments, Simulations, Crowdsourcing, Secondary Data analysis, Others.",
    "collection_timeframe": "Start and end date of collection, e.g. [2026-05-01, 2026-06-15].",
    "collection_geography": "Where data was recorded (country or region) and the kind of environment (lab, home, warehouse).",
    "operators": "How many operators, how they were trained, and whether episodes record an operator id.",
    "upstream_sources": "If this was converted, merged or filtered from other datasets, list them with links and revisions.",
    "modifications": "What you changed relative to those sources (format conversion, re-timing, filtering, re-captioning).",
    "task_annotation": "Who wrote the task strings (operators, annotators, a VLM such as Qwen2.5-VL) and how.",
    "personal_data": "Do cameras or audio capture people? Faces? If yes: lawful basis, consent reference, anonymisation.",
    "intended_use": "Uses the data is suitable for (pre-training, fine-tuning, evaluation) and uses it is not.",
    "limitations": "Known limitations: one robot, one scene, low frame rate, missing failures, ...",
    "known_biases": "Known biases: operator style, object set, lighting, demographics of people in view.",
    "maintenance": "Who maintains the dataset and how errors are reported or fixed.",
    "license_confirmed_by": "Name of the person confirming the licence is what the rights holder intends (leave empty if unsure).",
}


def spdx_id(tag) -> str | None:
    if tag is None or not str(tag).strip():
        return None
    raw = str(tag).strip()
    t = raw.lower()
    if t in HF_TO_SPDX:
        return HF_TO_SPDX[t]
    body = raw[len("licenseref-"):] if t.startswith("licenseref-") else raw
    # SPDX LicenseRef idstrings allow only letters, digits, '.' and '-'; keep '+' distinguishable.
    body = re.sub(r"[^A-Za-z0-9.+-]+", "-", body).replace("+", "-plus").strip("-")
    return "LicenseRef-" + (body or "unknown")


_RESTRICTION_MARKS = {"nc": "-nc", "nd": "-nd", "sa": "-sa"}


def _adds_restrictions(declared, concluded: str | None) -> bool:
    """True if the declared tag carries an NC/ND/SA term the upstream licence lacks."""
    if not declared or not concluded:
        return False
    d = "-" + str(declared).lower().replace("_", "-") + "-"
    c = "-" + concluded.lower() + "-"
    return any(f"{m}-" in d and f"{m}-" not in c for m in _RESTRICTION_MARKS.values())


def license_url(spdx: str | None) -> str | None:
    if not spdx or spdx.startswith("LicenseRef-"):
        return None
    if spdx.startswith("CC-"):
        parts = spdx.lower().split("-")  # cc-by-nc-4.0
        kind = "-".join(parts[1:-1])
        return f"https://creativecommons.org/licenses/{kind}/{parts[-1]}/" if kind else None
    return f"https://spdx.org/licenses/{spdx}.html"


def content_hash(root: str | Path, include_videos: bool = False) -> dict:
    """SHA-256 over every file's path and digest (meta/ and data/, optionally videos/).

    Two copies of a dataset with the same hash have identical tables and metadata, so a
    passport can be tied to an exact version even off the Hub.
    """
    root = Path(root)
    dirs = ["meta", "data"] + (["videos"] if include_videos else [])
    entries = []
    total_bytes = 0
    for d in dirs:
        base = root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file():
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                rel = p.relative_to(root).as_posix()
                entries.append(f"{rel}\t{h.hexdigest()}")
                total_bytes += p.stat().st_size
    top = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return {"algorithm": "sha256", "value": top, "files": len(entries), "bytes": total_bytes, "covers": dirs}


def load_answers(path: str | Path | None) -> dict:
    if not path:
        return {}
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping of question -> answer")
    unknown = sorted(set(data) - set(OWNER_QUESTIONS))
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; see `rdq passport --init-answers`")
    return {k: v for k, v in data.items() if v not in (None, "", [])}


def answers_template() -> str:
    lines = [
        "# rdq data passport: answers only the dataset owner can give.",
        "# Leave anything you don't know empty. Unanswered items are reported as missing, never guessed.",
        "",
    ]
    for key, question in OWNER_QUESTIONS.items():
        lines.append(f"# {question}")
        lines.append(f"{key}:")
        lines.append("")
    return "\n".join(lines)


def _text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x) for x in v)
    return str(v)


def _quality_block(result: dict) -> dict:
    sc = result["score"]
    by_check: dict[str, dict] = {}
    for f in result["findings"]:
        b = by_check.setdefault(f["check"], {"severity": f["severity"], "episodes": set(), "messages": 0})
        b["episodes"].update(f.get("episodes") or [])
        b["messages"] += 1
    return {
        "score": sc["dataset_score"],
        "severityCounts": sc["severity_counts"],
        "episodesTotal": result["dataset"]["episodes"],
        "episodesUsable": len(sc["usable_episodes"]),
        "usableFraction": sc["usable_fraction"],
        "findings": [
            {"check": k, "severity": v["severity"], "episodes": len(v["episodes"]), "findings": v["messages"]}
            for k, v in sorted(by_check.items())
        ],
        "thresholds": result.get("thresholds"),
    }


def build_passport(
    result: dict,
    readme: str | None,
    root: str | Path,
    repo_id: str | None = None,
    revision: str | None = None,
    answers: dict | None = None,
    include_videos_in_hash: bool = False,
) -> dict:
    """Assemble the passport (Croissant 1.1 + RAI + PROV-O JSON-LD) from an audit result."""
    answers = answers or {}
    meta, _ = split_card(readme or "")
    prov = result.get("provenance") or {}
    up = prov.get("upstream") or {}
    ds = result["dataset"]
    name = repo_id or Path(root).name
    url = f"https://huggingface.co/datasets/{repo_id}" if repo_id else None

    declared_spdx = spdx_id(prov.get("license"))
    concluded = up.get("concluded")
    declared_adds = _adds_restrictions(prov.get("license"), concluded)
    if up.get("status") == "no-upstream-grant":
        concluded_for_croissant = None  # the original states no data licence, so nothing can be concluded
    elif concluded and declared_adds and declared_spdx:
        # A publisher may add restrictions (e.g. NC on a CC BY derivative); both sets of terms apply.
        concluded_for_croissant = f"{concluded} AND {declared_spdx}"
    elif concluded:
        concluded_for_croissant = concluded  # upstream terms govern whatever the mirror's tag says
    elif prov.get("default_license_unconfirmed") and not answers.get("license_confirmed_by"):
        concluded_for_croissant = None  # NOASSERTION: nobody has confirmed the tag
    else:
        concluded_for_croissant = declared_spdx
    commercial_training = up.get("commercial_training")
    if prov.get("license_class") == "noncommercial":
        commercial_training = False  # the declared terms themselves forbid it
    lic_status = _license_status(prov, up, answers)

    missing = [k for k in OWNER_QUESTIONS if k not in answers]
    hashed = content_hash(root, include_videos=include_videos_in_hash)

    derived = []
    for u in up.get("upstream") or []:
        derived.append(
            {
                "@id": u["statement_url"],
                "name": u["name"],
                "rdq:license": u["license"],
                "rdq:evidence": u["evidence"],
            }
        )
    for s in prov.get("source_datasets") or []:
        derived.append({"@id": f"https://huggingface.co/datasets/{s}", "rdq:evidence": "card source_datasets"})
    if answers.get("upstream_sources"):
        derived.append({"rdq:ownerStatement": _text(answers["upstream_sources"])})
    if answers.get("modifications") and derived:
        for d in derived:
            d["rdq:modifications"] = _text(answers["modifications"])

    preprocessing = [
        f"rdq {__version__} audit: score {result['score']['dataset_score']}/100, "
        f"{len(result['score']['usable_episodes'])}/{ds['episodes']} episodes pass all critical and warning checks."
    ]
    if answers.get("modifications"):
        preprocessing.append(_text(answers["modifications"]))

    doc: dict = {
        "@context": _context(),
        "@type": "sc:Dataset",
        "conformsTo": ["http://mlcommons.org/croissant/1.1", "http://mlcommons.org/croissant/RAI/1.0"],
        "name": name,
        "description": _text(answers.get("description")) or meta.get("pretty_name") or f"LeRobot dataset {name}",
        "url": url or f"urn:rdq:local:{re.sub(r'[^A-Za-z0-9._-]', '-', name)}",
        "creator": {"@type": "sc:Organization", "name": _text(answers.get("creator")) or (repo_id.split("/")[0] if repo_id else "unknown")},
        **({"datePublished": str(answers["date_published"])} if answers.get("date_published") else {}),
        "dateModified": str(date.today()),
        "version": revision if revision and re.fullmatch(r"\d+\.\d+\.\d+", revision) else "0.0.0",
        "rdq:pinnedRevision": revision,
        "citeAs": _cite_as(readme),
        "keywords": ["robotics", "LeRobot", "imitation learning"] + ([ds["robot_type"]] if ds.get("robot_type") else []),
        "sdVersion": f"rdq-passport-{PASSPORT_VERSION}",
        "sdLicense": "https://spdx.org/licenses/CC0-1.0.html",
        "distribution": _distribution(ds, Path(root)),
        "recordSet": _record_sets(Path(root)),
        "rai:dataCollection": _text(answers.get("collection_method")),
        "rai:dataCollectionType": answers.get("collection_type"),
        "rai:dataCollectionTimeframe": answers.get("collection_timeframe"),
        "rai:dataPreprocessingProtocol": preprocessing,
        "rai:dataAnnotationProtocol": _text(answers.get("task_annotation")),
        "rai:annotatorDemographics": _text(answers.get("operators")),
        "rai:personalSensitiveInformation": _text(answers.get("personal_data")),
        "rai:dataUseCases": _text(answers.get("intended_use")),
        "rai:dataLimitations": _limitations(result, answers),
        "rai:dataBiases": _text(answers.get("known_biases")),
        "rai:dataReleaseMaintenancePlan": _text(answers.get("maintenance")),
        "rai:dataCollectionMissingData": _missing_data(result),
        "prov:wasDerivedFrom": derived or None,
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "prov:label": f"LeRobot {ds.get('codebase_version')} dataset" + (f", robot {ds['robot_type']}" if ds.get("robot_type") else ""),
        },
        "rdq:structure": {
            "codebaseVersion": ds.get("codebase_version"),
            "robotType": ds.get("robot_type"),
            "fps": ds.get("fps"),
            "episodes": ds.get("episodes"),
            "frames": ds.get("frames"),
            "hours": round(ds["frames"] / ds["fps"] / 3600, 3) if ds.get("fps") else None,
            "tasks": ds.get("tasks"),
            "numericFeatures": ds.get("numeric_features"),
            "cameras": ds.get("cameras"),
        },
        "rdq:contentHash": hashed,
        "rdq:license": {
            "declared": prov.get("license"),
            "declaredSpdx": declared_spdx,
            "declaredSource": "dataset card front matter" if prov.get("license") else None,
            "licenseLink": prov.get("license_link"),
            "status": lic_status,
            "verdict": prov.get("verdict"),
            "concluded": concluded_for_croissant or "NOASSERTION",
            "commercialTraining": commercial_training,
            "redistribution": up.get("redistribution"),
            "restrictions": up.get("restrictions") or [],
            "templateCard": prov.get("template_card"),
            "confirmedBy": _text(answers.get("license_confirmed_by")),
            "evidence": [
                {"statementUrl": u["statement_url"], "statement": u["statement"], "checkedAt": u["checked"], "match": u["evidence"], "reviewer": "rdq-auto"}
                for u in up.get("upstream") or []
            ],
            "notes": prov.get("notes") or [],
        },
        "rdq:quality": _quality_block(result),
        "rdq:fieldSources": {
            "data": ["distribution", "rdq:structure", "rdq:contentHash", "rdq:quality", "rai:dataCollectionMissingData"],
            "card": ["rdq:license.declared", "rdq:license.licenseLink", "prov:wasDerivedFrom (source_datasets)"],
            "rdq": ["rdq:license.status", "rdq:license.concluded", "rdq:license.restrictions", "prov:wasDerivedFrom (upstream matches)"],
            "owner": sorted(answers),
        },
        "rdq:unanswered": missing,
        "rdq:attestation": {
            "generatedBy": f"rdq {__version__}",
            "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "signedBy": None,
            "note": "Machine-generated. Owner answers are recorded as given and not verified by rdq. Not legal advice.",
        },
    }
    lic_url = license_url(concluded_for_croissant if concluded_for_croissant and " AND " not in concluded_for_croissant else None)
    doc["license"] = lic_url or (concluded_for_croissant if concluded_for_croissant else "NOASSERTION")
    if commercial_training is False or up.get("redistribution") is False:
        doc["usageInfo"] = _usage_info(up, commercial_training)
    return {k: v for k, v in doc.items() if v not in (None, [], "")}


def _cite_as(readme: str | None) -> str | None:
    m = re.search(r"```(?:bibtex)?\s*(@\w+\{.*?\})\s*```", readme or "", re.S)
    return m.group(1).strip() if m and "[More Information Needed]" not in m.group(1) else None


def _license_status(prov: dict, up: dict, answers: dict) -> str:
    status = up.get("status") or "unknown"
    if status == "no-known-upstream":
        if prov.get("default_license_unconfirmed"):
            status = "default-unconfirmed"
        elif prov.get("license"):
            status = "explicit"
    if answers.get("license_confirmed_by") and status in ("default-unconfirmed", "explicit"):
        status = "owner-confirmed"
    return status


def _usage_info(up: dict, commercial_training: bool | None = None) -> dict:
    prohibitions = []
    if up.get("commercial_training") is False or commercial_training is False:
        prohibitions.append({"odrl:action": "odrl:commercialize"})
    if up.get("redistribution") is False:
        prohibitions.append({"odrl:action": "odrl:distribute"})
    return {"@type": "odrl:Offer", "odrl:prohibition": prohibitions, "rdq:basis": up.get("concluded")}


def _limitations(result: dict, answers: dict) -> str | None:
    parts = []
    if answers.get("limitations"):
        parts.append(_text(answers["limitations"]))
    crit = result["score"]["severity_counts"].get("critical", 0)
    warn = result["score"]["severity_counts"].get("warning", 0)
    if crit or warn:
        parts.append(f"rdq found {crit} critical and {warn} warning findings; see rdq:quality.")
    return " ".join(parts) or None


def _missing_data(result: dict) -> str | None:
    msgs = [f["message"] for f in result["findings"] if f["check"].startswith(("time.", "counts.", "video.", "values.non_finite"))]
    return " ".join(msgs[:5]) or None


def _distribution(ds: dict, root: Path) -> list[dict]:
    info = root / "meta" / "info.json"
    out = [
        {
            "@type": "cr:FileObject",
            "@id": "info",
            "name": "info",
            "contentUrl": "meta/info.json",
            "encodingFormat": "application/json",
            "sha256": hashlib.sha256(info.read_bytes()).hexdigest() if info.exists() else None,
        },
        {
            "@type": "cr:FileSet",
            "@id": "frames",
            "name": "frames",
            "includes": "data/**/*.parquet",
            "encodingFormat": "application/x-parquet",
        },
    ]
    if ds.get("cameras"):
        out.append(
            {
                "@type": "cr:FileSet",
                "@id": "videos",
                "name": "videos",
                "includes": "videos/**/*.mp4",
                "encodingFormat": "video/mp4",
            }
        )
    return out


_SCALAR_FIELDS = {"episode_index": "sc:Integer", "frame_index": "sc:Integer", "index": "sc:Integer",
                  "task_index": "sc:Integer", "timestamp": "sc:Float"}


def _record_sets(root: Path) -> list[dict]:
    """A RecordSet over the frame tables' scalar index columns, so Croissant loaders can
    enumerate frames. Vector features (action, state) are described in rdq:structure."""
    files = sorted((root / "data").glob("**/*.parquet")) if (root / "data").exists() else []
    if not files:
        return []
    try:
        import pyarrow.parquet as pq

        names = set(pq.read_schema(files[0]).names)
    except Exception:  # noqa: BLE001
        return []
    fields = [
        {
            "@type": "cr:Field",
            "@id": f"frame_records/{col}",
            "name": col,
            "dataType": dtype,
            "source": {"fileSet": {"@id": "frames"}, "extract": {"column": col}},
        }
        for col, dtype in _SCALAR_FIELDS.items()
        if col in names
    ]
    if not fields:
        return []
    return [{"@type": "cr:RecordSet", "@id": "frame_records", "name": "frame_records",
             "description": "One record per frame (index columns only).", "field": fields}]


def _context() -> dict:
    return {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "sc": "https://schema.org/",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "dct": "http://purl.org/dc/terms/",
        "prov": "http://www.w3.org/ns/prov#",
        "odrl": "http://www.w3.org/ns/odrl/2/",
        "rdq": RDQ_NS,
        "conformsTo": "dct:conformsTo",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "data": {"@id": "cr:data", "@type": "@json"},
        "equivalentProperty": "cr:equivalentProperty",
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "field": "cr:field",
        "fileObject": "cr:fileObject",
        "fileProperty": "cr:fileProperty",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "samplingRate": "cr:samplingRate",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
    }


# --------------------------------------------------------------------------- #
# SPDX 3.0 export
# --------------------------------------------------------------------------- #


def to_spdx(passport: dict) -> dict:
    """SPDX 3.0.1 JSON-LD with a DatasetPackage and declared/concluded licence relationships."""
    lic = passport.get("rdq:license", {})
    st = passport.get("rdq:structure", {})
    name = passport["name"]
    base = "https://spdx.org/spdxdocs/rdq/" + hashlib.sha256(json.dumps(passport.get("rdq:contentHash"), sort_keys=True).encode()).hexdigest()[:16]
    creation = {"type": "CreationInfo", "@id": "_:creationinfo", "specVersion": "3.0.1", "created": passport["rdq:attestation"]["generatedAt"],
                "createdBy": [f"{base}#agent-rdq"], "createdUsing": [f"{base}#tool-rdq"]}
    pkg_id = f"{base}#dataset"
    graph = [
        creation,
        {"type": "SoftwareAgent", "spdxId": f"{base}#agent-rdq", "creationInfo": "_:creationinfo", "name": "rdq"},
        {"type": "Tool", "spdxId": f"{base}#tool-rdq", "creationInfo": "_:creationinfo", "name": passport["rdq:attestation"]["generatedBy"]},
        {
            "type": "dataset_DatasetPackage",
            "spdxId": pkg_id,
            "creationInfo": "_:creationinfo",
            "name": name,
            "software_downloadLocation": passport.get("url"),
            "software_primaryPurpose": "data",
            "dataset_datasetType": ["sensor", "timeseries"] + (["video"] if st.get("cameras") else []),
            "dataset_datasetSize": passport.get("rdq:contentHash", {}).get("bytes"),
            "dataset_dataCollectionProcess": passport.get("rai:dataCollection"),
            "dataset_dataPreprocessing": passport.get("rai:dataPreprocessingProtocol"),
            "dataset_intendedUse": passport.get("rai:dataUseCases"),
            "dataset_knownBias": [passport["rai:dataBiases"]] if passport.get("rai:dataBiases") else None,
            "dataset_datasetNoise": (passport.get("rai:dataLimitations") or None),
            "verifiedUsing": [{"type": "Hash", "algorithm": "sha256", "hashValue": passport["rdq:contentHash"]["value"]}],
        },
    ]
    declared = lic.get("declaredSpdx")
    concluded = lic.get("concluded")

    def lic_node(expr: str, suffix: str) -> str:
        nid = f"{base}#license-{suffix}"
        graph.append({"type": "simplelicensing_LicenseExpression", "spdxId": nid, "creationInfo": "_:creationinfo",
                      "simplelicensing_licenseExpression": expr})
        return nid

    if declared:
        graph.append({"type": "Relationship", "spdxId": f"{base}#rel-declared", "creationInfo": "_:creationinfo",
                      "from": pkg_id, "relationshipType": "hasDeclaredLicense", "to": [lic_node(declared, "declared")]})
    graph.append({"type": "Relationship", "spdxId": f"{base}#rel-concluded", "creationInfo": "_:creationinfo",
                  "from": pkg_id, "relationshipType": "hasConcludedLicense",
                  "to": [lic_node(concluded or "NOASSERTION", "concluded")]})
    for i, d in enumerate(passport.get("prov:wasDerivedFrom") or []):
        if d.get("@id"):
            up_id = f"{base}#upstream-{i}"
            graph.append({"type": "dataset_DatasetPackage", "spdxId": up_id, "creationInfo": "_:creationinfo",
                          "name": d.get("name") or d["@id"], "software_downloadLocation": d["@id"], "software_primaryPurpose": "data",
                          "dataset_datasetType": ["sensor"]})
            graph.append({"type": "Relationship", "spdxId": f"{base}#rel-upstream-{i}", "creationInfo": "_:creationinfo",
                          "from": pkg_id, "relationshipType": "descendantOf", "to": [up_id]})
    for node in graph:
        for k in [k for k, v in node.items() if v in (None, [], "")]:
            del node[k]
    return {"@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld", "@graph": graph}


# --------------------------------------------------------------------------- #
# Datasheet (Markdown)
# --------------------------------------------------------------------------- #


def to_datasheet(passport: dict) -> str:
    lic = passport.get("rdq:license", {})
    st = passport.get("rdq:structure", {})
    q = passport.get("rdq:quality", {})
    h = passport.get("rdq:contentHash", {})
    L = [f"# Datasheet: {passport['name']}", ""]
    L.append(f"*Generated by {passport['rdq:attestation']['generatedBy']} on {passport['rdq:attestation']['generatedAt'][:10]}. "
             "Facts marked (data) were computed from the files; (owner) items are the owner's statements, recorded as given. Not legal advice.*")
    L += ["", "## Summary", ""]
    L.append(f"- **Description (owner):** {passport.get('description')}")
    L.append(f"- **Robot / format (data):** {st.get('robotType') or 'not declared'}, LeRobot {st.get('codebaseVersion')}, {st.get('fps')} fps")
    L.append(f"- **Size (data):** {st.get('episodes')} episodes, {st.get('frames')} frames"
             + (f", {st['hours']} h" if st.get("hours") is not None else "") + f", {st.get('tasks')} tasks, cameras: {', '.join(st.get('cameras') or []) or 'none'}")
    L.append(f"- **Content hash (data):** `{h.get('value', '')[:16]}…` over {h.get('files')} files in {', '.join(h.get('covers') or [])}")
    L += ["", "## Licence", ""]
    L.append(f"- **Declared on the card:** {lic.get('declared') or 'none'}")
    L.append(f"- **Status:** {lic.get('status')} ({lic.get('verdict')})")
    L.append(f"- **Concluded:** {lic.get('concluded')}")
    if lic.get("restrictions"):
        L.append(f"- **Restrictions carried from upstream:** {', '.join(lic['restrictions'])}")
    if lic.get("commercialTraining") is False:
        L.append("- **Commercial training:** not permitted by the upstream terms")
    for e in lic.get("evidence") or []:
        L.append(f"- Evidence: \"{e['statement']}\" ([source]({e['statementUrl']}), checked {e['checkedAt']}, matched by {e['match']})")
    for n in lic.get("notes") or []:
        L.append(f"- Note: {n}")
    L += ["", "## Provenance", ""]
    derived = passport.get("prov:wasDerivedFrom") or []
    if derived:
        for d in derived:
            L.append(f"- Derived from {d.get('name') or d.get('@id') or ''} {('(' + d['@id'] + ')') if d.get('@id') and d.get('name') else ''}"
                     f"{' — owner: ' + d['rdq:ownerStatement'] if d.get('rdq:ownerStatement') else ''}")
    else:
        L.append("- No upstream source recorded.")
    rai = [
        ("Collection method (owner)", "rai:dataCollection"),
        ("Collection type (owner)", "rai:dataCollectionType"),
        ("Collection timeframe (owner)", "rai:dataCollectionTimeframe"),
        ("Operators (owner)", "rai:annotatorDemographics"),
        ("Task annotation (owner)", "rai:dataAnnotationProtocol"),
        ("Personal data (owner)", "rai:personalSensitiveInformation"),
        ("Intended use (owner)", "rai:dataUseCases"),
        ("Known biases (owner)", "rai:dataBiases"),
        ("Maintenance (owner)", "rai:dataReleaseMaintenancePlan"),
    ]
    L += ["", "## Collection and use", ""]
    for label, key in rai:
        v = passport.get(key)
        L.append(f"- **{label}:** {_text(v) if v else '_not provided_'}")
    L += ["", "## Quality (data)", ""]
    L.append(f"- rdq score {q.get('score')}/100; {q.get('episodesUsable')}/{q.get('episodesTotal')} episodes pass all critical and warning checks.")
    for f in q.get("findings") or []:
        if f["severity"] in ("critical", "warning"):
            L.append(f"- `{f['check']}` ({f['severity']}): {f['episodes']} episodes")
    if passport.get("rdq:unanswered"):
        L += ["", "## Not yet answered by the owner", ""]
        L.append(", ".join(f"`{k}`" for k in passport["rdq:unanswered"]))
    return "\n".join(L) + "\n"
