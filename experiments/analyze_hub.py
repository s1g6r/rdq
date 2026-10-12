"""Turn a hub scan (and optional audit) into the numbers for the public report.

    python experiments/analyze_hub.py RESULTS_DIR [--out summary.json]

Facts are recomputed from the raw README/info.json files saved during the
scan, so improvements to the checks apply without re-downloading anything.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq.hub import info_facts  # noqa: E402
from rdq.provenance import analyse_card  # noqa: E402


def load_scan(results: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(results / "scan.jsonl")]
    raw = results / "raw"
    for r in rows:
        d = raw / r["id"].replace("/", "--")
        readme = (d / "README.md").read_text(errors="replace") if (d / "README.md").exists() else None
        info_text = (d / "info.json").read_text() if (d / "info.json").exists() else None
        r["_raw"] = d.exists()
        if d.exists():
            r["provenance"] = analyse_card(readme, card_license=r.get("card_license"), repo_id=r["id"]).to_dict()
            try:
                r["info"] = info_facts(json.loads(info_text)) if info_text else None
            except json.JSONDecodeError:
                r["info"] = None
    return rows


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def summarise_scan(rows: list[dict]) -> dict:
    n = len(rows)
    verdicts = collections.Counter(r["provenance"]["verdict"] for r in rows if r.get("provenance"))
    licences = collections.Counter((r["provenance"]["license"] or "none") for r in rows if r.get("provenance"))
    apache = [r for r in rows if (r.get("provenance") or {}).get("license") == "apache-2.0"]
    apache_default = [r for r in apache if r["provenance"]["default_license_unconfirmed"]]
    with_info = [r for r in rows if r.get("info")]
    versions = collections.Counter(str(r["info"].get("codebase_version")) for r in with_info)
    no_robot = [r for r in with_info if not r["info"].get("robot_type") or str(r["info"]["robot_type"]).lower() == "unknown"]
    mismatch = [r for r in with_info if r["info"].get("names_shape_mismatch")]
    placeholder = [r for r in with_info if r["info"].get("names_placeholder")]
    missing_names = [r for r in with_info if r["info"].get("names_missing")]
    no_action = [r for r in with_info if not r["info"].get("has_action")]
    downloads_total = sum(r.get("downloads") or 0 for r in rows)
    dl_default = sum(r.get("downloads") or 0 for r in rows if (r.get("provenance") or {}).get("default_license_unconfirmed"))
    top100 = rows[:100]
    authors = collections.Counter(r["id"].split("/")[0] for r in rows)
    by_author_default = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        a = r["id"].split("/")[0]
        by_author_default[a][1] += 1
        if (r.get("provenance") or {}).get("default_license_unconfirmed"):
            by_author_default[a][0] += 1
    distinct_authors = len(authors)
    authors_any_default = sum(1 for a, (d, t) in by_author_default.items() if d > 0)
    def dl(rs):
        return sum(r.get("downloads") or 0 for r in rs)

    def up(r):
        return (r.get("provenance") or {}).get("upstream") or {}

    known = [r for r in rows if up(r).get("upstream")]
    mismatch = [r for r in known if up(r).get("status") in ("relabelled", "upstream-conflict")]
    nc_conflict = [r for r in known if up(r).get("status") == "upstream-conflict" and up(r).get("severity") == "critical"]
    by_source = collections.Counter(u["name"] for r in known for u in up(r)["upstream"])
    upstream_summary = {
        "known_upstream_datasets": pct(len(known), n),
        "known_upstream_download_share": f"{100 * dl(known) / downloads_total:.1f}%" if downloads_total else "n/a",
        "declared_differs_from_upstream": pct(len(mismatch), len(known)),
        "declared_differs_download_share": f"{100 * dl(mismatch) / downloads_total:.1f}%" if downloads_total else "n/a",
        "status_counts": collections.Counter(up(r).get("status") for r in rows if up(r)).most_common(),
        "by_source": by_source.most_common(),
        "critical_conflicts": [
            {"id": r["id"], "downloads": r.get("downloads"), "declared": up(r)["declared"], "upstream": up(r)["concluded"],
             "evidence": sorted({u["evidence"] for u in up(r)["upstream"]})}
            for r in sorted(nc_conflict, key=lambda r: -(r.get("downloads") or 0))
        ],
        "top_relabelled": [
            {"id": r["id"], "downloads": r.get("downloads"), "declared": up(r)["declared"], "upstream": up(r)["concluded"]}
            for r in sorted(mismatch, key=lambda r: -(r.get("downloads") or 0))[:15]
        ],
    }
    return {
        "datasets_scanned": n,
        "upstream_licences": upstream_summary,
        "distinct_owners": distinct_authors,
        "top_owners": authors.most_common(10),
        "owners_with_any_untouched_default": pct(authors_any_default, distinct_authors),
        "raw_files_available": sum(1 for r in rows if r.get("_raw")),
        "licence_counts": licences.most_common(),
        "verdict_counts": verdicts.most_common(),
        "apache_total": len(apache),
        "apache_that_are_untouched_default": pct(len(apache_default), len(apache)),
        "all_untouched_default": pct(sum(1 for r in rows if (r.get("provenance") or {}).get("default_license_unconfirmed")), n),
        "top100_untouched_default": pct(
            sum(1 for r in top100 if (r.get("provenance") or {}).get("default_license_unconfirmed")), len(top100)
        ),
        "downloads_share_untouched_default": f"{100 * dl_default / downloads_total:.1f}%" if downloads_total else "n/a",
        "no_licence": pct(verdicts.get("no licence", 0), n),
        "noncommercial": pct(verdicts.get("non-commercial", 0), n),
        "restricted_source_mentions": [r["id"] for r in rows if (r.get("provenance") or {}).get("restricted_sources")],
        "no_info_json_at_root": pct(n - len(with_info), n),
        "codebase_versions": versions.most_common(),
        "robot_type_missing_or_unknown": pct(len(no_robot), len(with_info)),
        "names_true_mismatch": pct(len(mismatch), len(with_info)),
        "names_true_mismatch_examples": [r["id"] for r in mismatch[:15]],
        "names_single_placeholder": pct(len(placeholder), len(with_info)),
        "names_missing": pct(len(missing_names), len(with_info)),
        "no_action_feature": pct(len(no_action), len(with_info)),
    }


def summarise_audit(results: Path) -> dict | None:
    index = results / "audit_index.jsonl"
    if not index.exists():
        return None
    last: dict[str, dict] = {}
    for l in open(index):
        r = json.loads(l)
        last[r["id"]] = r  # a dataset retried after a network error keeps only its latest row
    rows = list(last.values())
    done = [r for r in rows if "score" in r]
    if not done:
        return {"audited": 0, "rows": len(rows)}
    check_counts = collections.Counter()
    finding_eps = collections.Counter()
    total_eps = sum(r["episodes"] for r in done)
    flagged_eps = 0
    for r in done:
        rep = json.loads((results / r["report"]).read_text())
        eps_by_check: dict[str, set] = {}
        for f in rep["findings"]:
            eps_by_check.setdefault(f["check"], set()).update(f["episodes"])
        for check, eps in eps_by_check.items():  # count each dataset once per check
            check_counts[check] += 1
            finding_eps[check] += len(eps)
        flagged_eps += rep["dataset"]["episodes"] - len(rep["score"]["usable_episodes"])
    scores = sorted(r["score"] for r in done)
    usable = sorted(r["usable_fraction"] for r in done)
    crit = sum(1 for r in done if r["severity_counts"].get("critical", 0) > 0)
    warn_or_crit = sum(
        1 for r in done if r["severity_counts"].get("critical", 0) + r["severity_counts"].get("warning", 0) > 0
    )
    return {
        "attempted": len(rows),
        "audited": len(done),
        "skipped_too_large": sum(1 for r in rows if "skipped" in r),
        "errors": sum(1 for r in rows if "error" in r),
        "error_examples": [r["error"][:120] for r in rows if "error" in r][:5],
        "episodes_audited": total_eps,
        "episodes_flagged_unusable": pct(flagged_eps, total_eps),
        "datasets_with_critical": pct(crit, len(done)),
        "datasets_with_warning_or_critical": pct(warn_or_crit, len(done)),
        "median_score": scores[len(scores) // 2],
        "median_usable_fraction": usable[len(usable) // 2],
        "datasets_by_check": check_counts.most_common(),
        "episodes_by_check": finding_eps.most_common(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--audit-dir", type=Path, help="folder with audit_index.jsonl (default: RESULTS_DIR)")
    a = p.parse_args()
    rows = load_scan(a.results)
    summary = {"scan": summarise_scan(rows), "audit": summarise_audit(a.audit_dir or a.results)}
    text = json.dumps(summary, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
