"""Roll per-episode findings and scores up to groups: collectors, operators, sessions or tasks.

Data vendors that collect through many contractors or partners need quality per
collector, not per dataset. A group can come from a column in the episodes metadata
(for example an operator id written at recording time), from the task, or from a CSV
that maps episode_index to a group name.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .dataset import Dataset


def episode_groups(ds: Dataset, group_by: str | None = None, mapping_csv: str | Path | None = None) -> dict[int, str]:
    if mapping_csv:
        out = {}
        with open(mapping_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                keys = {k.lower(): v for k, v in row.items()}
                ep = keys.get("episode_index") or keys.get("episode")
                grp = keys.get("group") or keys.get("operator") or keys.get("collector")
                if ep is not None and grp is not None:
                    out[int(ep)] = str(grp)
        return out
    if group_by in (None, "task"):
        return {i: ds.tasks.get(int(ep.task_index[0]), "unknown") if ep.task_index is not None and len(ep.task_index) else "unknown"
                for i, ep in ds.episodes.items()}
    out = {}
    for i, ep in ds.episodes.items():
        v = (ep.meta or {}).get(group_by)
        if isinstance(v, (list, tuple, np.ndarray)):
            v = v[0] if len(v) else None
        out[i] = str(v) if v is not None else "unknown"
    return out


def scorecard(result: dict, scores: dict | None, groups: dict[int, str]) -> list[dict]:
    """One row per group: episodes, usable share, top findings and median scores."""
    usable = set(result["score"]["usable_episodes"])
    flagged = {int(k): v for k, v in result["score"]["flagged_episodes"].items()}
    eps = scores["episodes"] if scores else {}
    by_group: dict[str, list[int]] = defaultdict(list)
    for i in sorted(groups):
        by_group[groups[i]].append(i)
    rows = []
    for g, idx in by_group.items():
        issues = Counter(c for i in idx for c in flagged.get(i, []))
        comp = [eps[i]["composite"] for i in idx if i in eps and eps[i].get("composite") is not None]
        hfr = [eps[i]["hfr"] for i in idx if i in eps and eps[i].get("hfr") is not None and np.isfinite(eps[i]["hfr"])]
        idle = [eps[i]["idle_frac"] for i in idx if i in eps]
        rows.append(
            {
                "group": g,
                "episodes": len(idx),
                "usable": sum(1 for i in idx if i in usable),
                "usable_fraction": round(sum(1 for i in idx if i in usable) / len(idx), 3),
                "top_issues": issues.most_common(3),
                "median_composite": round(float(np.median(comp)), 3) if comp else None,
                "median_hfr": round(float(np.median(hfr)), 3) if hfr else None,
                "mean_idle_fraction": round(float(np.mean(idle)), 3) if idle else None,
            }
        )
    rows.sort(key=lambda r: (r["usable_fraction"], r["median_composite"] if r["median_composite"] is not None else 1))
    return rows


def format_scorecard(rows: list[dict]) -> str:
    lines = [f"{'group':<28} {'episodes':>8} {'usable':>8} {'composite':>9} {'idle':>6}  top issues"]
    for r in rows:
        comp = f"{r['median_composite']:.2f}" if r["median_composite"] is not None else "-"
        idle = f"{r['mean_idle_fraction']:.0%}" if r["mean_idle_fraction"] is not None else "-"
        issues = ", ".join(f"{c} ({n})" for c, n in r["top_issues"]) or "none"
        lines.append(f"{r['group'][:28]:<28} {r['episodes']:>8} {r['usable_fraction']:>8.0%} {comp:>9} {idle:>6}  {issues}")
    return "\n".join(lines)
