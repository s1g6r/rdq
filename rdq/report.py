"""Build the audit result object and render it as JSON, terminal text or HTML."""

from __future__ import annotations

import html
import re
import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .checks import CRITICAL, INFO, WARNING, Finding, Thresholds, run_checks
from .dataset import Dataset, load_dataset
from .provenance import ProvenanceResult, analyse_card
from .scoring import Score, score

SEVERITY_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}
REPORT_VERSION = 1  # bump only when a documented field (docs/json-report.md) changes name or meaning


def audit(
    path: str | Path,
    max_frames: int | None = None,
    readme: str | None = None,
    repo_id: str | None = None,
    video: bool = False,
    video_full: bool = False,
    expected: str | Path | None = None,
) -> dict:
    """Run every check on a local dataset folder and return a JSON-ready dict.

    With video=True, a few frames per episode and camera are decoded as well (needs PyAV). With
    video_full=True, every frame is decoded and episodes that repeat another episode's footage are found.
    Findings the owner lists as expected (``expected`` file, else ``meta/rdq_expected.yaml`` in the
    dataset) are reported as info with the owner's reason and do not drop episodes.
    """
    from .expected import apply as apply_expected, find_in_dataset, load as load_expected

    path = Path(path)
    ds = load_dataset(path, max_frames=max_frames)
    findings = run_checks(ds, Thresholds())
    if video_full:
        from .video import check_videos_full

        findings += check_videos_full(ds)
    elif video:
        from .video import check_videos

        findings += check_videos(ds)
    exp_path = Path(expected) if expected else find_in_dataset(path)
    entries = load_expected(exp_path) if exp_path else []
    findings = apply_expected(findings, entries)
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.check))
    sc = score(findings, sorted(ds.episodes))
    if readme is None and (path / "README.md").exists():
        readme = (path / "README.md").read_text(errors="replace")
    prov = analyse_card(readme, repo_id=repo_id) if readme is not None else None
    result = build_result(ds, findings, sc, prov)
    if exp_path:
        result["expected"] = {
            "file": str(exp_path) if expected else str(exp_path.relative_to(path)),
            "entries": len(entries),
            "findings_marked": sum(1 for f in findings if f.details.get("expected")),
        }
    return result


def build_result(ds: Dataset, findings: list[Finding], sc: Score, prov: ProvenanceResult | None) -> dict:
    lengths = [ep.length for ep in ds.episodes.values()]
    return {
        "report_version": REPORT_VERSION,
        "tool": {"name": "rdq", "version": __version__},
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": str(ds.root),
            "codebase_version": ds.version,
            "robot_type": ds.info.get("robot_type"),
            "fps": ds.fps,
            "episodes": len(ds.episodes),
            "frames": int(sum(lengths)),
            "median_episode_frames": int(sorted(lengths)[len(lengths) // 2]) if lengths else 0,
            "numeric_features": ds.numeric_keys,
            "cameras": ds.video_keys,
            "tasks": len(ds.tasks),
            "features_not_checked": dict(getattr(ds, "skipped_features", {}) or {}),
        },
        "not_checked": list(dict.fromkeys(getattr(ds, "not_checked", []) or [])),
        "score": sc.to_dict(),
        "findings": [f.to_dict() for f in findings],
        "provenance": prov.to_dict() if prov else None,
        "thresholds": Thresholds().__dict__,
    }


def write_json(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2, default=_json_default))


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


def write_filter(result: dict, path: str | Path) -> None:
    """Episodes that passed, in the form LeRobot's `episodes=` argument accepts."""
    Path(path).write_text(json.dumps({"episodes": result["score"]["usable_episodes"]}))


# --------------------------------------------------------------------------- #
# Terminal output
# --------------------------------------------------------------------------- #

_COLORS = {CRITICAL: "\033[31m", WARNING: "\033[33m", INFO: "\033[36m"}
_RESET, _BOLD, _DIM = "\033[0m", "\033[1m", "\033[2m"


_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _clean(text) -> str:
    """Strip control characters (e.g. ANSI escapes planted in task strings) before printing."""
    return _CONTROL.sub("", str(text))


def _fmt_fps(fps) -> str:
    return f"{fps:g}" if isinstance(fps, (int, float)) else "?"


def format_terminal(result: dict, color: bool = True) -> str:
    def c(text, code):
        return f"{code}{text}{_RESET}" if color else text

    d, s = result["dataset"], result["score"]
    lines = [
        c(_clean(d["path"]), _BOLD),
        f"  {d['episodes']} episodes · {d['frames']:,} frames · {_fmt_fps(d['fps'])} fps · "
        f"robot: {d['robot_type'] or 'unknown'} · LeRobot v{d['codebase_version']}",
        "",
        c(f"  Score {s['dataset_score']}/100", _BOLD)
        + f"   usable episodes {len(s['usable_episodes'])}/{d['episodes']} ({s['usable_fraction']:.0%})",
        "",
    ]
    exp = result.get("expected")
    if exp:
        lines += [c(f"  {exp['findings_marked']} finding(s) marked as expected by the dataset's owner "
                    f"({_clean(exp['file'])})", _DIM), ""]
    if not result["findings"]:
        lines.append("  No problems found by the checks that ran.")
    for f in result["findings"]:
        tag = c(f"{f['severity'].upper():8}", _COLORS[f["severity"]])
        eps = f["episodes"]
        where = ""
        if eps:
            shown = ", ".join(map(str, eps[:8])) + (f" … (+{len(eps) - 8})" if len(eps) > 8 else "")
            where = c(f"\n            episodes {shown}", _DIM)
        lines.append(f"  {tag} {_clean(f['message'])}{where}")
    if result.get("not_checked"):
        lines += ["", c("  Not checked", _BOLD)] + [f"  · {_clean(n)}" for n in result["not_checked"]]
    prov = result.get("provenance")
    if prov:
        tag = {"go": "OK", "check": "CHECK", "stop": "STOP"}.get(prov.get("bottom_level"), "CHECK")
        col = {"OK": _COLORS.get("info", ""), "CHECK": _COLORS.get("warning", ""), "STOP": _COLORS.get("critical", "")}[tag]
        lines += ["", c("  Licence & provenance", _BOLD)]
        if prov.get("bottom_line"):
            lines.append(f"  {c(tag, col)}  {_clean(prov['bottom_line'])}")
        lines.append(f"  licence: {prov['license'] or 'none'} → {prov['verdict']}")
        if prov.get("card_fix"):
            lines.append("  fix for the uploader (card YAML header):")
            lines += [f"    {_clean(l)}" for l in prov["card_fix"].splitlines()]
        up = prov.get("upstream") or {}
        if up.get("upstream"):
            terms = f" ({up.get('concluded')})" if up.get("concluded") else ""
            lines.append(f"  upstream: {', '.join(u['name'] for u in up['upstream'])}{terms} → {up.get('status')}")
        for note in prov["notes"]:
            lines.append(f"  · {_clean(note)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Markdown (GitHub job summaries, pull-request comments)
# --------------------------------------------------------------------------- #


def _md(text) -> str:
    """Plain text safe inside a Markdown table cell or list item."""
    return _clean(text).replace("|", "\\|").replace("\n", " ").replace("<", "&lt;")


def format_markdown(result: dict) -> str:
    d, s = result["dataset"], result["score"]
    counts = s["severity_counts"]
    sev_icon = {CRITICAL: "🔴", WARNING: "🟡", INFO: "⚪"}
    lines = [
        f"### rdq: {_md(d['path'])}",
        "",
        f"**Score {s['dataset_score']}/100** · usable episodes {len(s['usable_episodes'])}/{d['episodes']} "
        f"({s['usable_fraction']:.0%}) · {counts.get(CRITICAL, 0)} critical, {counts.get(WARNING, 0)} warning, "
        f"{counts.get(INFO, 0)} info",
        "",
    ]
    exp = result.get("expected")
    if exp:
        lines += [f"{exp['findings_marked']} finding(s) marked as expected by the dataset's owner ({_md(exp['file'])}).", ""]
    if result["findings"]:
        lines += ["| | Check | Finding | Episodes |", "|---|---|---|---|"]
        for f in result["findings"]:
            eps = f["episodes"]
            shown = ", ".join(map(str, eps[:10])) + (f" … (+{len(eps) - 10})" if len(eps) > 10 else "")
            lines.append(f"| {sev_icon.get(f['severity'], '')} | `{_md(f['check'])}` | {_md(f['message'])} | {shown} |")
    else:
        lines.append("No problems found by the checks that ran.")
    prov = result.get("provenance")
    if prov:
        tag = {"go": "OK", "check": "CHECK", "stop": "STOP"}.get(prov.get("bottom_level"), "CHECK")
        lines += ["", f"**Licence:** {_md(prov['license'] or 'none')} → {_md(prov['verdict'])} ({tag})"]
        if prov.get("bottom_line"):
            lines.append(f"{_md(prov['bottom_line'])}")
        if prov.get("card_fix"):
            lines += ["", "Suggested card fix (YAML header):", "", "```yaml", prov["card_fix"].strip(), "```"]
    if result.get("not_checked"):
        lines += ["", "<details><summary>Not checked</summary>", ""] + [f"- {_md(n)}" for n in result["not_checked"]] + ["", "</details>"]
    lines += ["", f"<sub>rdq {_md(result['tool']['version'])}</sub>"]
    return "\n".join(lines) + "\n"


def write_markdown(result: dict, path: str | Path) -> None:
    """Write the Markdown report; GitHub's job-summary file is appended to, not replaced."""
    import os

    append = os.environ.get("GITHUB_STEP_SUMMARY") and Path(path).resolve() == Path(os.environ["GITHUB_STEP_SUMMARY"]).resolve()
    with open(path, "a" if append else "w") as fh:
        fh.write(format_markdown(result))


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #


def _esc(x) -> str:
    return html.escape(str(x))


def render_html(result: dict) -> str:
    d, s = result["dataset"], result["score"]
    counts = s["severity_counts"]
    total = d["episodes"] or 1
    usable_pct = s["usable_fraction"] * 100
    name = Path(d["path"]).name or d["path"]

    rows = []
    for f in result["findings"]:
        eps = f["episodes"]
        eps_html = (
            f'<span class="eps">{len(eps)} episode{"s" if len(eps) != 1 else ""}: '
            + _esc(", ".join(map(str, eps[:30])))
            + ("…" if len(eps) > 30 else "")
            + "</span>"
            if eps
            else '<span class="eps">whole dataset</span>'
        )
        rows.append(
            f'<li class="finding {f["severity"]}"><span class="sev">{f["severity"]}</span>'
            f'<div><p>{_esc(f["message"])}</p>{eps_html}<code>{_esc(f["check"])}</code></div></li>'
        )
    findings_html = "\n".join(rows) or '<li class="finding ok"><div><p>No problems found by the checks that ran.</p></div></li>'
    not_checked_html = ""
    if result.get("not_checked"):
        not_checked_html = "<h3>Not checked</h3><ul>" + "".join(f"<li>{_esc(n)}</li>" for n in result["not_checked"]) + "</ul>"

    # One cell per episode, shaded by its score.
    cells = []
    flagged = s["flagged_episodes"]
    severities: dict[str, set] = {}
    for f in result["findings"]:
        for e in f["episodes"]:
            severities.setdefault(str(e), set()).add(f["severity"])
    for ep, val in s["episode_scores"].items():
        tip = f"episode {ep}: {val}/100" + (f" — {', '.join(flagged[ep])}" if ep in flagged else "")
        sev = severities.get(ep, set())
        band = "bad" if "critical" in sev else "poor" if "warning" in sev else "fair" if sev else "good"
        cells.append(f'<span class="cell {band}" title="{_esc(tip)}"></span>')

    prov = result.get("provenance")
    prov_html = ""
    if prov:
        notes = "".join(f"<li>{_esc(n)}</li>" for n in prov["notes"]) or "<li>No issues found in the dataset card.</li>"
        up = prov.get("upstream") or {}
        for u in up.get("upstream") or []:
            notes += (f'<li>Upstream evidence: “{_esc(u["statement"])}” (<a href="{_esc(u["statement_url"])}">{_esc(u["name"])}</a>, '
                      f'checked {_esc(u["checked"])}, matched by {_esc(u["evidence"])})</li>')
        prov_html = f"""
<section class="card">
  <h2>Licence &amp; provenance</h2>
  <p class="bottom"><b>{_esc({"go": "OK", "check": "Check first", "stop": "Stop"}.get(prov.get("bottom_level"), "Check first"))}:</b> {_esc(prov.get("bottom_line", ""))}</p>
  <div class="prov"><span class="pill {('ok' if prov['verdict'] == 'documented' and (prov.get('upstream') or {}).get('severity', 'ok') in ('ok', 'info') else 'flag')}">{_esc(prov['verdict'])}</span>
  <span>licence: <b>{_esc(prov['license'] or 'none')}</b></span></div>
  <ul class="notes">{notes}</ul>{('<p><b>Fix for the uploader</b> (paste into the card&#39;s YAML header):</p><pre>' + _esc(prov['card_fix']) + '</pre>') if prov.get('card_fix') else ''}
</section>"""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data audit · {_esc(name)}</title>
<style>
:root {{ --bg:#f6f5f1; --panel:#fff; --ink:#1b1d1f; --muted:#666b70; --line:#e3e1db;
  --crit:#c2362b; --warn:#b7791f; --info:#2f6f8f; --good:#2e7d4f; --fair:#9bbf6a; --poor:#e0a33a; --bad:#c2362b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141516; --panel:#1d1f21; --ink:#ececea; --muted:#9a9fa4; --line:#2e3134; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:32px 16px 64px }}
header h1 {{ font-size:22px; margin:0 0 4px; word-break:break-all }}
header p {{ color:var(--muted); margin:0 }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:24px 0 }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px }}
.stat b {{ display:block; font-size:28px; font-variant-numeric:tabular-nums; letter-spacing:-.02em }}
.stat span {{ color:var(--muted); font-size:13px }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px 20px; margin-top:16px }}
h2 {{ font-size:15px; margin:0 0 12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted) }}
.cells {{ display:flex; flex-wrap:wrap; gap:3px }}
.cell {{ width:12px; height:12px; border-radius:2px }}
.cell.good{{background:var(--good)}} .cell.fair{{background:var(--fair)}} .cell.poor{{background:var(--poor)}} .cell.bad{{background:var(--bad)}}
.legend {{ color:var(--muted); font-size:12px; margin-top:8px }}
ul.findings {{ list-style:none; margin:0; padding:0 }}
.finding {{ display:flex; gap:12px; padding:12px 0; border-top:1px solid var(--line) }}
.finding:first-child {{ border-top:0 }}
.finding p {{ margin:0 0 4px }}
.sev {{ flex:0 0 72px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; padding-top:3px }}
.critical .sev{{color:var(--crit)}} .warning .sev{{color:var(--warn)}} .info .sev{{color:var(--info)}}
.eps {{ display:block; color:var(--muted); font-size:13px; word-break:break-word }}
code {{ font-size:12px; color:var(--muted) }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:13px; font-weight:600; margin-right:10px }}
.pill.ok {{ background:color-mix(in srgb,var(--good) 18%,transparent); color:var(--good) }}
.pill.flag {{ background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn) }}
.notes {{ margin:10px 0 0; padding-left:18px; color:var(--muted) }}
footer {{ color:var(--muted); font-size:12px; margin-top:24px }}
</style></head><body><main>
<header><h1>{_esc(name)}</h1>
<p>{d['episodes']} episodes · {d['frames']:,} frames · {_fmt_fps(d['fps'])} fps · robot {_esc(d['robot_type'] or 'unknown')} · LeRobot v{_esc(d['codebase_version'])}</p></header>
<div class="grid">
  <div class="stat"><b>{s['dataset_score']}</b><span>quality score / 100</span></div>
  <div class="stat"><b>{usable_pct:.0f}%</b><span>usable episodes ({len(s['usable_episodes'])} of {total})</span></div>
  <div class="stat"><b style="color:var(--crit)">{counts.get('critical', 0)}</b><span>critical findings</span></div>
  <div class="stat"><b style="color:var(--warn)">{counts.get('warning', 0)}</b><span>warnings</span></div>
</div>
<section class="card"><h2>Episodes</h2><div class="cells">{''.join(cells)}</div>
<div class="legend">Each square is one episode. Green: clean · light green: notes only · amber: warning · red: critical. Hover for details.</div></section>
<section class="card"><h2>Findings</h2><ul class="findings">{findings_html}</ul>{not_checked_html}</section>
{prov_html}
<footer>Generated by rdq {_esc(result['tool']['version'])} on {_esc(result['audited_at'])}. Findings are rule-based flags for human review; thresholds are listed in the JSON report.</footer>
</main></body></html>"""


def write_html(result: dict, path: str | Path) -> None:
    Path(path).write_text(render_html(result))
