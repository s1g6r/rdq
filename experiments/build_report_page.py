"""Build the public "State of Open Robot Data" page from analysis outputs.

    python experiments/build_report_page.py summary.json toy_results.json OUT.html
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path


def esc(x) -> str:
    return html.escape(str(x))


def frac(s: str) -> tuple[int, int, float]:
    """Parse '45/100 (45.0%)' into (45, 100, 45.0)."""
    m = re.match(r"(\d+)/(\d+)", s)
    n, d = int(m.group(1)), int(m.group(2))
    return n, d, 100 * n / d if d else 0.0


def waffle(n_hit: int, total: int = 100) -> str:
    cells = []
    for i in range(total):
        cls = "hit" if i < n_hit else ""
        cells.append(f'<span class="w {cls}"></span>')
    return f'<div class="waffle" role="img" aria-label="{n_hit} of {total}">{"".join(cells)}</div>'


def hbars(rows: list[tuple[str, float, str]], unit: str = "%", max_value: float | None = None, cls: str = "") -> str:
    """Horizontal bars. rows = (label, value, css-class)."""
    top = max_value or max(v for _, v, _ in rows) or 1
    out = []
    for label, value, c in rows:
        width = 100 * value / top
        if unit == "%":
            shown = f"{value:.1f}%" if 0 < value < 1 else f"{value:.0f}%"
        else:
            shown = f"{value:g}{unit}"
        out.append(
            f'<div class="bar-row"><span class="bar-label">{esc(label)}</span>'
            f'<span class="bar-track"><span class="bar {c}" style="width:{width:.1f}%"></span></span>'
            f'<span class="bar-value">{esc(shown)}</span></div>'
        )
    return f'<div class="bars {cls}">{"".join(out)}</div>'


def toy_chart(by_rate: list[dict]) -> str:
    """Grouped bars: raw / random / filtered success per corruption rate, as SVG."""
    W, H, left, bottom, top = 640, 260, 44, 40, 14
    plot_w, plot_h = W - left - 10, H - bottom - top
    groups = len(by_rate)
    gw = plot_w / groups
    bw = gw / 4.2
    series = [("raw", "s-raw", "All episodes"), ("random", "s-rand", "Random subset, same size"), ("filtered", "s-filt", "Episodes rdq kept")]
    parts = []
    for t in (0, 25, 50, 75, 100):
        y = top + plot_h * (1 - t / 100)
        parts.append(f'<line x1="{left}" x2="{W - 10}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{t}%</text>')
    for gi, row in enumerate(by_rate):
        x0 = left + gi * gw + (gw - 3 * bw) / 2
        for si, (key, c, _) in enumerate(series):
            v = row[key] * 100
            h = plot_h * v / 100
            x = x0 + si * bw
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 3:.1f}" height="{h:.1f}" class="{c}"><title>{v:.0f}%</title></rect>')
        cx = left + gi * gw + gw / 2
        parts.append(f'<text x="{cx:.1f}" y="{H - bottom + 18}" class="tick" text-anchor="middle">{row["corruption_rate"] * 100:.0f}%</text>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{H - 4}" class="axis" text-anchor="middle">share of episodes with recording faults</text>')
    legend = "".join(f'<span class="key"><i class="{c}"></i>{esc(label)}</span>' for _, c, label in series)
    return (
        f'<div class="chart-wrap"><svg viewBox="0 0 {W} {H}" class="chart" role="img" '
        f'aria-label="Policy success rate by training data">{"".join(parts)}</svg></div>'
        f'<div class="legend">{legend}</div>'
    )


CHECK_LABELS = {
    "stats.stale": "stats.json does not match the data",
    "episodes.trajectory_outlier": "episodes that move unlike the rest",
    "info.no_robot_type": "no robot type recorded",
    "episodes.idle": "episodes where the robot barely moves",
    "episodes.idle_padding": "long idle stretches at start or end",
    "episodes.too_short": "aborted, very short episodes",
    "episodes.too_long": "runaway, very long episodes",
    "episodes.spikes": "one-frame sensor or teleop glitches",
    "time.dropped_frames": "dropped frames",
    "time.fps_mismatch": "recorded at a different fps than declared",
    "time.non_monotonic": "timestamps that go backwards",
    "sync.action_state_lag": "action stream shifted against observations",
    "sync.action_state_uncorrelated": "state does not follow the actions",
    "tasks.placeholder": "placeholder task descriptions",
    "tasks.too_short": "one- or two-word task descriptions",
    "values.dead_dimension": "a channel that never changes",
    "values.non_finite": "NaN or infinite values",
    "episodes.duplicate": "duplicated episodes",
    "video.span_mismatch": "video segment longer or shorter than the data",
    "counts.episodes": "episode count disagrees with info.json",
    "counts.frames": "frame count disagrees with info.json",
    "counts.length_mismatch": "episode lengths disagree with metadata",
    "counts.episode_gaps": "missing episode indices",
    "schema.names_shape_mismatch": "channel names do not match the shape",
    "schema.unnamed_channels": "channels without individual names",
    "schema.action_state_dims_differ": "action and state sizes differ",
    "stats.missing": "no stats.json",
    "schema.no_state": "no observation.state",
}


def build(summary: dict, toy: dict, bench: dict | None = None) -> str:
    scan, audit = summary["scan"], summary.get("audit")
    n = scan["datasets_scanned"]
    top_hit, top_tot, top_pct = frac(scan["top100_untouched_default"])
    all_hit, _, all_pct = frac(scan["all_untouched_default"])
    ap_hit, ap_tot, ap_pct = frac(scan["apache_that_are_untouched_default"])
    own_hit, own_tot, own_pct = frac(scan["owners_with_any_untouched_default"])
    noinfo = frac(scan["no_info_json_at_root"])
    robot = frac(scan["robot_type_missing_or_unknown"])
    names_mm = frac(scan["names_true_mismatch"])
    names_ph = frac(scan["names_single_placeholder"])
    names_none = frac(scan["names_missing"])
    no_action = frac(scan["no_action_feature"])
    dl_share = f'{float(scan["downloads_share_untouched_default"].rstrip("%")):.0f}%'
    top_owner, top_owner_n = scan["top_owners"][0]

    verdict_names = {
        "documented": ("Card written by the owner", "ok"),
        "default licence, unconfirmed": ("Untouched template + default apache-2.0", "hl"),
        "no licence": ("No licence at all", "crit"),
        "licensed, no provenance": ("Licensed, template card", "warn"),
        "custom licence": ("Custom or other licence", "neutral"),
        "non-commercial": ("Non-commercial licence", "neutral"),
        "restricted source": ("Mentions a restricted source", "warn"),
        "licence conflicts with upstream": ("Licence conflicts with its source", "crit"),
        "relabelled (attribution owed)": ("CC BY source relabelled permissive", "warn"),
        "differs from upstream": ("Differs from its source's terms", "warn"),
        "no upstream data licence": ("Source states no data licence", "warn"),
        "no-derivatives licence": ("No-derivatives licence", "neutral"),
    }
    verdict_rows = [
        (verdict_names.get(v, (v, "neutral"))[0], 100 * c / n, verdict_names.get(v, (v, "neutral"))[1])
        for v, c in scan["verdict_counts"]
    ]

    meta_rows = [
        ("no meta/info.json at repo root", noinfo[2], "neutral"),
        ("robot type missing or 'unknown'", robot[2], "warn"),
        ("no 'action' feature", no_action[2], "warn"),
        ("joint names missing", names_none[2], "neutral"),
        ("one generic name for all channels", names_ph[2], "neutral"),
        ("names contradict the declared shape", names_mm[2], "crit"),
    ]

    upstream_html = ""
    up = scan.get("upstream_licences")
    if up:
        k_hit, k_tot, k_pct = frac(up["known_upstream_datasets"])
        d_hit, d_tot, d_pct = frac(up["declared_differs_from_upstream"])
        src_rows = [(name.split(" (")[0], 100 * cnt / k_hit, "neutral") for name, cnt in up["by_source"][:10]]
        status_names = {"relabelled": ("CC BY upstream, relabelled permissive", "hl"), "consistent": ("Matches upstream", "ok"),
                        "upstream-conflict": ("Conflicts with upstream terms", "crit"), "missing": ("No licence declared", "warn"),
                        "review": ("Custom licence, needs review", "warn"), "restricted-redistribution": ("Upstream forbids sharing copies", "warn"),
                        "no-upstream-grant": ("Original states no data licence", "warn"),
                        "stricter-than-upstream": ("Stricter than upstream", "ok")}
        st_rows = [(status_names.get(k, (k, "neutral"))[0], 100 * c / k_hit, status_names.get(k, (k, "neutral"))[1])
                   for k, c in up["status_counts"] if k in status_names and k != "missing"]
        n_crit = up.get("critical_conflicts") or 0
        if isinstance(n_crit, list):
            n_crit = len(n_crit)
        pub_hit = frac(up["publisher_uploads"])[0] if up.get("publisher_uploads") else 0
        ev = dict(up.get("declared_differs_by_strongest_evidence") or [])
        by_name = ev.get("repo-id", 0)
        upstream_html = f"""
<section id="upstream" class="sec">
  <div class="col">
    <p class="eyebrow">Licence &amp; provenance</p>
    <h2>Conversions often drop their source's licence</h2>
    <p>We checked the terms of 65 public robot datasets against their primary sources (26 with an explicit data licence from the original authors). {k_hit:,} of the {n:,} Hub datasets (<b>{up['known_upstream_download_share']} of all downloads</b>) can be matched to one of them, by the card or by repository name. We leave out a further {pub_hit:,} uploads by an original publisher's own organisation, whose tag is theirs to choose.</p>
    <p><b>{d_hit} of those {d_tot:,} declare a licence that differs from the original's terms: {up['declared_differs_download_share']} of all downloads</b>, or {up['declared_differs_high_confidence_download_share']} counting only high-confidence sources. Most are CC BY 4.0 datasets relabelled <code>apache-2.0</code> or MIT, which drops the attribution and change-notice duties on paper but not in law. {by_name} of the {d_hit} are matched by repository name alone (for example <code>droid_lerobot</code> or <code>bridge_v2_lerobot</code>). For the 17 most-downloaded of these we compared robot type and episode count with the original release: the robot matches in all 17, 13 are full copies, 3 are subsets and 1 is a small sample. One of them is a sample of a non-commercial, share-alike dataset, tagged apache-2.0. Another {frac(up['no_upstream_grant'])[0]} datasets ({up['no_upstream_grant_download_share']} of downloads) carry a licence for data whose original release never stated one: popular simulation demos among them.</p>
  </div>
  <figure class="fig">
    <figcaption>Declared licence vs the original terms · {k_hit:,} datasets traced to a known original</figcaption>
    {hbars(st_rows)}
  </figure>
  <figure class="fig">
    <figcaption>Most common originals · share of the {k_hit:,}</figcaption>
    {hbars(src_rows)}
  </figure>
  <p class="note">A tag on a mirror cannot grant rights its uploader does not hold. Matches are made from the card's <code>source_datasets</code>, a link to the original project in a derivation context (a "Homepage" field, "converted from"), or the repository name; a link or mention that only cites a project does not count. No single signal is treated as proof: a finding is only marked critical when the card declares its source or two signals agree. An umbrella statement (such as the Open X-Embodiment README's CC BY 4.0) counts as medium confidence. When the uploader is the original rights holder, a different tag may be a deliberate relicence. Data recorded <em>on</em> a robot brand (for example an AgiBot G1) is not counted as that brand's dataset. Not legal advice.</p>
</section>"""

    bench_html = ""
    if bench:
        bs = bench["summary"]
        scen_titles = {"structural": "30% recording faults", "skill": "40% imperfect operators", "mixed": "Faults and imperfect operators"}
        show = [("all data", "All data", "neutral"), ("random", "Random subset", "neutral"), ("keep shortest", "Keep shortest", "neutral"),
                ("psd (Sojib et al.)", "Spectral metric (PSD)", "neutral"), ("deminf-lite", "DemInf-lite", "warn"),
                ("composite", "rdq composite score", "warn"), ("rules (rdq check)", "rdq rules", "hl"), ("oracle", "Oracle", "ok")]
        figs = []
        for key in ("structural", "skill", "mixed"):
            if key not in bs:
                continue
            rows = [(label, 100 * bs[key]["success"][m]["success"], cls) for m, label, cls in show if m in bs[key]["success"]]
            figs.append(f'<figure class="fig"><figcaption>{esc(scen_titles[key])} · policy success, {bs[key]["seeds"]} seeds × 400 rollouts</figcaption>{hbars(rows, max_value=100)}</figure>')
        st, sk = bs.get("structural", {}), bs.get("skill", {})
        bench_html = f"""
<section id="curation" class="sec">
  <div class="col">
    <p class="eyebrow">Which episodes to keep</p>
    <h2>Broken recordings cost far more than imperfect operators</h2>
    <p>We separated two kinds of bad data. <b>Recording faults</b> are a frozen log, actions written late or a teleop glitch. <b>Imperfect operators</b> are shaky hands, hesitation, detours and attempts that stop short. We then compared ways of choosing which episodes to train on: random subsets, keeping the shortest episodes, a published training-free spectral metric, a reimplementation of DemInf, rdq's rules and an oracle that knows the truth.</p>
  </div>
  {"".join(figs)}
  <div class="callout"><p>With 30% of episodes broken by recording faults, training on everything succeeded {100 * st['success']['all data']['success']:.0f}% of the time. Training on what rdq's rules kept succeeded <mark>{100 * st['success']['rules (rdq check)']['success']:.0f}%</mark>, the same as the oracle. Length and smoothness metrics did no better than random.</p></div>
  <div class="col"><p class="note">Imperfect operators hurt much less: {100 * sk['success']['all data']['success']:.0f}% success with all the data. No training-free score reliably beat using everything; keeping the shortest episodes did as well as any. So rdq drops broken recordings but only <em>ranks</em> episodes by skill signals for a human to review. This is a controlled 2D task. On real multi-operator data (RoboMimic, keeping half of 300 demonstrations), rdq's skill scores beat random (operator tier 2.16 vs 2.00 on Can) but not published methods (2.32–2.37) or simply keeping the shortest demonstrations (2.53). On real Push-T demonstrations rdq caught all 62 injected recording faults and dropped none of the clean episodes.</p></div>
</section>"""

    audit_html = ""
    if audit and audit.get("audited"):
        a_n = audit["audited"]
        by_check = dict(audit["datasets_by_check"])
        # How much each kind of finding can be trusted on real data.
        groups = [
            ("Exact or arithmetic checks", "high", [
                "episodes.duplicate", "tasks.placeholder", "counts.length_mismatch",
                "counts.episodes", "counts.frames", "time.non_monotonic", "values.non_finite",
                "schema.names_shape_mismatch", "stats.stale",
            ]),
            ("Rule-based checks, likely real", "medium", [
                "episodes.idle", "episodes.too_short", "episodes.idle_padding", "sync.action_state_lag",
            ]),
            ("Flagged for review, not confirmed", "review", ["episodes.spikes", "video.span_mismatch", "sync.action_state_uncorrelated"]),
        ]
        group_html = []
        for title, level, keys in groups:
            rows = [(CHECK_LABELS.get(k, k), 100 * by_check[k] / a_n, "hl" if level == "high" else "warn" if level == "medium" else "neutral") for k in keys if by_check.get(k)]
            if not rows:
                continue
            rows.sort(key=lambda r: -r[1])
            group_html.append(f'<figure class="fig"><figcaption>{esc(title)} · share of {a_n} audited datasets</figcaption>{hbars(rows)}</figure>')
        dup = by_check.get("episodes.duplicate", 0)
        empty = by_check.get("tasks.placeholder", 0)
        idle = by_check.get("episodes.idle", 0)
        audit_html = f"""
<section id="episodes" class="sec">
  <div class="col">
    <p class="eyebrow">Episodes</p>
    <h2>Inside the data: duplicates, empty instructions and robots that never move</h2>
    <p>For {a_n} datasets we downloaded the tables (actions, states, timestamps and metadata, no video) and checked all {audit['episodes_audited']:,} episodes. The sample is half the most downloaded datasets and half picked at random, limited to datasets whose tables fit in 250 MB, so the largest corpora are not included.</p>
    <p>{dup} of the {a_n} datasets contain exact duplicate episodes, {empty} contain episodes with an empty or placeholder task instruction, and {idle} contain episodes where the robot barely moves. We group the findings by how much they can be trusted: an exact duplicate is certain, while a statistical rule can be wrong.</p>
  </div>
  {''.join(group_html)}
  <p class="note">{audit['skipped_too_large']} sampled datasets were skipped because their tables exceeded 250 MB, and {audit['errors']} could not be read. The glitch check is new and fires on many datasets with noisy or unusual signals, and video length mismatches often follow harmless trimming, so treat the review group as leads to inspect, not as faults. Since this audit ran we have changed three checks (the glitch threshold, how action lag is measured, and glitches that occur in most episodes of a channel, which are now reported once per dataset rather than per episode); a re-audit with the current version is under way and will change the counts for those rows.</p>
</section>"""
    else:
        audit_html = """
<section id="episodes" class="sec">
  <p class="eyebrow">Episodes</p>
  <h2>Inside the data</h2>
  <p class="note">The episode-level audit is still running. This section will fill in when it finishes.</p>
</section>"""

    ts = toy["summary"]
    rate30 = next(r for r in ts["by_rate"] if abs(r["corruption_rate"] - 0.3) < 1e-9)

    return f"""<title>State of Open Robot Data</title>
<meta name="description" content="An audit of {n:,} LeRobot datasets on Hugging Face: licences, metadata and episode-level recording faults.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column with wider figure bands; robot-cell safety yellow marks the findings. */
:root {{
  --paper: #eef1f0; --panel: #f8faf9; --ink: #15201d; --muted: #566561; --rule: #d3dbd8;
  --signal: #f2c200; --signal-ink: #1a1500; --link: #6f5600;
  --ok: #2f7d5b; --warn: #b7791f; --crit: #b9412d; --neutral: #8a9a96;
  --display: "Archivo", "Arial Narrow", "Helvetica Neue", sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #0f1514; --panel: #151d1b; --ink: #e4ebe9; --muted: #92a29e; --rule: #26312e;
  --signal: #f2c200; --signal-ink: #1a1500; --link: #f2c200;
  --ok: #4fae84; --warn: #e0a640; --crit: #e2694f; --neutral: #5d6d69; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --paper: #0f1514; --panel: #151d1b; --ink: #e4ebe9; --muted: #92a29e; --rule: #26312e;
  --signal: #f2c200; --signal-ink: #1a1500; --link: #f2c200;
  --ok: #4fae84; --warn: #e0a640; --crit: #e2694f; --neutral: #5d6d69; color-scheme: dark; }}
* {{ box-sizing: border-box }}
body {{ background: var(--paper); color: var(--ink); font: 400 1.0625rem/1.65 var(--body); }}
.page {{ max-width: 980px; margin: 0 auto; padding-inline: 16px; padding-block: 40px 72px; }}
.col {{ max-width: 680px; }}
a {{ color: var(--link); text-underline-offset: 3px; }}
a:focus-visible {{ outline: 2px solid var(--signal); outline-offset: 2px; }}
.eyebrow {{ font: 500 .75rem/1 var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin: 0 0 14px; }}
h1, h2 {{ font-family: var(--display); font-stretch: 75%; text-wrap: balance; margin: 0; letter-spacing: -.01em; }}
h1 {{ font-size: clamp(2.3rem, 6.2vw, 4.1rem); line-height: .98; font-weight: 800; max-width: 15ch; }}
h2 {{ font-size: clamp(1.55rem, 3.6vw, 2.15rem); line-height: 1.08; font-weight: 750; max-width: 24ch; margin-bottom: 14px; }}
p {{ margin: 0 0 14px; max-width: 65ch; }}
mark {{ background: var(--signal); color: var(--signal-ink); padding: 0 .18em; border-radius: 2px; }}
header.hero {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 40px; align-items: end; padding-bottom: 36px; border-bottom: 1px solid var(--rule); }}
.hero .lede {{ font-size: 1.15rem; color: var(--ink); margin-top: 20px; }}
.hero .by {{ font: 400 .8rem/1.5 var(--mono); color: var(--muted); margin-top: 18px; }}
.waffle-box {{ display: grid; gap: 10px; }}
.waffle {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 5px; max-width: 320px; }}
.w {{ aspect-ratio: 1.6; border-radius: 2px; background: var(--rule); max-width: 100%; }}
.w.hit {{ background: var(--signal); }}
.waffle-cap {{ font: 400 .8rem/1.45 var(--mono); color: var(--muted); max-width: 320px; }}
.waffle-cap b {{ color: var(--ink); font-weight: 500; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0; border-bottom: 1px solid var(--rule); }}
.stat {{ padding: 22px 18px 22px 0; display: grid; gap: 4px; }}
.stat + .stat {{ padding-left: 18px; border-left: 1px solid var(--rule); }}
.stat b {{ font: 750 2.1rem/1 var(--display); font-stretch: 75%; font-variant-numeric: tabular-nums; }}
.stat span {{ font: 400 .82rem/1.4 var(--mono); color: var(--muted); }}
.sec {{ padding-block: 44px 8px; border-bottom: 1px solid var(--rule); }}
.sec:last-of-type {{ border-bottom: 0; }}
.fig {{ margin: 24px 0 18px; background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 18px 14px; }}
.fig figcaption {{ font: 500 .78rem/1.3 var(--mono); color: var(--muted); letter-spacing: .04em; margin-bottom: 14px; }}
.bars {{ display: grid; gap: 9px; }}
.bar-row {{ display: grid; grid-template-columns: minmax(0, 15rem) minmax(0, 1fr) 3.4rem; gap: 12px; align-items: center; }}
.bar-label {{ font-size: .92rem; line-height: 1.25; min-width: 0; }}
.bar-track {{ height: 14px; background: color-mix(in srgb, var(--rule) 55%, transparent); border-radius: 2px; overflow: hidden; }}
.bar {{ display: block; height: 100%; border-radius: 2px; background: var(--neutral); }}
.bar.hl {{ background: var(--signal); }} .bar.ok {{ background: var(--ok); }} .bar.warn {{ background: var(--warn); }} .bar.crit {{ background: var(--crit); }}
.bar-value {{ font: 500 .85rem/1 var(--mono); text-align: right; font-variant-numeric: tabular-nums; }}
.chart-wrap {{ overflow-x: auto; }}
.chart {{ width: 100%; min-width: 480px; height: auto; display: block; }}
.chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
.chart .tick {{ fill: var(--muted); font: 400 11px var(--mono); }}
.chart .axis {{ fill: var(--muted); font: 400 11px var(--mono); }}
.s-raw {{ fill: var(--neutral); background: var(--neutral); }}
.s-rand {{ fill: var(--muted); background: var(--muted); }}
.s-filt {{ fill: var(--signal); background: var(--signal); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 10px; font: 400 .8rem/1.3 var(--mono); color: var(--muted); }}
.key i {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }}
.note {{ font-size: .93rem; color: var(--muted); }}
.callout {{ border-left: 3px solid var(--signal); padding: 4px 0 4px 16px; margin: 20px 0; max-width: 65ch; }}
.callout p {{ margin: 0; }}
code, pre {{ font-family: var(--mono); font-size: .85rem; }}
pre {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; overflow-x: auto; line-height: 1.55; margin: 10px 0 18px; }}
ul.plain {{ padding-left: 20px; max-width: 65ch; }}
ul.plain li {{ margin-bottom: 6px; }}
footer {{ padding-top: 28px; font: 400 .8rem/1.6 var(--mono); color: var(--muted); }}
@media (max-width: 720px) {{
  header.hero {{ grid-template-columns: 1fr; gap: 28px; }}
  .bar-row {{ grid-template-columns: minmax(0, 1fr) 3.2rem; }}
  .bar-row .bar-track {{ grid-column: 1 / -1; grid-row: 2; }}
  .stat + .stat {{ border-left: 0; padding-left: 0; border-top: 1px solid var(--rule); }}
}}
@media (prefers-reduced-motion: no-preference) {{
  .w.hit {{ animation: pop .5s ease-out both; }}
  @keyframes pop {{ from {{ transform: scale(.4); }} to {{ transform: none; }} }}
}}
</style>

<div class="page">
<header class="hero">
  <div>
    <p class="eyebrow">Open robot data audit · October 2026</p>
    <h1>Nearly half of the most-used robot datasets carry a licence nobody chose</h1>
    <p class="lede">We audited {n:,} robot-learning datasets on Hugging Face, every dataset tagged LeRobot from the most downloaded down. We checked their licences, their metadata and, for a sample, every recorded episode.</p>
    <p class="by">Built with rdq, our checker for LeRobot datasets. Method and limits at the end.</p>
  </div>
  <div class="waffle-box">
    {waffle(top_hit, top_tot)}
    <p class="waffle-cap"><b>{top_hit} of the {top_tot} most-downloaded</b> LeRobot datasets ship LeRobot's unedited card template with its default <code>apache-2.0</code> tag.</p>
  </div>
</header>

<div class="stats">
  <div class="stat"><b>{all_pct:.0f}%</b><span>of all {n:,} datasets have the untouched default licence</span></div>
  <div class="stat"><b>{dl_share}</b><span>of all downloads go to those datasets</span></div>
  <div class="stat"><b>{own_pct:.0f}%</b><span>of {own_tot} dataset owners publish at least one</span></div>
</div>

<section id="licence" class="sec">
  <div class="col">
    <p class="eyebrow">Licence &amp; provenance</p>
    <h2>The upload tool picks the licence</h2>
    <p>LeRobot's <code>push_to_hub()</code> fills in <code>license: apache-2.0</code> unless you change it, and writes a README whose homepage, paper and citation all read <em>[More Information Needed]</em>. When a dataset still has exactly that card, the licence is most likely a default nobody chose, and there is no record of who collected the data or how.</p>
    <p>Of the {ap_tot:,} datasets tagged <code>apache-2.0</code>, {ap_hit:,} ({ap_pct:.0f}%) are in that state. Some of these are conversions of well-known research datasets whose original terms may differ from the tag. A team that trains a commercial model on them inherits a licence question the tag does not answer.</p>
  </div>
  <figure class="fig">
    <figcaption>What the dataset card says, all {n:,} datasets</figcaption>
    {hbars(verdict_rows)}
  </figure>
  <p class="note">One organisation, {esc(top_owner)}, publishes {top_owner_n} of the {n:,} datasets with its own card template, which our checker counts as written by the owner. Without it, the untouched-default share would be higher, not lower.</p>
</section>

{upstream_html}
<section id="metadata" class="sec">
  <div class="col">
    <p class="eyebrow">Metadata</p>
    <h2>Smaller gaps that make datasets hard to combine</h2>
    <p>Mixing datasets from different robots is how most teams get enough data. That only works when each dataset says which robot recorded it and what each number means. Shares below are of the {robot[1]:,} datasets with a readable <code>meta/info.json</code>, except the first row, which is of all {n:,}.</p>
  </div>
  <figure class="fig">
    <figcaption>Metadata problems, share of datasets</figcaption>
    {hbars(meta_rows)}
  </figure>
  <p class="note">An earlier version of our checker reported far more name mismatches. Most turned out to be valid conventions (names wrapped in an extra list, or named by the last axis of an action chunk), so we fixed the check before publishing. The {names_mm[0]} that remain are real, such as a 44-value state vector labelled <code>height, width, channels</code>.</p>
</section>
{audit_html}
<section id="training" class="sec">
  <div class="col">
    <p class="eyebrow">Training</p>
    <h2>Do these faults matter? In a controlled test, a lot</h2>
    <p>We recorded expert demonstrations for a simple position-controlled reaching robot, broke a share of the episodes with realistic faults (a frozen log, actions written 8 frames late, a 3-frame teleop dropout, a failed attempt), and trained behaviour-cloning policies three ways. Each bar is the success rate over 1,200 evaluation rollouts.</p>
  </div>
  <figure class="fig">
    <figcaption>Policy success rate by training data</figcaption>
    {toy_chart(ts["by_rate"])}
  </figure>
  <div class="callout"><p>With 30% of episodes broken, training on everything succeeded {rate30['raw'] * 100:.0f}% of the time and a random subset of the same size {rate30['random'] * 100:.0f}%. Training on only the episodes rdq kept succeeded <mark>{rate30['filtered'] * 100:.0f}%</mark> of the time.</p></div>
  <div class="col">
    <p class="note">This is a toy task with faults we designed, so it shows the mechanism, not the size of the effect on real robots. rdq caught every planted idle, late-action and glitch episode ({ts['detection']['idle']}, {ts['detection']['desync']}, {ts['detection']['glitch']}) and none of the failed attempts ({ts['detection']['wrong_goal']}), which look like normal motion to a rule-based check. It wrongly dropped {ts['clean_episodes_wrongly_dropped']} clean episodes. The same test on a real vision-language-action model in simulation is next.</p>
  </div>
</section>

{bench_html}
<section id="method" class="sec">
  <div class="col">
    <p class="eyebrow">Method &amp; limits</p>
    <h2>How we did this</h2>
    <ul class="plain">
      <li>Population: the {n:,} most-downloaded Hugging Face datasets tagged <code>LeRobot</code>, scanned in October 2026. For each we read only the README and <code>meta/info.json</code>.</li>
      <li>"Untouched default" means the licence is <code>apache-2.0</code> and the README still contains LeRobot's template text with no description, homepage or paper. It is evidence that nobody chose the licence, not proof. Some owners may have meant apache-2.0.</li>
      <li>The episode audit reads actions, states and timestamps, never video. Its checks are rules with published thresholds, and every flag points to the episode and the rule that raised it.</li>
      <li>We name no individual dataset as having a problem in this report. We plan to release the checker as open source so any owner can run it on their own data.</li>
    </ul>
    <p class="eyebrow" style="margin-top:28px">Run it on your data</p>
    <p>One command reads a LeRobot dataset folder and writes a report plus a list of the episodes worth training on:</p>
<pre>rdq check path/to/lerobot_dataset --html report.html --filter keep.json</pre>
  </div>
</section>

<footer>State of Open Robot Data · October 2026 · Data: Hugging Face Hub. Counts reflect the hub at scan time and will drift as datasets change.</footer>
</div>
"""


if __name__ == "__main__":
    summary = json.loads(Path(sys.argv[1]).read_text())
    toy = json.loads(Path(sys.argv[2]).read_text())
    bench = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 else None
    Path(sys.argv[3]).write_text(build(summary, toy, bench))
    print("wrote", sys.argv[3])
