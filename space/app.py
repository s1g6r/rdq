"""rdq on Hugging Face Spaces: check any LeRobot dataset on the Hub.

Paste a dataset id. The app reads the card and metadata to check the licence and
provenance (seconds). It can also download the tables, never the videos, to run every
quality check (up to a size limit). It returns the report and a data passport.

Deploy: create a Gradio Space and upload app.py, requirements.txt and README.md from
this folder. requirements.txt installs rdq from GitHub.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq.passport import build_passport, to_datasheet  # noqa: E402
from rdq.provenance import analyse_card  # noqa: E402
from rdq.report import audit, render_html  # noqa: E402

MAX_MB = 300


def _hub():
    import huggingface_hub as hf

    return hf


def licence_only(repo_id: str) -> tuple[str, dict]:
    hf = _hub()
    try:
        readme = Path(hf.hf_hub_download(repo_id, "README.md", repo_type="dataset")).read_text(errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"Could not read the card of `{repo_id}`: {type(exc).__name__}", {}
    prov = analyse_card(readme, repo_id=repo_id)
    up = prov.upstream or {}
    lines = [f"### `{repo_id}`", "", f"**Declared licence:** {prov.license or 'none'}", f"**Verdict:** {prov.verdict}",
             f"**Upstream check:** {up.get('status')}" + (f" (upstream: {up.get('concluded')})" if up.get("concluded") else "")]
    for n in prov.notes:
        lines.append(f"- {n}")
    for u in up.get("upstream") or []:
        lines.append(f"- Evidence: “{u['statement']}” — [{u['name']}]({u['statement_url']}), checked {u['checked']}, matched by {u['evidence']}")
    lines += ["", "_A licence tag on a mirror cannot grant rights the uploader does not hold. This is a diligence aid, not legal advice._"]
    return "\n".join(lines), prov.to_dict()


def full_check(repo_id: str, run_quality: bool):
    md, prov = licence_only(repo_id)
    if not run_quality or not prov:
        return md, "", None, None
    hf = _hub()
    try:
        info = hf.HfApi().dataset_info(repo_id, files_metadata=True)
        size = sum((s.size or 0) for s in info.siblings if s.rfilename.startswith(("data/", "meta/"))) / 1e6
    except Exception as exc:  # noqa: BLE001
        return md + f"\n\nCould not list files: {type(exc).__name__}", "", None, None
    if size > MAX_MB:
        return md + f"\n\nTables are {size:.0f} MB, over this Space's {MAX_MB} MB limit. Run `rdq check` locally instead.", "", None, None
    work = Path(tempfile.mkdtemp())
    try:
        local = Path(hf.snapshot_download(repo_id, repo_type="dataset", local_dir=str(work / "ds"),
                                          allow_patterns=["meta/*", "meta/**", "data/**", "README.md"]))
        readme = (local / "README.md").read_text(errors="replace") if (local / "README.md").exists() else None
        result = audit(local, readme=readme, repo_id=repo_id)
        result["dataset"]["path"] = repo_id
        passport = build_passport(result, readme, local, repo_id=repo_id, revision=getattr(info, "sha", None))
        out = Path(tempfile.mkdtemp())
        (out / "passport.json").write_text(json.dumps(passport, indent=2, default=str))
        (out / "DATASHEET.md").write_text(to_datasheet(passport))
        sc = result["score"]
        md += (f"\n\n### Quality\n**Score {sc['dataset_score']}/100**: {len(sc['usable_episodes'])}/{result['dataset']['episodes']} "
               "episodes pass every critical and warning check.")
        return md, render_html(result), str(out / "passport.json"), str(out / "DATASHEET.md")
    except Exception as exc:  # noqa: BLE001
        return md + f"\n\nQuality check failed: {type(exc).__name__}: {exc}", "", None, None
    finally:
        shutil.rmtree(work, ignore_errors=True)


with gr.Blocks(title="rdq: robot dataset check") as demo:
    gr.Markdown("# rdq: check a LeRobot dataset before you train on it\n"
                "Licence and provenance in seconds, plus full quality checks on the tables (no videos downloaded).")
    with gr.Row():
        repo = gr.Textbox(label="Hugging Face dataset id", placeholder="lerobot/svla_so101_pickplace", scale=4)
        quality = gr.Checkbox(label="Also run quality checks", value=True)
        go = gr.Button("Check", variant="primary")
    summary = gr.Markdown()
    report = gr.HTML()
    with gr.Row():
        passport_file = gr.File(label="Data passport (Croissant + RAI JSON-LD)")
        datasheet_file = gr.File(label="Datasheet (Markdown)")
    go.click(full_check, [repo, quality], [summary, report, passport_file, datasheet_file])

if __name__ == "__main__":
    demo.launch()
