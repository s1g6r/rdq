"""rdq licence and provenance check, running entirely in the browser (Gradio-Lite).

Paste a LeRobot dataset id. The page fetches the dataset card from the Hugging Face Hub
and compares its declared licence with the terms of the public dataset it was converted
from, using rdq's knowledge base. Full quality checks need the command line:
pip install git+https://github.com/s1g6r/rdq
"""

import json
import re

import gradio as gr
from pyodide.http import pyfetch

from rdq.provenance import analyse_card

REPO = "https://github.com/s1g6r/rdq"
SEV = {"critical": "🔴 critical", "warning": "🟠 warning", "info": "🔵 info", "ok": "🟢 ok"}


async def fetch_card(repo_id: str):
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/README.md"
    resp = await pyfetch(url)
    if resp.status == 404:
        return None, "no dataset card (README.md) found"
    if not resp.ok:
        return None, f"the Hub answered {resp.status}"
    return await resp.string(), None


async def check(repo_id: str) -> str:
    repo_id = (repo_id or "").strip().strip("/")
    m = re.search(r"huggingface\.co/datasets/([^/\s]+/[^/\s?#]+)", repo_id)
    if m:
        repo_id = m.group(1)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        return "Enter a dataset id like `lerobot/droid_1.0.1`."
    try:
        readme, err = await fetch_card(repo_id)
    except Exception as exc:  # noqa: BLE001
        return f"Could not reach the Hub: {type(exc).__name__}"
    if err:
        return f"### `{repo_id}`\n\nCould not read the card: {err}. Private and gated datasets can't be checked here."
    p = analyse_card(readme, repo_id=repo_id)
    up = p.upstream or {}
    lines = [f"### `{repo_id}`", "", f"**Declared licence:** {p.license or 'none'}", f"**Verdict:** {p.verdict}"]
    if up.get("upstream"):
        lines.append(f"**Compared with the original:** {up.get('status')} ({SEV.get(up.get('severity'), up.get('severity'))})")
        lines.append("")
        for u in up["upstream"]:
            lines.append(f"- **{u['name']}**: {u['license']} (matched by {u['evidence']}, {u['confidence']} confidence). "
                         f"Source: [{u['statement_url']}]({u['statement_url']}), checked {u['checked']}")
    if p.notes:
        lines += ["", "**Notes**"] + [f"- {n}" for n in p.notes]
    lines += ["", "---",
              "A licence tag on a copy can't grant rights its uploader doesn't hold. This is a diligence aid, not legal advice. "
              f"For full quality checks (desync, frozen logs, glitches, duplicates) and a data passport, run rdq locally: {REPO}"]
    return "\n".join(lines)


with gr.Blocks(title="rdq: check a robot dataset's licence") as demo:
    gr.Markdown("# rdq: can you train on this robot dataset?\n"
                "Paste a LeRobot dataset id from the Hugging Face Hub. rdq checks its declared licence against the "
                "terms of the original dataset it was converted from (65 public datasets checked against their primary "
                "sources). Runs in your browser; the first load takes a few seconds.")
    with gr.Row():
        box = gr.Textbox(label="Dataset id", placeholder="lerobot/droid_1.0.1", scale=4)
        btn = gr.Button("Check", variant="primary")
    out = gr.Markdown()
    gr.Examples(["lerobot/droid_1.0.1", "lerobot/pusht", "lerobot/svla_so101_pickplace"], inputs=box)
    btn.click(check, box, out)
    box.submit(check, box, out)

demo.launch()
