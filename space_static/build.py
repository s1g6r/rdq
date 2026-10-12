"""Build index.html for the free Static Space: Gradio-Lite running rdq's licence check in the browser.

    python space_static/build.py
"""

import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "app.py": ROOT / "space_static" / "app.py",
    "rdq/__init__.py": ROOT / "rdq" / "__init__.py",
    "rdq/licenses.py": ROOT / "rdq" / "licenses.py",
    "rdq/provenance.py": ROOT / "rdq" / "provenance.py",
    "rdq/data/upstream_licences.json": ROOT / "rdq" / "data" / "upstream_licences.json",
}
LITE = "5.45.0"


def build() -> str:
    parts = []
    for name, path in FILES.items():
        entry = " entrypoint" if name == "app.py" else ""
        parts.append(f'<gradio-file name="{name}"{entry}>\n{html.escape(path.read_text())}\n</gradio-file>')
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>rdq: robot dataset licence check</title>
<script type="module" crossorigin src="https://cdn.jsdelivr.net/npm/@gradio/lite@{LITE}/dist/lite.js"></script>
<link rel="stylesheet" crossorigin href="https://cdn.jsdelivr.net/npm/@gradio/lite@{LITE}/dist/lite.css" />
</head>
<body>
<gradio-lite>
<gradio-requirements>
pyyaml
</gradio-requirements>
{chr(10).join(parts)}
</gradio-lite>
</body>
</html>
"""


if __name__ == "__main__":
    out = ROOT / "space_static" / "index.html"
    out.write_text(build())
    print("wrote", out)
