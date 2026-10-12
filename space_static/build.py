"""Build index.html for the free Static Space: rdq's licence check running in the browser (Pyodide).

    python space_static/build.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "rdq/__init__.py": ROOT / "rdq" / "__init__.py",
    "rdq/licenses.py": ROOT / "rdq" / "licenses.py",
    "rdq/provenance.py": ROOT / "rdq" / "provenance.py",
    "rdq/data/upstream_licences.json": ROOT / "rdq" / "data" / "upstream_licences.json",
}
PYODIDE = "v0.27.3"
TEMPLATE = (ROOT / "space_static" / "page.html").read_text()


def build() -> str:
    blocks = []
    for name, path in FILES.items():
        text = path.read_text()
        # Script contents are not HTML-decoded, so the text goes in raw; it must not close the tag.
        assert "</script" not in text.lower(), name
        blocks.append(f'<script type="text/plain" class="rdq-file" data-path="{name}">{text}</script>')
    blocks = "\n".join(blocks)
    return TEMPLATE.replace("{{PYODIDE}}", PYODIDE).replace("{{FILES}}", blocks)


if __name__ == "__main__":
    out = ROOT / "space_static" / "index.html"
    out.write_text(build())
    print("wrote", out)
