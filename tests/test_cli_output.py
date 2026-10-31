"""`rdq check` output that goes to a pipe or a file (issue #1)."""

import io
import sys

from rdq import synthetic as syn
from rdq.cli import main


def test_check_prints_the_report_to_a_cp1252_pipe(tmp_path, monkeypatch):
    """Windows encodes redirected output with the ANSI code page. The report's "→" and "·" used to raise
    UnicodeEncodeError there; now redirected output is written as UTF-8."""
    root = tmp_path / "ds"
    syn.write_dataset(root, syn.make_episodes(6, 120), with_video=False)
    (root / "README.md").write_text("---\nlicense: apache-2.0\n---\n# Test dataset\n")  # prints "licence: ... → ..."
    raw = io.BytesIO()
    pipe = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")  # what `rdq check ... > report.txt` gets
    monkeypatch.setattr(sys, "stdout", pipe)

    assert main(["check", str(root)]) == 0
    pipe.flush()
    text = raw.getvalue().decode("utf-8")
    assert "→" in text and "episodes" in text
