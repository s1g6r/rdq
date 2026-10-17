"""Expected findings: episodes the dataset's owner says are unusual on purpose.

Some episodes look like defects to a checker but were recorded that way deliberately: an empty
scene so a policy learns not to grasp, a camera held still for a calibration shot, a long pause
that is part of the task. The owner lists them in a small YAML file, and rdq reports matching
findings as expected (info, with the owner's reason) instead of warnings, so those episodes stay
in the keep list.

The file lives in the dataset at ``meta/rdq_expected.yaml``, so anyone who checks the dataset,
including from the Hub, gets the same answer. ``rdq check --expected FILE`` uses another file.

    # meta/rdq_expected.yaml
    - episodes: [342, 343, 349, 350, 351]
      checks: ["video.*"]          # optional; default: every check
      reason: no bottle on the table, so the policy learns not to grasp

Episodes can also be ranges ("10-20"). An entry with no ``episodes`` applies to dataset-level
findings (and to every episode) for the checks it names, so it must name at least one check.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

from .checks import INFO, Finding

DATASET_FILE = Path("meta") / "rdq_expected.yaml"


class ExpectedError(ValueError):
    """The expected-findings file could not be read or is malformed."""


@dataclass
class Expected:
    reason: str
    episodes: set[int] | None = None  # None = every episode and dataset-level findings
    checks: list[str] = field(default_factory=lambda: ["*"])

    def matches_check(self, check: str) -> bool:
        return any(fnmatch.fnmatchcase(check, pat) for pat in self.checks)


def _episodes(value, where: str) -> set[int]:
    if isinstance(value, (int, str)):
        value = [value]
    if not isinstance(value, list):
        raise ExpectedError(f"{where}: 'episodes' must be a list of numbers or 'a-b' ranges")
    out: set[int] = set()
    for v in value:
        if isinstance(v, bool):
            raise ExpectedError(f"{where}: bad episode {v!r}")
        if isinstance(v, int):
            out.add(v)
            continue
        s = str(v).strip()
        try:
            if "-" in s.lstrip("-"):
                a, b = s.split("-", 1)
                lo, hi = int(a), int(b)
                if hi < lo or hi - lo > 1_000_000:
                    raise ValueError
                out.update(range(lo, hi + 1))
            else:
                out.add(int(s))
        except ValueError:
            raise ExpectedError(f"{where}: bad episode {v!r} (use a number or a range like 10-20)") from None
    return out


def parse(data, source: str = "expected file") -> list[Expected]:
    if data is None:
        return []
    if isinstance(data, dict):
        data = data.get("expected", data.get("intentional"))
        if data is None:
            raise ExpectedError(f"{source}: expected a list of entries (or an 'expected:' key)")
    if not isinstance(data, list):
        raise ExpectedError(f"{source}: expected a list of entries")
    out = []
    for i, item in enumerate(data, 1):
        where = f"{source}, entry {i}"
        if not isinstance(item, dict):
            raise ExpectedError(f"{where}: each entry needs 'episodes' and/or 'checks', and a 'reason'")
        reason = str(item.get("reason") or "").strip()
        if not reason:
            raise ExpectedError(f"{where}: 'reason' is required, so the report can say why")
        checks = item.get("checks", item.get("check"))
        if checks is None:
            checks = ["*"]
        elif isinstance(checks, str):
            checks = [checks]
        if not isinstance(checks, list) or not all(isinstance(c, str) and c.strip() for c in checks):
            raise ExpectedError(f"{where}: 'checks' must be a list of check names (wildcards like video.* work)")
        eps = _episodes(item["episodes"], where) if "episodes" in item else None
        if eps is None and checks == ["*"]:
            raise ExpectedError(f"{where}: an entry without 'episodes' must name the checks it covers")
        out.append(Expected(reason=reason[:300], episodes=eps, checks=[c.strip() for c in checks]))
    return out


def load(path: str | Path) -> list[Expected]:
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise ExpectedError(f"cannot read {path}: {exc.strerror or exc}") from None
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001 - YAML and JSON raise different types
        raise ExpectedError(f"{path} is not valid {'JSON' if path.suffix.lower() == '.json' else 'YAML'}: "
                            f"{str(exc).splitlines()[0][:200]}") from None
    return parse(data, source=path.name)


def find_in_dataset(root: str | Path) -> Path | None:
    for name in (DATASET_FILE, DATASET_FILE.with_suffix(".yml"), DATASET_FILE.with_suffix(".json")):
        p = Path(root) / name
        if p.is_file():
            return p
    return None


def apply(findings: list[Finding], entries: list[Expected]) -> list[Finding]:
    """Split findings into what the owner expected (info, with the reason) and what remains."""
    if not entries:
        return findings
    out: list[Finding] = []
    for f in findings:
        rules = [e for e in entries if e.matches_check(f.check)]
        if not rules:
            out.append(f)
            continue
        if not f.episodes:
            hit = next((e for e in rules if e.episodes is None), None)
            out.append(_expected(f, [], hit.reason) if hit else f)
            continue
        by_reason: dict[str, list[int]] = {}
        rest: list[int] = []
        for ep in f.episodes:
            hit = next((e for e in rules if e.episodes is None or ep in e.episodes), None)
            if hit:
                by_reason.setdefault(hit.reason, []).append(ep)
            else:
                rest.append(ep)
        if not by_reason:
            out.append(f)
            continue
        if rest:
            out.append(Finding(f.check, f.severity, f.message, rest,
                               {**f.details, "expected_episodes_removed": len(f.episodes) - len(rest)}))
        for reason, eps in by_reason.items():
            out.append(_expected(f, eps, reason))
    return out


def _expected(f: Finding, episodes: list[int], reason: str) -> Finding:
    what = f"{len(episodes)} episode{'s' if len(episodes) != 1 else ''}" if episodes else "the dataset"
    msg = f"Expected by the dataset's owner ({reason}): {f.check} in {what}"
    if f.severity == INFO:
        msg += "."
    else:
        msg += f", which would otherwise be a {f.severity}."
    return Finding(f.check, INFO, msg, list(episodes),
                   {**f.details, "expected": True, "expected_reason": reason,
                    "original_severity": f.severity, "original_message": f.message})
