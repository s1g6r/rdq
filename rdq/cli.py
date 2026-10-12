"""Command line interface.

    rdq check PATH [--json FILE] [--html FILE] [--filter FILE]
    rdq scan   --limit 2000 --out results/       # licence + metadata for many hub datasets
    rdq audit  --from-scan results/ --n 150 --out results/   # full checks on a sample
    rdq audit  --repos lerobot/svla_so101_pickplace --out results/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .dataset import DatasetLoadError


def _cmd_check(args) -> int:
    from .report import audit, format_terminal, write_filter, write_html, write_json

    try:
        result = audit(args.path, max_frames=args.max_frames)
    except DatasetLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_terminal(result, color=sys.stdout.isatty() and not args.no_color))
    if args.json:
        write_json(result, args.json)
    if args.html:
        write_html(result, args.html)
    if args.filter:
        write_filter(result, args.filter)
    if args.fail_under is not None and result["score"]["dataset_score"] < args.fail_under:
        return 1
    return 0


def _cmd_scan(args) -> int:
    from .hub import scan

    scan(args.limit, args.out, sort=args.sort)
    return 0


def _cmd_audit(args) -> int:
    from .hub import audit_repos

    repos: list[str] = list(args.repos or [])
    if args.from_scan:
        from .hub import backfill_raw

        backfill_raw(args.from_scan)
        rows = [json.loads(l) for l in open(Path(args.from_scan) / "scan.jsonl")]
        eligible = [
            r["id"]
            for r in rows
            if r.get("info")
            and str(r["info"].get("codebase_version", "")).lstrip("v").startswith(("3", "2"))
            and r["info"].get("has_action")
            and not r.get("gated")
        ]
        rng = random.Random(args.seed)
        # Half the sample from the most downloaded datasets, half at random from the rest.
        top = eligible[: args.n // 2]
        rest = eligible[args.n // 2 :]
        sample = top + rng.sample(rest, min(len(rest), args.n - len(top)))
        repos += sample
        (Path(args.out)).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "audit_sample.json").write_text(json.dumps(sample, indent=1))
        print(f"auditing {len(sample)} datasets ({len(top)} most downloaded + {len(sample) - len(top)} random)")
    if not repos:
        print("nothing to audit: pass --repos or --from-scan", file=sys.stderr)
        return 2
    audit_repos(repos, args.out, max_mb=args.max_mb, keep_data=args.keep_data)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rdq", description="Quality and provenance checks for robot-learning datasets")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="audit a local LeRobot dataset folder")
    c.add_argument("path")
    c.add_argument("--json", help="write the full report as JSON")
    c.add_argument("--html", help="write a shareable HTML report")
    c.add_argument("--filter", help="write the list of usable episodes (for LeRobot's episodes=)")
    c.add_argument("--max-frames", type=int, help="only read this many frames (quick look)")
    c.add_argument("--fail-under", type=int, help="exit with code 1 if the score is below this (for CI)")
    c.add_argument("--no-color", action="store_true")
    c.set_defaults(fn=_cmd_check)

    s = sub.add_parser("scan", help="licence and metadata scan of hub datasets tagged LeRobot")
    s.add_argument("--limit", type=int, default=2000)
    s.add_argument("--sort", default="downloads")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_cmd_scan)

    a = sub.add_parser("audit", help="download and fully audit hub datasets (no videos)")
    a.add_argument("--repos", nargs="*")
    a.add_argument("--from-scan", help="folder with scan.jsonl to sample from")
    a.add_argument("--n", type=int, default=150)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--max-mb", type=float, default=300)
    a.add_argument("--keep-data", action="store_true")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=_cmd_audit)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
