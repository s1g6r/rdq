"""Command line interface.

    rdq check PATH [--json FILE] [--html FILE] [--filter FILE]
    rdq scan   --limit 2000 --out results/       # licence + metadata for many hub datasets
    rdq audit  --from-scan results/ --n 150 --out results/   # full checks on a sample
    rdq audit  --repos lerobot/svla_so101_pickplace --out results/
    rdq passport PATH --repo-id owner/name --answers answers.yaml --out passport.json
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
        result = audit(args.path, max_frames=args.max_frames, repo_id=args.repo_id)
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


def _cmd_passport(args) -> int:
    from .passport import answers_template, build_passport, load_answers, to_datasheet, to_spdx
    from .report import audit

    if args.init_answers:
        Path(args.init_answers).write_text(answers_template())
        print(f"wrote {args.init_answers}: fill in what you know, then pass it with --answers")
        return 0
    if not args.path:
        print("error: PATH is required (or use --init-answers FILE)", file=sys.stderr)
        return 2
    root = Path(args.path)
    readme = (root / "README.md").read_text(errors="replace") if (root / "README.md").exists() else None
    try:
        answers = load_answers(args.answers)
        result = audit(root, readme=readme, repo_id=args.repo_id)
    except (DatasetLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    passport = build_passport(result, readme, root, repo_id=args.repo_id, revision=args.revision,
                              answers=answers, include_videos_in_hash=args.hash_videos)
    out = Path(args.out or "passport.json")
    out.write_text(json.dumps(passport, indent=2, default=str))
    lic = passport["rdq:license"]
    print(f"passport: {out}  (licence {lic.get('declared') or 'none'} -> {lic['status']}; "
          f"quality {passport['rdq:quality']['score']}/100; {len(passport.get('rdq:unanswered', []))} owner questions unanswered)")
    if args.spdx:
        Path(args.spdx).write_text(json.dumps(to_spdx(passport), indent=2, default=str))
        print(f"spdx:     {args.spdx}")
    if args.datasheet:
        Path(args.datasheet).write_text(to_datasheet(passport))
        print(f"datasheet: {args.datasheet}")
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
    c.add_argument("--repo-id", help="hub id of the dataset (enables upstream-licence matching by name)")
    c.set_defaults(fn=_cmd_check)

    pp = sub.add_parser("passport", help="write a data passport (Croissant + RAI JSON-LD, optional SPDX and datasheet)")
    pp.add_argument("path", nargs="?")
    pp.add_argument("--repo-id", help="hub id, e.g. owner/name")
    pp.add_argument("--revision", help="hub commit or version the passport describes")
    pp.add_argument("--answers", help="YAML file with the owner's answers (see --init-answers)")
    pp.add_argument("--out", help="passport JSON-LD path (default passport.json)")
    pp.add_argument("--spdx", help="also write an SPDX 3.0 JSON-LD dataset package")
    pp.add_argument("--datasheet", help="also write a Markdown datasheet")
    pp.add_argument("--hash-videos", action="store_true", help="include videos/ in the content hash (slower)")
    pp.add_argument("--init-answers", metavar="FILE", help="write a blank answers template and exit")
    pp.set_defaults(fn=_cmd_passport)

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
