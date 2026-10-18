"""Command line interface.

    rdq check PATH [--json FILE] [--html FILE] [--filter FILE]
    rdq scan   --limit 2000 --out results/       # licence + metadata for many hub datasets
    rdq audit  --from-scan results/ --n 150 --out results/   # full checks on a sample
    rdq audit  --repos lerobot/svla_so101_pickplace --out results/
    rdq passport PATH --repo-id owner/name --answers answers.yaml --out passport.json
    rdq check-video delivery/ --by-folder --html report.html   # raw video: usable hours, defects, duplicates
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .dataset import DatasetLoadError


def _resolve_path(path: str, repo_id: str | None, video: bool, max_mb: float | None = None,
                  revision: str | None = None) -> tuple[str, str | None]:
    """A local folder is used as is. Anything shaped like owner/name that isn't a folder is
    fetched from the Hugging Face Hub (tables and card only, plus videos with --video)."""
    if Path(path).exists() or path.count("/") != 1 or path.startswith((".", "/")):
        return path, repo_id
    from .hub import _hf

    hf = _hf()
    patterns = ["meta/*", "meta/**", "data/**", "README.md"] + (["videos/**"] if video else [])
    print(f"no local folder '{path}', so fetching it from the Hugging Face Hub "
          f"({'with' if video else 'without'} videos)…", file=sys.stderr)
    try:
        if max_mb:
            rev = {"revision": revision} if revision else {}
            info = hf.HfApi().dataset_info(path, files_metadata=True, **rev)
            prefixes = ("data/", "meta/", "videos/") if video else ("data/", "meta/")
            size = sum((f.size or 0) for f in info.siblings if f.rfilename.startswith(prefixes)) / 1e6
            if size > max_mb:
                raise DatasetLoadError(f"'{path}' is {size:,.0f} MB to download, over --max-mb {max_mb:g}; "
                                       "raise the limit or download a subset yourself")
        local = hf.snapshot_download(path, repo_type="dataset", allow_patterns=patterns,
                                     **({"revision": revision} if revision else {}))
    except DatasetLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - hub errors come in many types
        raise DatasetLoadError(f"'{path}' is not a local folder and could not be fetched from the Hub "
                               f"({type(exc).__name__}: {str(exc).splitlines()[0][:200]})") from exc
    return local, repo_id or path


def _json_safe(o):
    """NaN/inf are not valid JSON; write them as null."""
    import math

    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if hasattr(o, "item"):
        return _json_safe(o.item())
    return o


def _cmd_check(args) -> int:
    from .expected import ExpectedError
    from .report import audit, format_terminal, write_filter, write_html, write_json, write_markdown

    try:
        want_video = args.video or args.video_full
        path, repo_id = _resolve_path(args.path, args.repo_id, want_video, args.max_mb, getattr(args, "revision", None))
        result = audit(path, max_frames=args.max_frames, repo_id=repo_id, video=args.video,
                       video_full=args.video_full, expected=args.expected)
        if repo_id:
            result["dataset"]["path"] = repo_id
        if getattr(args, "revision", None):
            result["dataset"]["revision"] = args.revision
    except (DatasetLoadError, ExpectedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_terminal(result, color=sys.stdout.isatty() and not args.no_color))
    if args.json:
        write_json(result, args.json)
    if args.html:
        write_html(result, args.html)
    if args.markdown:
        write_markdown(result, args.markdown)
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


def _cmd_rank(args) -> int:
    import numpy as np

    from .dataset import load_dataset
    from .scores import ScoreConfig, score_episodes

    try:
        ds = load_dataset(args.path)
    except DatasetLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from .checks import CRITICAL, WARNING, run_checks

    res = score_episodes(ds, ScoreConfig(chunk=args.chunk, seed=args.seed))
    eps = res["episodes"]
    if not eps:
        print(f"error: {res['summary'].get('error', 'nothing to score')}", file=sys.stderr)
        return 2
    # Recording faults found by the checks take precedence over the soft scores' notes: a
    # desynced episode must not be presented as a "minority strategy" to keep.
    rule_flags: dict[int, list[str]] = {}
    for fnd in run_checks(ds):
        if fnd.severity in (CRITICAL, WARNING):
            for i in fnd.episodes:
                rule_flags.setdefault(int(i), []).append(fnd.check)
    for i, e in eps.items():
        e["rule_flags"] = sorted(set(rule_flags.get(int(i), [])))
        if e["rule_flags"]:
            e["notes"] = [n for n in e["notes"] if not n.startswith("minority strategy")]
            e["notes"].insert(0, "check failed: " + ", ".join(e["rule_flags"]))
    key = args.by
    higher_better = key in ("composite", "deminf", "ldlj")

    def val(e):
        v = e.get(key)
        return v if v is not None and np.isfinite(v) else (-np.inf if higher_better else np.inf)

    order = sorted(eps, key=lambda i: val(eps[i]), reverse=higher_better)
    n_scored = sum(1 for e in eps.values() if e.get("composite") is not None or e.get("hfr") is not None)
    print(f"{n_scored} of {len(eps)} episodes scored ({res['summary'].get('frames_scored', 0)} frames, "
          f"{'relative' if res['summary'].get('relative_actions') else 'absolute'} actions); ranked by {key}")
    if res["summary"].get("length_confound_warning"):
        print("warning: " + res["summary"]["length_confound_warning"])
    for note in res["summary"].get("skipped", []):
        print("note: " + note)
    print(f"{'episode':>8} {'len':>6} {'composite':>9} {'deminf':>8} {'consist':>8} {'hfr':>6} {'dither':>6}  notes")

    def f(v, w=8, p=2):
        return f"{v:>{w}.{p}f}" if v is not None and np.isfinite(v) else " " * (w - 1) + "-"

    show = order if args.all else order[:10] + (["..."] if len(order) > 20 else []) + order[-10:] if len(order) > 20 else order
    for i in show:
        if i == "...":
            print(f"{'...':>8}")
            continue
        e = eps[i]
        print(f"{i:>8} {e['length']:>6} {f(e['composite'], 9)} {f(e['deminf'])} {f(e['consistency'])} {f(e['hfr'], 6)} "
              f"{f(e['dither'], 6)}  {'; '.join(e['notes'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(_json_safe(res), indent=1))
    if args.keep_top is not None:
        # Episodes with a failed check are never kept on the strength of a soft score.
        scored = [i for i in order if eps[i].get(key) is not None and np.isfinite(eps[i][key]) and not eps[i]["rule_flags"]]
        if not scored:
            print(f"error: no episode has a {key} score, so --keep-top cannot rank them", file=sys.stderr)
            return 2
        n = max(1, int(round(args.keep_top * len(scored))))
        Path(args.filter or "keep.json").write_text(json.dumps({"episodes": sorted(scored[:n])}))
        print(f"kept the top {n}/{len(scored)} by {key} -> {args.filter or 'keep.json'} (experimental: in rdq's "
              "benchmark, dropping episodes on soft scores did not beat training on all clean data; "
              "see experiments/README.md)")
    if args.keep_ranges:
        Path(args.keep_ranges).write_text(json.dumps({str(i): e["keep_ranges"] for i, e in eps.items()}))
        print(f"idle-trimmed frame ranges -> {args.keep_ranges}")
    return 0


def _cmd_scorecard(args) -> int:
    from .dataset import load_dataset
    from .report import audit
    from .scorecard import episode_groups, format_scorecard, scorecard
    from .scores import score_episodes

    try:
        path, repo_id = _resolve_path(args.path, None, False, args.max_mb)
        ds = load_dataset(path)
        result = audit(path, repo_id=repo_id)
    except DatasetLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    groups = episode_groups(ds, args.group_by, args.groups)
    if not groups:
        print("error: no groups found (check --group-by or --groups)", file=sys.stderr)
        return 2
    scores = None if args.no_scores else score_episodes(ds)
    rows = scorecard(result, scores, groups)
    print(format_scorecard(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


def _cmd_check_mcap(args) -> int:
    from .checks import run_checks
    from .mcap_reader import load_mcap
    from .report import build_result, format_terminal, write_filter, write_html, write_json
    from .scoring import score

    try:
        ds = load_mcap(args.files, args.state_topic, args.action_topic)
    except DatasetLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    findings = run_checks(ds, signal_only=True)
    order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (order[f.severity], f.check))
    result = build_result(ds, findings, score(findings, sorted(ds.episodes)), None)
    result["dataset"]["path"] = f"{len(args.files)} MCAP file(s)"
    result["dataset"]["episode_files"] = {str(i): e.meta.get("file") for i, e in ds.episodes.items()}
    print(format_terminal(result, color=sys.stdout.isatty() and not args.no_color))
    for w in ds.load_warnings:
        print(f"  note: {w}")
    if args.json:
        write_json(result, args.json)
    if args.html:
        write_html(result, args.html)
    if args.filter:
        write_filter(result, args.filter)
    return 0


def _cmd_check_video(args) -> int:
    from .rawvideo import Spec, check_video_batch, format_terminal, write_outputs

    try:
        import av  # noqa: F401
        import cv2  # noqa: F401
    except ImportError:
        print("error: check-video needs PyAV and OpenCV: pip install av opencv-python-headless", file=sys.stderr)
        return 2

    def progress(k, n, name):
        if sys.stderr.isatty():
            print(f"\r  analysed {k}/{n}: {name[:60]:<60}", end="", file=sys.stderr, flush=True)

    try:
        report = check_video_batch(args.paths, contributors_csv=args.contributors, by_folder=args.by_folder,
                                   spec=Spec.load(args.spec), store=args.store, workers=args.workers,
                                   progress=progress)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if sys.stderr.isatty():
        print(file=sys.stderr)
    print(format_terminal(report, color=sys.stdout.isatty() and not args.no_color))
    write_outputs(report, args.json, args.html, args.csv)
    return 1 if report["summary"]["files_critical"] else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rdq", description="Quality and provenance checks for robot-learning datasets")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="audit a LeRobot dataset folder, or a Hub dataset id (owner/name)")
    c.add_argument("path", help="local dataset folder, or owner/name on the Hugging Face Hub")
    c.add_argument("--json", help="write the full report as JSON")
    c.add_argument("--html", help="write a shareable HTML report")
    c.add_argument("--markdown", help="write a Markdown summary (e.g. $GITHUB_STEP_SUMMARY or a PR comment)")
    c.add_argument("--filter", help="write the list of usable episodes (for LeRobot's episodes=)")
    c.add_argument("--max-frames", type=int, help="only read this many frames (quick look)")
    c.add_argument("--fail-under", type=int, help="exit with code 1 if the score is below this (for CI)")
    c.add_argument("--no-color", action="store_true")
    c.add_argument("--repo-id", help="hub id of the dataset (enables upstream-licence matching by name)")
    c.add_argument("--max-mb", type=float, default=2000, help="refuse to fetch a Hub dataset larger than this (default 2000 MB)")
    c.add_argument("--revision", help="Hub commit, branch or tag to check (default: the latest)")
    c.add_argument("--video", action="store_true", help="also decode a few frames per episode (frozen/black/blur/decode checks; needs PyAV)")
    c.add_argument("--video-full", action="store_true",
                   help="decode every frame: frozen/black/blurry stretches, timing gaps and duplicated episodes (needs rdq[video])")
    c.add_argument("--expected", metavar="FILE",
                   help="YAML listing episodes that are unusual on purpose (default: meta/rdq_expected.yaml in the "
                        "dataset); matching findings become info with the owner's reason")
    c.set_defaults(fn=_cmd_check)

    r = sub.add_parser("rank", help="score and rank episodes with literature-based quality signals (no training)")
    r.add_argument("path")
    r.add_argument("--by", default="composite",
                   choices=["composite", "deminf", "consistency", "hfr", "ldlj", "dither", "psd_power", "final_outlier", "length"])
    r.add_argument("--chunk", type=int, default=4, help="action chunk length for DemInf-lite")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--json", help="write all per-episode scores as JSON")
    r.add_argument("--keep-top", type=float, help="experimental: keep this fraction of scored episodes, best first (writes --filter)")
    r.add_argument("--filter", help="where --keep-top writes the episode list (default keep.json)")
    r.add_argument("--keep-ranges", help="write idle-trimmed frame ranges per episode (openpi DROID recipe)")
    r.add_argument("--all", action="store_true", help="print every episode")
    r.set_defaults(fn=_cmd_rank)

    m = sub.add_parser("check-mcap", help="audit MCAP recordings (one file per episode) before converting them")
    m.add_argument("files", nargs="+")
    m.add_argument("--state-topic", required=True, help="e.g. /joint_states")
    m.add_argument("--action-topic", help="e.g. /leader/joint_states or /joint_commands")
    m.add_argument("--json")
    m.add_argument("--html")
    m.add_argument("--filter", help="write the usable episode indices (in file order)")
    m.add_argument("--no-color", action="store_true")
    m.set_defaults(fn=_cmd_check_mcap)

    v = sub.add_parser("check-video", help="audit a raw video delivery (folder or files): usable hours, defects, "
                                           "duplicates, per-contributor scorecard")
    v.add_argument("paths", nargs="+", help="video files and/or folders (searched recursively)")
    v.add_argument("--contributors", help="CSV with file,contributor columns")
    v.add_argument("--by-folder", action="store_true", help="contributor = first sub-folder under the given folder")
    v.add_argument("--spec", help="YAML delivery spec: fps, min_width, min_height, codecs, min_duration_s, "
                                  "max_duration_s, max_idle_share, max_blur_share")
    v.add_argument("--store", help="JSON hash store; new files are also compared against earlier batches in it, "
                                   "and this batch is added")
    v.add_argument("--workers", type=int, help="parallel processes (default: up to 4)")
    v.add_argument("--json")
    v.add_argument("--html")
    v.add_argument("--csv", help="one row per file with its findings")
    v.add_argument("--no-color", action="store_true")
    v.set_defaults(fn=_cmd_check_video)

    sc = sub.add_parser("scorecard", help="quality per collector, operator, session or task")
    sc.add_argument("path", help="local dataset folder, or owner/name on the Hugging Face Hub")
    sc.add_argument("--group-by", default="task", help="episodes-metadata column to group by (default: task)")
    sc.add_argument("--groups", help="CSV with episode_index,group columns (overrides --group-by)")
    sc.add_argument("--max-mb", type=float, default=2000, help="refuse to fetch a Hub dataset larger than this")
    sc.add_argument("--no-scores", action="store_true", help="skip the literature-based scores (faster)")
    sc.add_argument("--json", help="write the scorecard as JSON")
    sc.set_defaults(fn=_cmd_scorecard)

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
