"""Benchmark rdq check-video on a synthetic delivery with known defects, copies and contributor profiles.

Five contributors, each with a typical failure pattern of paid collection:
  careful      clean footage
  padder       long idle stretches (camera set down while paid by the hour)
  reuploader   re-encoded, cut and mirrored copies of other contributors' clips
  bad_lens     persistent blur (dirty or fogged lens) and some covered-lens time
  flaky_device stalls, black frames, timestamp gaps, a truncated file

Measures, against ground truth: detection recall/precision per defect type (file level, with the
right time), duplicate-pair recall and false duplicate pairs, error of the usable-hours estimate,
and whether the contributor ranking matches the true usable share.

    python experiments/video_benchmark.py --out results/video_benchmark --clips 8 --seconds 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdq import synthvideo as sv  # noqa: E402
from rdq.rawvideo import check_video_batch  # noqa: E402

DEFECT_CHECK = {"freeze": "video.frozen", "black": "video.black", "covered": "video.covered",
                "blur": "video.blur", "idle": "video.idle", "drop": "video.recording_gap"}


def build(root: Path, clips: int, seconds: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    truth = {}
    k = 0

    def clip(contrib, defects=None, **kw):
        nonlocal k
        k += 1
        p = root / contrib / f"{contrib}_{k:03d}.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        sv.render_clip(p, sv.ClipSpec(seconds=seconds, seed=1000 * seed + k, defects=defects or {}, **kw))
        truth[str(p)] = {"contributor": contrib, "defects": defects or {}, "copy_of": None, "seconds": seconds}
        return p

    def span(lo, hi, minlen, maxlen):
        a = float(rng.uniform(lo, hi - maxlen))
        return (round(a, 1), round(a + float(rng.uniform(minlen, maxlen)), 1))

    originals = []
    for _ in range(clips):
        originals.append(clip("careful"))
    for _ in range(clips):
        clip("padder", {"idle": span(1, seconds - 1, 6, seconds * 0.6)})
    for _ in range(clips):
        d = {"blur": span(1, seconds - 1, 4, 8)}
        if rng.random() < 0.5:
            d["covered"] = span(1, seconds - 1, 1.5, 3)
            if d["covered"][0] < d["blur"][1] and d["blur"][0] < d["covered"][1]:
                d.pop("covered")
        clip("bad_lens", d)
    for j in range(clips):
        kind = ["freeze", "black", "drop"][j % 3]
        clip("flaky_device", {kind: span(1, seconds - 1, 1.5, 3)})
    # truncated file from the flaky device
    p = clip("flaky_device")
    sv.faststart(p, root / "tmp_fs.mp4")
    sv.truncate(root / "tmp_fs.mp4", p, keep=0.5)
    (root / "tmp_fs.mp4").unlink()
    truth[str(p)]["truncated"] = True
    # reuploader: copies of careful clips
    (root / "reuploader").mkdir(exist_ok=True)
    for j, src in enumerate(originals[: clips]):
        dst = root / "reuploader" / f"reuploader_{j:03d}.mp4"
        mode = j % 4
        if mode == 0:
            sv.reencode(src, dst, width=426, crf=32)
        elif mode == 1:
            sv.reencode(src, dst, trim=(seconds * 0.2, seconds * 0.8), crf=28)
        elif mode == 2:
            sv.reencode(src, dst, mirror=True, brightness=0.06, crf=30)
        else:
            sv.reencode(src, dst, fps=25, width=480, crf=30)
        truth[str(dst)] = {"contributor": "reuploader", "defects": {}, "copy_of": str(src),
                           "seconds": seconds * (0.6 if mode == 1 else 1.0), "copy_mode": ["reencode", "cut", "mirror", "fps25"][mode]}
    return truth


def true_usable(t: dict, decoded_s: float | None = None) -> float:
    if t.get("copy_of"):
        return 0.0
    if t.get("truncated"):       # the part that decodes is fine footage; nothing after the cut exists
        return decoded_s or 0.0
    bad = [v for k, v in t["defects"].items() if k != "drop"]
    from rdq.rawvideo import _span_len, _union

    dur = t["seconds"] - sum(b - a for k, (a, b) in t["defects"].items() if k == "drop")
    return max(0.0, dur - _span_len(_union(bad)))


def evaluate(report: dict, truth: dict) -> dict:
    files = {f["path"]: f for f in report["files"]}
    res = {}
    for d, check in DEFECT_CHECK.items():
        tp = fn = fp = 0
        for path, t in truth.items():
            if t.get("copy_of") or t.get("truncated"):
                continue
            f = files[path]
            hits = [i for i in f["issues"] if i["check"] == check]
            if d in t["defects"]:
                a, b = t["defects"][d]
                ok = any(any(sa < b + 1.5 and a - 1.5 < sb for sa, sb in i["spans"]) or (d == "drop" and i)
                         for i in hits)
                tp += ok
                fn += not ok
            elif hits and not (d == "freeze" and "drop" in t["defects"]):
                fp += 1
        res[d] = {"recall": tp / max(1, tp + fn), "false_positive_files": fp, "n": tp + fn}
    # truncation
    tr = [p for p, t in truth.items() if t.get("truncated")]
    res["truncated"] = {"recall": float(np.mean([any(i["check"] == "video.truncated" for i in files[p]["issues"])
                                                 for p in tr])) if tr else None, "n": len(tr)}
    # duplicates
    copies = {p: t for p, t in truth.items() if t.get("copy_of")}
    found = {}
    for d in report["duplicates"]:
        found[(d["a"], d["b"])] = d["kind"]
        found[(d["b"], d["a"])] = d["kind"]
    per_mode = {}
    for p, t in copies.items():
        hit = (t["copy_of"], p) in found
        per_mode.setdefault(t["copy_mode"], []).append(hit)
    copy_pairs = {(t["copy_of"], p) for p, t in copies.items()}
    false_pairs = [d for d in report["duplicates"] if d["kind"] != "loop_within_file"
                   and (d["a"], d["b"]) not in copy_pairs and (d["b"], d["a"]) not in copy_pairs
                   and not (truth.get(d["a"], {}).get("copy_of") and truth.get(d["b"], {}).get("copy_of")
                            and truth[d["a"]]["copy_of"] == truth[d["b"]]["copy_of"])]
    res["duplicates"] = {"recall": float(np.mean([h for v in per_mode.values() for h in v])) if copies else None,
                         "by_copy_mode": {k: float(np.mean(v)) for k, v in per_mode.items()},
                         "false_pairs": len(false_pairs), "false_pair_examples": [(Path(d["a"]).name, Path(d["b"]).name, d["kind"])
                                                                                  for d in false_pairs[:5]],
                         "loops_reported": sum(d["kind"] == "loop_within_file" for d in report["duplicates"])}
    # usable hours
    est = {p: files[p]["usable_s"] for p in truth}
    tru = {p: true_usable(t, files[p]["duration_s"]) for p, t in truth.items()}
    err = [est[p] - tru[p] for p in truth]
    tot_t, tot_e = sum(tru.values()), sum(est.values())
    res["usable"] = {"true_hours": round(tot_t / 3600, 4), "estimated_hours": round(tot_e / 3600, 4),
                     "total_error_pct": round(100 * (tot_e - tot_t) / max(1e-9, tot_t), 1),
                     "mean_abs_error_s_per_file": round(float(np.mean(np.abs(err))), 2)}
    # contributor ranking
    contrib_true = {}
    for p, t in truth.items():
        c = contrib_true.setdefault(t["contributor"], [0.0, 0.0])
        c[0] += tru[p]
        c[1] += t["seconds"]
    true_share = {c: v[0] / v[1] for c, v in contrib_true.items()}
    est_share = {r["contributor"]: r["usable_share"] for r in report["scorecard"]}
    names = sorted(true_share)
    from scipy.stats import spearmanr

    rho = spearmanr([true_share[n] for n in names], [est_share.get(n, 0) for n in names]).statistic
    res["contributors"] = {"true_usable_share": {n: round(true_share[n], 3) for n in names},
                           "estimated_usable_share": {n: round(est_share.get(n, 0), 3) for n in names},
                           "spearman": round(float(rho), 3),
                           "worst_estimated": report["scorecard"][0]["contributor"],
                           "worst_true": min(true_share, key=true_share.get)}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/video_benchmark")
    ap.add_argument("--clips", type=int, default=8, help="clips per contributor")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    root = out / "delivery"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True)
    t0 = time.time()
    truth = build(root, args.clips, args.seconds, args.seed)
    t1 = time.time()
    report = check_video_batch([str(root)], by_folder=True, workers=args.workers)
    t2 = time.time()
    res = evaluate(report, truth)
    res["setup"] = {"files": len(truth), "clips_per_contributor": args.clips, "seconds_per_clip": args.seconds,
                    "seed": args.seed, "render_s": round(t1 - t0, 1), "analysis_s": round(t2 - t1, 1),
                    "analysis_s_per_video_minute": round((t2 - t1) / (sum(t["seconds"] for t in truth.values()) / 60), 2),
                    "workers": args.workers}
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
