"""Run check-video on real robot camera footage from public LeRobot datasets.

Downloads a few episode videos per dataset (each dataset = one "contributor"), adds two copies
made from one real episode (a re-encode and a mirrored cut, put under another contributor), then
runs rdq check-video. Prints a compact summary, and emits GitHub annotations so the result can
be read without downloading artifacts.

    python experiments/video_real.py --out vreal
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATASETS = [
    "lerobot/aloha_static_coffee",
    "lerobot/droid_100",
    "lerobot/svla_so101_pickplace",
    "lerobot/aloha_mobile_cabinet",
    "lerobot/berkeley_autolab_ur5",
    "lerobot/pusht",
]


def fetch(repo: str, dest: Path, n: int, max_seconds: float) -> list[Path]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    out = []
    for rev in ("v2.1", None):
        try:
            files = api.list_repo_files(repo, repo_type="dataset", revision=rev)
        except Exception as exc:  # noqa: BLE001
            print(f"  {repo}@{rev}: {type(exc).__name__}", file=sys.stderr)
            continue
        vids = sorted(f for f in files if f.endswith(".mp4"))
        if not vids:
            continue
        cam = vids[0].split("/")[-2]               # one camera
        vids = [v for v in vids if v.split("/")[-2] == cam][:n]
        for v in vids:
            p = Path(hf_hub_download(repo, v, repo_type="dataset", revision=rev))
            target = dest / f"{repo.split('/')[1]}__{Path(v).stem}.mp4"
            if p.stat().st_size > 60e6 or "file-" in p.name:   # v3 concatenated chunk: cut into pieces
                for k in range(n):
                    seg = dest / f"{repo.split('/')[1]}__seg{k}.mp4"
                    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(k * max_seconds), "-t", str(max_seconds),
                                    "-i", str(p), "-c", "copy", str(seg)], check=False)
                    if seg.exists() and seg.stat().st_size > 1000:
                        out.append(seg)
                return out
            shutil.copy(p, target)
            out.append(target)
        if out:
            return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vreal")
    ap.add_argument("--per-dataset", type=int, default=6)
    args = ap.parse_args()
    root = Path(args.out) / "delivery"
    root.mkdir(parents=True, exist_ok=True)
    got = {}
    for repo in DATASETS:
        d = root / repo.split("/")[1]
        d.mkdir(exist_ok=True)
        try:
            got[repo] = fetch(repo, d, args.per_dataset, 30.0)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
            got[repo] = []
        print(f"{repo}: {len(got[repo])} files", file=sys.stderr)
    # two known copies of one real episode, filed under another "contributor"
    src = next((fs[1] for fs in got.values() if len(fs) > 1), None)
    if src:
        cp = root / "copier"
        cp.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", "scale=iw/2:-2", "-c:v", "libx264",
                        "-crf", "32", str(cp / "reupload.mp4")], check=False)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(src)], capture_output=True, text=True, check=False).stdout.strip() or 0)
        if dur > 12:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(dur * 0.2), "-to", str(dur * 0.8), "-i", str(src),
                            "-vf", "hflip,eq=brightness=0.04", "-c:v", "libx264", "-crf", "28", str(cp / "cut.mp4")],
                           check=False)
        print(f"copies made from {src.name}", file=sys.stderr)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rdq.rawvideo import check_video_batch, format_terminal, write_outputs

    t0 = time.time()
    rep = check_video_batch([str(root)], by_folder=True, workers=2)
    elapsed = time.time() - t0
    write_outputs(rep, json_path=str(Path(args.out) / "report.json"), html_path=str(Path(args.out) / "report.html"))
    print(format_terminal(rep))
    s = rep["summary"]
    print(f"\nanalysis time {elapsed:.1f} s for {s['hours'] * 60:.1f} min of video "
          f"({elapsed / max(1e-9, s['hours'] * 60):.2f} s per minute, 2 workers)")
    lines = [f"files={s['files']} min={s['hours'] * 60:.1f} usable={s['usable_share']:.2f} "
             f"crit={s['files_critical']} warn={s['files_warning']} sec_per_min={elapsed / max(1e-9, s['hours'] * 60):.2f}"]
    for f in rep["files"]:
        checks = ",".join(sorted({i["check"].replace("video.", "") + ("!" if i["severity"] == "critical" else "")
                                  for i in f["issues"] if i["severity"] != "info"}))
        m = f["meta"]
        lines.append(f"{Path(f['path']).parent.name}/{Path(f['path']).name[-28:]} {m.get('width')}x{m.get('height')} "
                     f"{f['usable_s']:.0f}/{f['duration_s']:.0f}s {checks or 'ok'}")
    for d in rep["duplicates"]:
        lines.append(f"DUP {d['kind']} {Path(d['a']).name} -> {Path(d['b']).name} {d.get('overlap_s')}")
    text = "\n".join(lines)
    (Path(args.out) / "summary.txt").write_text(text)
    chunk, k = [], 0
    for ln in lines:
        chunk.append(ln)
        if sum(len(x) + 3 for x in chunk) > 3000:
            print(f"::notice title=rdq-video-{k}::" + " %0A".join(chunk))
            chunk, k = [], k + 1
    if chunk:
        print(f"::notice title=rdq-video-{k}::" + " %0A".join(chunk))
    json.dumps(s, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
