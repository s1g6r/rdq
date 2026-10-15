"""Many takes of one task from one fixed camera, plus three copies: does duplicate detection stay correct?

Every take shares the background, so frame hashes match across all of them (the hash index saturates);
only small moving objects differ. Copies: a half-size re-encode, a mirrored cut starting off the 2 fps
grid, and a 25 fps conversion.

    python experiments/video_scale.py OUTDIR 80 0.7     # 80 takes, object size 0.7 (default 0.35: very small)
"""
import sys, time, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdq import synthvideo as sv
from rdq.rawvideo import check_video_batch
from concurrent.futures import ProcessPoolExecutor

def render(args):
    path, seed = args
    sv.render_clip(path, sv.ClipSpec(seconds=12, seed=seed, scene_seed=5, static_camera=True, object_scale=float(sys.argv[3]) if len(sys.argv)>3 else 0.35, width=480, height=270))
    return path

def main():
    d = Path(sys.argv[1]); n = int(sys.argv[2])
    if not (d / "op").exists():
        (d / "op").mkdir(parents=True); (d / "cp").mkdir()
        with ProcessPoolExecutor(2) as ex:
            list(ex.map(render, [(d / "op" / f"take{k:03d}.mp4", 100 + k) for k in range(n)]))
        sv.reencode(d / "op" / "take005.mp4", d / "cp" / "c1.mp4", width=320, crf=32)
        sv.reencode(d / "op" / "take017.mp4", d / "cp" / "c2.mp4", trim=(2.3, 10.9), mirror=True)
        sv.reencode(d / "op" / "take029.mp4", d / "cp" / "c3.mp4", fps=25)
    t = time.time()
    r = check_video_batch([str(d)], by_folder=True, workers=2, debug=True)
    el = time.time() - t
    pairs = r["debug"]["pairs"]
    print(f"{n} takes; {len(pairs)} candidate pairs aligned; analysis {el:.0f}s")
    for x in r["duplicates"]:
        print("DUP", x["kind"], Path(x["a"]).name, Path(x["b"]).name)
    rej = sorted(p["motion_corr"] for p in pairs if not p["verified"])
    print("rejected max motion_corr", rej[-3:] if rej else None, "accepted", sorted(p["motion_corr"] for p in pairs if p["verified"]))
    for p in pairs:
        if p["motion_corr"] > 0.6:
            print("  pair", Path(p["a"]).name, Path(p["b"]).name, p["verified"], {k: p[k] for k in ("overlap_s","density","norm_diff","motion_corr","motion","foreground_corr")})
    print("findings:", sum(1 for f in r["files"] if f["status"] != "ok"))

if __name__ == "__main__":
    main()
