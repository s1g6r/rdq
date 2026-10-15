"""Checks for raw video deliveries (egocentric or teleop camera footage), before it becomes a dataset.

    rdq check-video delivery/ --by-folder --html report.html --json report.json
    rdq check-video clips/*.mp4 --contributors contributors.csv --spec spec.yaml --store hashes.json

Collectors pay contributors per accepted hour of footage, and buyers pay per hour too. This module
measures how much of a delivery is actually usable and why the rest is not, per file and per
contributor:

  integrity   file can't be opened, has no video, decodes to nothing, is truncated (decodes shorter
              than its header says) or has corrupt frames
  timing      dropped frames and recording gaps, variable frame rate, non-monotonic timestamps
  content     black or covered lens, over-exposure, frozen camera (bit-identical frames), static scene
              with nothing moving (idle footage), persistent blur (defocus, dirty lens), low light
  metadata    values outside a delivery spec (fps, resolution, duration, codec), and outliers
              against the rest of the batch
  duplicates  byte-identical files, remuxes of the same stream, re-encoded or trimmed copies and
              partial overlaps (also across contributors and against earlier batches), and footage
              that loops within one file

Everything runs on CPU with PyAV, OpenCV and numpy (pip install "rdq[video]"); ffprobe is used when
available. Frames are decoded once, downscaled to 320 px wide, and sampled on a fixed 2 fps grid for
perceptual hashes and motion, so a minute of 720p costs a few seconds. Thresholds come from synthetic
and public test clips (see docs/raw-video.md); calibrate them on a few real files per device before
using any finding as a hard rejection rule.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".ts"}

ANALYSIS_WIDTH = 320          # px; every content measure is computed at this width so files compare
IDENT_WIDTH = 640             # px; width for the bit-identical (frozen) test
SAMPLE_FPS = 2.0              # fixed time grid for hashes and motion
BLUR_EVERY = 5                # Laplacian on every 5th frame
DARK_LEVEL = 26               # grey level counted as black (about 10% of range, full or limited)
BLACK_FRAC = 0.98             # share of dark pixels that makes a frame black
COVERED_STD = 8.0             # low detail: flat image ...
COVERED_LAP = 20.0            # ... with almost no edges
OVER_MEAN, OVER_STD = 235.0, 10.0
LOW_LIGHT_MEAN = 45.0
BLUR_REL = 0.25               # soft: Laplacian variance below this share of the camera's typical value ...
BLUR_RATIO_REL = 0.8          # ... and fine detail relative to edges (Laplacian/gradient energy) below this share
# Sharpness is judged against the same camera's (contributor and resolution) typical footage, never a fixed
# number: a white wall, a smooth towel or an upscaled low-resolution camera has little fine detail yet is in
# focus. The detail-to-edge ratio does not depend on how much texture is in view, so both must drop.
IDLE_CHANGE = 2.0             # grey levels: 98th-percentile change of the 32x18 thumbnail over 1 s below which nothing moves
WINDOW_S = 10.0               # idle / blur windows
HASH_THR = 10                 # Hamming distance at which two 64-bit pHashes count as the same frame
HASH_MIN_Q = 8.0              # skip flat samples (black, covered) when matching: they match everything
MIN_DUP_RUN_S = 5.0
LOOP_MIN_OFFSET_S = 30.0

CRITICAL, WARNING, INFO = "critical", "warning", "info"


@dataclass
class Issue:
    check: str
    severity: str
    message: str
    seconds: float = 0.0
    spans: list = field(default_factory=list)   # [(start_s, end_s), ...]

    def to_dict(self) -> dict:
        return {"check": self.check, "severity": self.severity, "message": self.message,
                "seconds": round(self.seconds, 2), "spans": [[round(a, 2), round(b, 2)] for a, b in self.spans[:20]]}


@dataclass
class Spec:
    """Delivery requirements; every field is optional."""

    min_duration_s: float | None = None
    max_duration_s: float | None = None
    fps: float | None = None
    fps_tolerance: float = 0.5
    min_width: int | None = None
    min_height: int | None = None
    codecs: list[str] | None = None
    max_idle_share: float = 0.15
    max_blur_share: float = 0.10

    @classmethod
    def load(cls, path: str | Path | None) -> Spec:
        if not path:
            return cls()
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(f"unknown spec keys: {', '.join(unknown)}")
        return cls(**known)


# --------------------------------------------------------------------------------------------- #
# Collecting files and contributors


def find_videos(paths: list[str]) -> tuple[list[Path], Path | None]:
    """Video files under the given files/folders, and the common root (for --by-folder)."""
    out: list[Path] = []
    roots = []
    for p in map(Path, paths):
        if p.is_dir():
            roots.append(p)
            out += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
                          and not f.name.startswith("."))
        elif p.is_file():
            out.append(p)
    root = roots[0] if len(roots) == 1 else None
    return list(dict.fromkeys(out)), root


def load_contributors(csv_path: str | Path) -> dict[str, str]:
    """Map file name or relative path -> contributor, from a CSV with 'file' and 'contributor' columns."""
    mapping: dict[str, str] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}
        fcol = next((cols[c] for c in ("file", "path", "filename", "video") if c in cols), None)
        ccol = next((cols[c] for c in ("contributor", "collector", "operator", "worker", "partner", "device")
                     if c in cols), None)
        if not fcol or not ccol:
            raise ValueError("contributors CSV needs a 'file' column and a 'contributor' (or collector/operator) column")
        for row in reader:
            if row.get(fcol) and row.get(ccol):
                mapping[row[fcol].strip()] = row[ccol].strip()
    return mapping


def contributor_of(path: Path, root: Path | None, mapping: dict[str, str] | None, by_folder: bool) -> str:
    if mapping:
        rel = str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name
        for key in (rel, path.name, str(path)):
            if key in mapping:
                return mapping[key]
        return "(unmapped)"
    if by_folder and root and path.is_relative_to(root):
        parts = path.relative_to(root).parts
        return parts[0] if len(parts) > 1 else "(root)"
    return "(all)"


# --------------------------------------------------------------------------------------------- #
# Per-file measurement


def _ffprobe(path: Path) -> dict | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=60, check=False)
        return json.loads(out.stdout) if out.stdout.strip() else None
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None


def _ratio(s) -> float | None:
    try:
        if s in (None, "", "0/0"):
            return None
        if "/" in str(s):
            a, b = str(s).split("/")
            return float(a) / float(b) if float(b) else None
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: Path) -> dict:
    """Container and stream metadata (ffprobe if present, else PyAV)."""
    meta: dict = {"size_bytes": path.stat().st_size}
    info = _ffprobe(path)
    if info:
        fmt = info.get("format", {})
        vids = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
                and not (s.get("disposition") or {}).get("attached_pic")]
        meta["container"] = fmt.get("format_name")
        meta["header_duration_s"] = _ratio(fmt.get("duration"))
        meta["bit_rate"] = _ratio(fmt.get("bit_rate"))
        meta["has_audio"] = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        meta["video_streams"] = len(vids)
        tags = {**(fmt.get("tags") or {})}
        meta["encoder"] = tags.get("encoder")
        meta["device"] = tags.get("com.apple.quicktime.model") or tags.get("model") or tags.get("make")
        if vids:
            v = vids[0]
            rotation = 0
            for sd in v.get("side_data_list") or []:
                if "rotation" in sd:
                    rotation = int(float(sd["rotation"]))
            rotation = rotation or int(float((v.get("tags") or {}).get("rotate", 0) or 0))
            meta.update(codec=v.get("codec_name"), width=v.get("width"), height=v.get("height"),
                        pix_fmt=v.get("pix_fmt"), fps=_ratio(v.get("r_frame_rate")),
                        avg_fps=_ratio(v.get("avg_frame_rate")), nb_frames=_ratio(v.get("nb_frames")),
                        field_order=v.get("field_order"), rotation=rotation,
                        hdr=v.get("color_transfer") in ("arib-std-b67", "smpte2084"))
            if meta.get("header_duration_s") is None:
                meta["header_duration_s"] = _ratio(v.get("duration"))
        return meta
    import av

    try:
        with av.open(str(path)) as c:
            meta["container"] = c.format.name
            meta["header_duration_s"] = c.duration / 1e6 if c.duration else None
            meta["has_audio"] = bool(c.streams.audio)
            meta["video_streams"] = len(c.streams.video)
            if c.streams.video:
                s = c.streams.video[0]
                meta.update(codec=s.codec_context.name, width=s.codec_context.width, height=s.codec_context.height,
                            pix_fmt=s.codec_context.pix_fmt, fps=float(s.base_rate) if s.base_rate else None,
                            avg_fps=float(s.average_rate) if s.average_rate else None,
                            nb_frames=s.frames or None, rotation=0, hdr=False)
    except Exception as exc:  # noqa: BLE001
        meta["probe_error"] = str(exc)[:200]
    return meta


def timestamps(path: Path) -> dict:
    """Packet timestamps from demuxing (no decode): drops, gaps, VFR, non-monotonic PTS, and a hash of
    the coded video stream (equal for remuxes of the same recording)."""
    import av

    h = hashlib.blake2b(digest_size=16)
    try:
        with av.open(str(path)) as c:
            if not c.streams.video:
                return {}
            s = c.streams.video[0]
            tb = float(s.time_base)
            nominal = float(s.base_rate or s.average_rate or 0) or None
            pts = []
            for p in c.demux(s):
                if p.pts is not None:
                    pts.append(p.pts)
                if p.size:
                    h.update(bytes(p))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:200]}
    if len(pts) < 3:
        return {"stream_hash": h.hexdigest(), "packets": len(pts)}
    # decode order differs from display order with B-frames, so only sorted timestamps are meaningful
    t = np.sort(np.array(pts, dtype=np.float64) * tb)
    d = np.diff(t)
    dup = int((d <= 0).sum())
    d = d[d > 0]
    if not len(d):
        return {"stream_hash": h.hexdigest(), "packets": len(pts), "duplicate_pts": dup}
    med = float(np.median(d))
    gaps = d > 1.5 * med
    big = d > 1.0
    # runs of a different but constant interval are a frame-rate change, not drops
    rounded = np.round(d * 1000).astype(int)
    common = Counter(rounded.tolist()).most_common(3)
    vfr = len(common) > 1 and common[1][1] >= max(10, 0.1 * len(d)) and abs(common[1][0] - common[0][0]) > 1
    gap_spans = [(float(t[i]), float(t[i + 1])) for i in np.nonzero(gaps)[0][:50]]
    return {
        "stream_hash": h.hexdigest(), "packets": len(pts), "nominal_fps": nominal,
        "median_fps": 1.0 / med, "dropped_frames_est": int(np.round(d[gaps & ~big] / med - 1).sum()) if not vfr else 0,
        "gaps": int((gaps & ~big).sum()) if not vfr else 0, "recording_gaps": int(big.sum()),
        "recording_gap_s": float(d[big].sum()), "gap_spans": gap_spans, "vfr": bool(vfr),
        "duplicate_pts": dup,
        "jitter_cv": float(np.std(d[~gaps]) / med) if (~gaps).any() else 0.0,
    }


def _phash64(g) -> int:
    import cv2

    s = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(s)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int(np.packbits(bits).view(">u8")[0])


def decode_pass(path: Path) -> dict:
    """One decode of the whole file. Per frame: brightness, detail, change, bit-identical repeats,
    corrupt flags. Every 5th frame: sharpness. On a 2 fps grid: perceptual hash (and of the mirrored
    frame), detail, a 32x18 thumbnail, and consecutive-frame optical flow (camera vs scene motion)."""
    import av
    import cv2

    out = {"frames": 0, "corrupt_frames": 0, "decode_errors": 0}
    try:
        c = av.open(str(path))
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0]
        if "Invalid data" in msg or "moov" in msg:
            msg = "invalid or incomplete container (often a recording that was cut off before it was finalised)"
        out["open_error"] = msg[:200]
        return out
    if not c.streams.video:
        c.close()
        out["open_error"] = "no video stream"
        return out
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    tb = float(s.time_base) if s.time_base else None
    t_list, mean_l, std_l, dark_l, mad_l, ident_l, exact_l = [], [], [], [], [], [], []
    lap_t, lap_v, lapr_v = [], [], []
    samp = defaultdict(list)
    prev = prev_id = prev_small = None
    t0 = None
    t_next = 0.0
    it = c.decode(s)
    i = 0
    while True:
        try:
            fr = next(it)
        except StopIteration:
            break
        except av.error.FFmpegError:
            out["decode_errors"] += 1
            if out["decode_errors"] > 200:
                break
            continue
        except Exception:  # noqa: BLE001
            out["decode_errors"] += 1
            break
        if fr.pts is not None and tb:
            t = fr.pts * tb
        else:
            t = (t_list[-1] + 1 / 30) if t_list else 0.0
        t0 = t if t0 is None else t0
        t -= t0
        w, hgt = fr.width, fr.height
        aw = max(16, min(ANALYSIS_WIDTH, w - w % 2))   # never upscale: a small frame upscaled looks soft
        H = max(2, int(round(hgt * aw / w / 2) * 2))
        g = fr.reformat(width=aw, height=H, format="gray").to_ndarray()
        iw = min(IDENT_WIDTH, w)
        ih = max(2, int(round(hgt * iw / w / 2) * 2))
        gid = fr.reformat(width=iw, height=ih, format="gray").to_ndarray() if iw != aw else g
        out["corrupt_frames"] += int(bool(getattr(fr, "is_corrupt", False)))
        t_list.append(t)
        mean_l.append(float(g.mean()))
        std_l.append(float(g.std()))
        dark_l.append(float((g < DARK_LEVEL).mean()))
        g16 = g.astype(np.int16)
        mad_l.append(float(np.abs(g16 - prev).mean() / 255) if prev is not None else np.nan)
        # a repeated frame: the same picture re-encoded comes back with tiny refinements on a handful of
        # pixels, so "identical" means no more than 0.05% of pixels differ by more than 2 grey levels
        # (sensor noise alone changes far more than that between two real frames)
        same = prev_id is not None and prev_id.shape == gid.shape
        dif = cv2.absdiff(gid, prev_id) if same else None
        ident_l.append(bool(same and np.count_nonzero(dif > 2) <= 0.0005 * gid.size))
        exact_l.append(bool(same and not dif.any()))
        if i % BLUR_EVERY == 0:
            lap_t.append(t)
            lv = float(cv2.Laplacian(g, cv2.CV_64F).var())
            gf = g.astype(np.float32)
            ge = float((cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3) ** 2).mean()) / 16
            lap_v.append(lv)
            lapr_v.append(lv / ge if ge > 1e-3 else np.nan)
        if t + 1e-6 >= t_next:
            hsm = cv2.resize(g, (256, 144), interpolation=cv2.INTER_AREA)
            sm = cv2.resize(g, (160, max(2, 160 * H // aw)), interpolation=cv2.INTER_AREA)
            fg = fr_ = np.nan
            if prev_small is not None and prev_small.shape == sm.shape:
                f = cv2.calcOpticalFlowFarneback(prev_small, sm, None, 0.5, 3, 15, 3, 5, 1.2, 0).reshape(-1, 2)
                med = np.median(f, 0)
                fg = float(np.hypot(*med))
                fr_ = float(np.percentile(np.linalg.norm(f - med, axis=1), 90))
            samp["t"].append(t)
            samp["h"].append(_phash64(hsm))
            samp["hf"].append(_phash64(hsm[:, ::-1]))
            samp["q"].append(float(hsm.std()))
            samp["thumb"].append(cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA))
            samp["flow_g"].append(fg)
            samp["flow_r"].append(fr_)
            samp["mad"].append(mad_l[-1])
            while t_next <= t + 1e-6:
                t_next += 1.0 / SAMPLE_FPS
        prev = g16
        prev_id = gid
        prev_small = cv2.resize(g, (160, max(2, 160 * H // aw)), interpolation=cv2.INTER_AREA)
        i += 1
    c.close()
    out["frames"] = i
    out["t"] = np.array(t_list)
    out["mean"] = np.array(mean_l)
    out["std"] = np.array(std_l)
    out["dark"] = np.array(dark_l)
    out["mad"] = np.array(mad_l)
    out["ident"] = np.array(ident_l, dtype=bool)
    out["exact"] = np.array(exact_l, dtype=bool)
    out["lap_t"] = np.array(lap_t)
    out["lap"] = np.array(lap_v)
    out["lapr"] = np.array(lapr_v)
    out["samples"] = {k: (np.array(v, dtype=np.uint64) if k in ("h", "hf") else
                          (np.stack(v) if k == "thumb" and v else np.array(v))) for k, v in samp.items()}
    out["decoded_duration_s"] = float(t_list[-1] + (np.median(np.diff(t_list)) if len(t_list) > 1 else 0)) if t_list else 0.0
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runs(times: np.ndarray, flags: np.ndarray, min_len: float, dt: float) -> list[tuple[float, float]]:
    """Spans where flags hold for at least min_len seconds (each frame covers dt seconds)."""
    out, start = [], None
    for t, f in zip(times, flags):
        if f and start is None:
            start = t
        if not f and start is not None:
            if t - start >= min_len:
                out.append((float(start), float(t)))
            start = None
    if start is not None and len(times) and times[-1] + dt - start >= min_len:
        out.append((float(start), float(times[-1] + dt)))
    return out


def _span_len(spans) -> float:
    return float(sum(b - a for a, b in spans))


def _union(*span_lists) -> list[tuple[float, float]]:
    allspans = sorted(s for lst in span_lists for s in lst)
    out: list[list[float]] = []
    for a, b in allspans:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def measure_file(path: str) -> dict:
    """Everything rdq measures about one file (picklable, so files can run in parallel)."""
    p = Path(path)
    rec: dict = {"path": str(p), "name": p.name}
    try:
        rec["sha256"] = file_sha256(p)
        rec["meta"] = probe(p)
        rec["ts"] = timestamps(p)
        rec["dec"] = decode_pass(p)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop a batch
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


# --------------------------------------------------------------------------------------------- #
# Turning measurements into findings


def _frozen_runs(t, near, exact, dt) -> list[tuple[float, float]]:
    """Runs of repeated frames. A stalled camera's frames come back bit-identical much of the time (the
    encoder only refines a few pixels); a still scene recorded live is near-identical after compression
    but almost never bit-identical, and is reported as idle instead. So a run of near-identical frames
    counts as frozen when at least a quarter of its frames are exact repeats."""
    spans = runs(t, near, 1.0, dt)
    if exact is None or not len(exact):
        return spans
    keep = []
    for a, b in spans:
        m = (t >= a) & (t < b)
        if m.any() and exact[m].mean() >= 0.25:
            keep.append((a, b))
    return keep


def content_spans(dec: dict) -> dict:
    """Black, covered, over-exposed, frozen and low-light spans from per-frame stats."""
    t = dec.get("t")
    if t is None or not len(t):
        return {}
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    lap_i = np.interp(t, dec["lap_t"], dec["lap"]) if len(dec.get("lap", [])) else np.full(len(t), 1e9)
    black_f = dec["dark"] >= BLACK_FRAC
    covered_f = (dec["std"] < COVERED_STD) & (lap_i < COVERED_LAP) & ~black_f
    over_f = (dec["mean"] > OVER_MEAN) & (dec["std"] < OVER_STD)
    flat = black_f | covered_f | over_f
    ident = dec["ident"] & ~flat
    return {
        "dt": dt,
        "black": runs(t, black_f, 0.5, dt),
        "covered": runs(t, covered_f, 1.0, dt),
        "overexposed": runs(t, over_f, 1.0, dt),
        "frozen": _frozen_runs(t, ident, dec.get("exact"), dt),
        "low_light_share": float(np.mean((dec["mean"] < LOW_LIGHT_MEAN) & ~flat)),
        "flat_frames": flat,
    }


def change_1s(sm: dict) -> np.ndarray:
    """How much the picture changes over one second, per 2 fps sample: the 98th percentile of the absolute
    difference between the 32x18 thumbnail and the one a second earlier, after removing any global
    brightness shift. Averaging into thumbnails removes sensor noise and compression artefacts; a moving
    arm, hand or object changes some cells by tens of grey levels even when it is small in the frame."""
    t = sm.get("t")
    th = sm.get("thumb")
    out = np.full(len(t) if t is not None else 0, np.nan)
    if t is None or th is None or len(t) < 2:
        return out
    th = th.astype(np.float32)
    j = np.clip(np.searchsorted(t, t - 1.0), 0, len(t) - 1)
    for k in range(len(t)):
        jj = j[k]
        if jj > 0 and abs(t[jj - 1] - (t[k] - 1.0)) < abs(t[jj] - (t[k] - 1.0)):
            jj -= 1
        if jj >= k or abs(t[jj] - (t[k] - 1.0)) > 0.3:
            continue
        d = th[k] - th[jj]
        d -= np.median(d)
        out[k] = float(np.percentile(np.abs(d), 98))
    return out


def windowed(dec: dict, spans: dict, base: dict | None) -> dict:
    """Idle (static scene) and persistent blur from the 2 fps samples, each judged on a rolling
    4 s neighbourhood, kept when they last at least 5 s (idle) or 3 s (blur)."""
    sm = dec.get("samples") or {}
    t = sm.get("t")
    if t is None or not len(t):
        return {"idle": [], "blur": [], "motion_blur_s": 0.0}
    flat_spans = _union(spans.get("black", []), spans.get("covered", []), spans.get("overexposed", []),
                        spans.get("frozen", []))
    in_flat = np.array([any(a <= x < b for a, b in flat_spans) for x in t], dtype=bool)
    lap_t, lap = dec.get("lap_t", np.array([])), dec.get("lap", np.array([]))
    lapr = dec.get("lapr", np.full(len(lap), np.nan))
    base = base or {}
    thr = BLUR_REL * base["lap"] if base.get("lap") else 0.0
    rthr = BLUR_RATIO_REL * base["ratio"] if base.get("ratio") else np.inf
    soft_frame = (lap < thr) & ~(np.nan_to_num(lapr, nan=np.inf) >= rthr) if len(lap) else np.zeros(0, bool)
    soft_at = (np.interp(t, lap_t, soft_frame.astype(float)) > 0.5) if len(lap) else np.zeros(len(t), bool)
    fg = sm["flow_g"]
    chg = change_1s(sm)
    static = np.zeros(len(t), bool)
    soft = np.zeros(len(t), bool)
    moving = np.zeros(len(t), bool)
    half = 2.0
    for k in range(len(t)):
        m = (np.abs(t - t[k]) <= half) & ~in_flat
        if m.sum() < 3:
            continue
        g_ = np.nanmedian(fg[m]) if np.isfinite(fg[m]).any() else np.nan
        c_ = np.nanmedian(chg[m]) if np.isfinite(chg[m]).any() else np.nan
        static[k] = bool(np.isfinite(c_) and c_ < IDLE_CHANGE)
        soft[k] = bool(np.mean(soft_at[m]) >= 0.6)
        moving[k] = bool(np.isfinite(g_) and g_ > 1.0)
    dt = 1.0 / SAMPLE_FPS
    dur = dec.get("decoded_duration_s") or float(t[-1] + dt)
    clip = lambda spans: [(a, min(b, dur)) for a, b in spans if a < dur]
    idle = clip(runs(t, static & ~in_flat, 5.0, dt))
    blur = clip(runs(t, soft & ~moving & ~in_flat, 3.0, dt))
    motion_blur = _span_len(runs(t, soft & moving & ~in_flat, 3.0, dt))
    return {"idle": idle, "blur": blur, "motion_blur_s": motion_blur, "blur_threshold": thr, "ratio_threshold": rthr}


def _merge(spans):
    return _union(spans)


def file_issues(rec: dict, spec: Spec, sharp_baseline: dict | None) -> tuple[list[Issue], dict]:
    """Findings for one file and its usable-time accounting."""
    issues: list[Issue] = []
    meta, ts, dec = rec.get("meta") or {}, rec.get("ts") or {}, rec.get("dec") or {}
    acct = {"header_s": meta.get("header_duration_s") or 0.0, "decoded_s": dec.get("decoded_duration_s") or 0.0}
    if rec.get("error"):
        issues.append(Issue("video.unreadable", CRITICAL, f"could not be analysed ({rec['error']})"))
        acct.update(usable_s=0.0, unusable={})
        return issues, acct
    if dec.get("open_error") or not dec.get("frames"):
        why = dec.get("open_error") or "no frames could be decoded"
        issues.append(Issue("video.unreadable", CRITICAL, f"cannot be opened or decoded: {why}"))
        acct.update(usable_s=0.0, unusable={"unreadable": acct["header_s"]})
        return issues, acct

    hdr, decd = acct["header_s"], acct["decoded_s"]
    if hdr and decd < 0.98 * hdr - 0.5:
        issues.append(Issue("video.truncated", CRITICAL,
                            f"decodes to {decd:.1f} s but the header says {hdr:.1f} s (cut off, e.g. battery or upload failure)",
                            seconds=hdr - decd))
    if dec.get("corrupt_frames") or dec.get("decode_errors"):
        n = dec.get("corrupt_frames", 0) + dec.get("decode_errors", 0)
        share = n / max(1, dec["frames"])
        issues.append(Issue("video.corrupt_frames", CRITICAL if share > 0.01 else WARNING,
                            f"{n} corrupt or undecodable frames ({share:.1%})"))

    # timing
    if ts.get("recording_gaps"):
        issues.append(Issue("video.recording_gap", WARNING,
                            f"{ts['recording_gaps']} gap(s) longer than 1 s in the timestamps "
                            f"({ts['recording_gap_s']:.1f} s missing; spliced or paused recording)",
                            seconds=ts["recording_gap_s"]))
    if ts.get("dropped_frames_est", 0) > max(2, 0.002 * dec["frames"]):
        issues.append(Issue("video.dropped_frames", WARNING,
                            f"about {ts['dropped_frames_est']} dropped frames in {ts['gaps']} gaps",
                            spans=ts.get("gap_spans", [])))
    if ts.get("vfr"):
        issues.append(Issue("video.variable_frame_rate", WARNING,
                            "the frame rate changes during the recording (common on phones in low light); "
                            "training loaders that assume a constant rate will reject or misalign frames"))
    if ts.get("duplicate_pts"):
        issues.append(Issue("video.bad_timestamps", WARNING,
                            f"{ts['duplicate_pts']} repeated timestamps (frames that players and loaders will drop)"))

    # content
    sp = content_spans(dec)
    win = windowed(dec, sp, sharp_baseline)
    dur = decd or 1e-9
    for key, check, sev, what in (
        ("black", "video.black", WARNING, "black (covered lens, lens cap or failed exposure)"),
        ("covered", "video.covered", WARNING, "featureless (lens covered by a hand, sleeve or surface)"),
        ("overexposed", "video.overexposed", WARNING, "washed out (over-exposed)"),
        ("frozen", "video.frozen", WARNING, "frozen (the same frame repeated: a stalled camera or inserted duplicates)"),
    ):
        s = sp.get(key) or []
        if s:
            issues.append(Issue(check, sev, f"{_span_len(s):.1f} s {what}", seconds=_span_len(s), spans=s))
    idle_s = _span_len(win["idle"])
    if idle_s:
        share = idle_s / dur
        issues.append(Issue("video.idle", WARNING if share > spec.max_idle_share else INFO,
                            f"{idle_s:.0f} s ({share:.0%}) with nothing moving at all (camera set down or wearer idle)",
                            seconds=idle_s, spans=win["idle"]))
    blur_s = _span_len(win["blur"])
    if blur_s:
        share = blur_s / dur
        issues.append(Issue("video.blur", WARNING if share > spec.max_blur_share else INFO,
                            f"{blur_s:.0f} s ({share:.0%}) persistently blurry without camera motion "
                            "(out of focus, dirty or fogged lens)", seconds=blur_s, spans=win["blur"]))
    if sp.get("low_light_share", 0) > 0.5:
        issues.append(Issue("video.low_light", INFO, f"{sp['low_light_share']:.0%} of frames are dark (low light)"))

    # metadata against the spec
    fps = meta.get("fps") or ts.get("nominal_fps")
    if spec.fps and fps and abs(fps - spec.fps) > spec.fps_tolerance:
        issues.append(Issue("video.spec_fps", WARNING, f"{fps:.2f} fps; the spec asks for {spec.fps:g}"))
    if spec.min_width and (meta.get("width") or 0) < spec.min_width or \
            spec.min_height and (meta.get("height") or 0) < spec.min_height:
        issues.append(Issue("video.spec_resolution", WARNING,
                            f"{meta.get('width')}x{meta.get('height')}; the spec asks for at least "
                            f"{spec.min_width or '?'}x{spec.min_height or '?'}"))
    if spec.codecs and meta.get("codec") and meta["codec"] not in spec.codecs:
        issues.append(Issue("video.spec_codec", WARNING, f"codec {meta['codec']}; the spec allows {', '.join(spec.codecs)}"))
    if spec.min_duration_s and decd < spec.min_duration_s:
        issues.append(Issue("video.spec_duration", WARNING, f"{decd:.0f} s, shorter than the {spec.min_duration_s:g} s minimum"))
    if spec.max_duration_s and decd > spec.max_duration_s:
        issues.append(Issue("video.spec_duration", WARNING, f"{decd:.0f} s, longer than the {spec.max_duration_s:g} s maximum"))
    if meta.get("field_order") not in (None, "progressive", "unknown"):
        issues.append(Issue("video.interlaced", INFO, f"interlaced ({meta['field_order']})"))
    if meta.get("rotation"):
        issues.append(Issue("video.rotation", INFO, f"rotation flag {meta['rotation']}° (players rotate it; most loaders don't)"))
    if meta.get("hdr"):
        issues.append(Issue("video.hdr", INFO, "HDR transfer (HLG/PQ); colours differ from SDR footage unless converted"))
    if (meta.get("video_streams") or 0) > 1:
        issues.append(Issue("video.multiple_streams", INFO, f"{meta['video_streams']} video streams; only the first is checked"))

    unusable = {"black": sp.get("black", []), "covered": sp.get("covered", []), "overexposed": sp.get("overexposed", []),
                "frozen": sp.get("frozen", []), "idle": win["idle"], "blur": win["blur"]}
    bad = _union(*unusable.values())
    trunc = max(0.0, hdr - decd) if hdr and decd < 0.98 * hdr - 0.5 else 0.0
    gap_s = float(ts.get("recording_gap_s") or 0.0)   # time on the clock with no frames: not footage
    acct["footage_s"] = max(0.0, decd - gap_s)
    acct["usable_s"] = max(0.0, decd - gap_s - _span_len(bad))
    acct["unusable"] = {k: _span_len(v) for k, v in unusable.items() if v}
    acct["bad_spans"] = bad
    acct["idle_spans"] = win["idle"]
    if trunc:
        acct["unusable"]["truncated"] = trunc
    if any(i.severity == CRITICAL and i.check in ("video.corrupt_frames",) for i in issues):
        acct["usable_s"] = 0.0
        acct["unusable"]["corrupt"] = decd
    acct["motion_blur_s"] = win.get("motion_blur_s", 0.0)
    return issues, acct


# --------------------------------------------------------------------------------------------- #
# Duplicates


_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def _ham(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.bitwise_xor(a[:, None], b[None, :]).astype(">u8").view(np.uint8).reshape(len(a), len(b), 8)
    return _POP[x].sum(-1)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.ravel() - x.mean()
    y = y.ravel() - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-6 else 0.0


def _motion_agreement(ta: np.ndarray, tb: np.ndarray, bga: np.ndarray, bgb: np.ndarray) -> tuple[float, float, float]:
    """(correlation of frame-to-frame changes, correlation of foreground = thumbnail minus the file's median
    background, motion energy in grey levels) for two aligned thumbnail runs."""
    if len(ta) < 3 or len(tb) < 3:
        return 0.0, 0.0, 0.0
    za = (ta - ta.mean(axis=(1, 2), keepdims=True))
    zb = (tb - tb.mean(axis=(1, 2), keepdims=True))
    da, db = np.diff(za, axis=0), np.diff(zb, axis=0)
    motion = float(min(np.abs(da).mean(), np.abs(db).mean()))
    fa = za - (bga - bga.mean())
    fb = zb - (bgb - bgb.mean())
    return round(_corr(da, db), 3), round(_corr(fa, fb), 3), round(motion, 2)


def align(sa: dict, sb: dict, *, same_file: bool = False) -> dict | None:
    """Longest run of matching frames at one time offset between two hash sequences (2 fps grid).
    A single similar frame is not evidence (frames of one scene seconds apart look alike); only a long
    run at a consistent offset is. Mirrored copies are matched through the flipped-frame hashes."""
    best = None
    for hb_key in ("h", "hf"):
        ha, hb = sa["h"], sb[hb_key]
        if not len(ha) or not len(hb):
            continue
        M = (_ham(ha, hb) <= HASH_THR) & (sa["q"][:, None] >= HASH_MIN_Q) & (sb["q"][None, :] >= HASH_MIN_Q)
        if same_file:
            k = int(LOOP_MIN_OFFSET_S * SAMPLE_FPS)
            ii, jj = np.indices(M.shape)
            M &= (jj - ii) >= k
        ii, jj = np.nonzero(M)
        if not len(ii):
            continue
        vals, cnt = np.unique(jj - ii, return_counts=True)
        for o in vals[np.argsort(-cnt)[:6]]:
            idx = np.arange(max(0, -o), min(len(ha), len(hb) - o))
            ok = M[idx, idx + o]
            run = gap = 0
            start = None
            top = (0, None, 0)
            matched = 0
            for k, m in zip(idx, ok):
                if m:
                    if start is None:
                        start, run, matched = k, 0, 0
                    run += 1 + gap
                    matched += 1
                    gap = 0
                    if run > top[0]:
                        top = (run, start, matched)
                elif start is not None:
                    gap += 1
                    if gap > 2:
                        start, run, gap = None, 0, 0
            length = top[0] / SAMPLE_FPS
            if length < MIN_DUP_RUN_S or top[1] is None:
                continue
            a0 = int(top[1])
            n = top[0]
            # verify on thumbnails: copies keep hands/arms/objects in the same place, other takes don't
            ta = sa["thumb"][a0:a0 + n].astype(np.float32)
            tbm = sb["thumb"][a0 + o:a0 + o + n].astype(np.float32)
            if hb_key == "hf":
                tbm = tbm[:, :, ::-1]
            m_n = min(len(ta), len(tbm))
            diff = float(np.median(np.abs(ta[:m_n] - tbm[:m_n]).mean(axis=(1, 2)))) if m_n else 99.0
            # brightness/contrast changes on a re-encode shift the thumbnails; compare after normalising
            if m_n:
                za = (ta[:m_n] - ta[:m_n].mean(axis=(1, 2), keepdims=True)) / (ta[:m_n].std(axis=(1, 2), keepdims=True) + 1)
                zb = (tbm[:m_n] - tbm[:m_n].mean(axis=(1, 2), keepdims=True)) / (tbm[:m_n].std(axis=(1, 2), keepdims=True) + 1)
                zdiff = float(np.median(np.abs(za - zb).mean(axis=(1, 2))))
            else:
                zdiff = 9.0
            density = top[2] / max(1, n)
            # The decisive test. Takes of the same task from a fixed camera share the background, so frames
            # match; what differs is where and when things move. A copy repeats the same changes at the
            # same moments: correlate the frame-to-frame changes, and the departures from each file's
            # own static background, over the aligned run.
            bb = sb["thumb"].astype(np.float32)
            bga = np.median(sa["thumb"].astype(np.float32), axis=0)
            bgb = np.median(bb[:, :, ::-1] if hb_key == "hf" else bb, axis=0)
            dcorr, fgcorr, motion = _motion_agreement(ta[:m_n], tbm[:m_n], bga, bgb)
            cand = dict(offset_s=o / SAMPLE_FPS, start_a=a0 / SAMPLE_FPS, start_b=(a0 + o) / SAMPLE_FPS,
                        overlap_s=length, density=round(density, 2), thumb_diff=round(diff, 1),
                        norm_diff=round(zdiff, 3), mirrored=hb_key == "hf", motion_corr=dcorr,
                        foreground_corr=fgcorr, motion=motion)
            if (best is None or length > best["overlap_s"]):
                best = cand
    return best


def _band_candidates(store: dict[str, dict], max_bucket: int = 500, min_votes: int = 8) -> list[tuple[str, str]]:
    idx: dict = defaultdict(list)
    for f, s in store.items():
        h, q = s["h"], s["q"]
        hf = s["hf"]
        for k in range(len(h)):
            if q[k] < HASH_MIN_Q:
                continue
            for b in range(4):
                idx[(b, (int(h[k]) >> (16 * b)) & 0xFFFF)].append((f, k, 0))
                idx[(b, (int(hf[k]) >> (16 * b)) & 0xFFFF)].append((f, k, 1))
    votes: Counter = Counter()
    for lst in idx.values():
        if len(lst) > max_bucket:
            continue
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (fa, ka, xa), (fb, kb, xb) = lst[i], lst[j]
                if fa == fb or (xa and xb):
                    continue
                if fa > fb:
                    fa, ka, fb, kb = fb, kb, fa, ka
                votes[(fa, fb, kb - ka)] += 1
    best: dict = defaultdict(int)
    for (fa, fb, _), v in votes.items():
        best[(fa, fb)] = max(best[(fa, fb)], v)
    return [p for p, v in best.items() if v >= min_votes]


DUP_NORM_DIFF = 0.45     # normalised thumbnail difference below which aligned frames are the same footage
DUP_MOTION_CORR = 0.75   # correlation of frame-to-frame changes above which aligned runs are the same recording
                         # (real robot footage: copies 0.90-0.94; other takes of the same task, same camera: at most 0.55)
DUP_LOW_MOTION = 0.4     # grey levels; below this there is too little movement to compare changes
DUP_STILL_FG_CORR = 0.85


FINE_FPS = 6.0           # grid for the frame-accurate re-check of candidate pairs
FINE_MAX_S = 30.0        # longest stretch re-decoded per candidate pair


def _fine_thumbs(path: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Every frame's 32x18 thumbnail between t0 and t1 (seconds from the first frame)."""
    import av
    import cv2

    ts, th = [], []
    try:
        c = av.open(path)
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        tb = float(s.time_base)
        first = None
        for fr in c.decode(s):
            if fr.pts is None:
                continue
            t = fr.pts * tb
            first = t if first is None else first
            t -= first
            if t < t0:
                continue
            if t > t1:
                break
            w, h = fr.width, fr.height
            aw = max(16, min(ANALYSIS_WIDTH, w - w % 2))
            g = fr.reformat(width=aw, height=max(2, int(round(h * aw / w / 2) * 2)), format="gray").to_ndarray()
            ts.append(t)
            th.append(cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA))
        c.close()
    except Exception:  # noqa: BLE001
        pass
    return np.array(ts), (np.stack(th).astype(np.float32) if th else np.zeros((0, 18, 32), np.float32))


def refine(path_a: str, path_b: str, res: dict) -> dict:
    """Frame-accurate re-check of a candidate pair. The 2 fps grids of two files fall at unrelated
    moments (a copy cut at 3.3 s is sampled 0.2 s off), and over 0.2 s fast motion changes a thumbnail a
    lot. Re-decode the matched stretch of both files, find the sub-second shift that lines them up best,
    and recompute the motion agreement on a 6 fps grid."""
    span = min(res["overlap_s"], FINE_MAX_S)
    a0 = res["start_a"] + max(0.0, (res["overlap_s"] - span) / 2)
    ta, tha = _fine_thumbs(path_a, a0, a0 + span)
    off = res["start_b"] - res["start_a"]
    tb, thb = _fine_thumbs(path_b, max(0.0, a0 + off - 0.75), a0 + off + span + 0.75)
    if len(ta) < 6 or len(tb) < 6:
        return res
    if res.get("mirrored"):
        thb = thb[:, :, ::-1]
    grid = np.arange(a0 + 0.2, a0 + span - 0.2, 1.0 / FINE_FPS)
    ia = np.clip(np.searchsorted(ta, grid), 0, len(ta) - 1)
    A = tha[ia]
    best = None
    for d in np.arange(-0.6, 0.61, 0.05):
        tq = grid + off + d
        ib = np.clip(np.searchsorted(tb, tq), 0, len(tb) - 1)
        ib = np.where((ib > 0) & (np.abs(tb[np.maximum(ib - 1, 0)] - tq) < np.abs(tb[ib] - tq)), ib - 1, ib)
        ok = np.abs(tb[ib] - tq) <= 0.6 / FINE_FPS
        if ok.sum() < 6:
            continue
        mc, fg, mo = _motion_agreement(A[ok], thb[ib[ok]], np.median(tha, 0), np.median(thb, 0))
        if best is None or mc > best[0]:
            best = (mc, fg, mo, d)
    if best is None:
        return res
    return dict(res, motion_corr=best[0], foreground_corr=best[1], motion=best[2], fine_shift_s=round(float(best[3]), 2),
                coarse_motion_corr=res.get("motion_corr"), refined=True)


def _verified(res: dict | None) -> bool:
    if not res or res["norm_diff"] > DUP_NORM_DIFF or res["density"] < 0.6:
        return False
    if res["motion"] >= DUP_LOW_MOTION:
        return res["motion_corr"] >= DUP_MOTION_CORR
    return res["foreground_corr"] >= DUP_STILL_FG_CORR


def find_duplicates(records: list[dict], history: dict | None = None, debug: list | None = None) -> list[dict]:
    """Exact copies (same bytes, or same coded stream in a different container), near-duplicates and
    partial overlaps between files, against an optional store of earlier batches, plus loops within a file."""
    out: list[dict] = []
    by_sha: dict = defaultdict(list)
    by_stream: dict = defaultdict(list)
    for r in records:
        if r.get("sha256"):
            by_sha[r["sha256"]].append(r["path"])
        sh = (r.get("ts") or {}).get("stream_hash")
        if sh and (r.get("ts") or {}).get("packets", 0) > 1:
            by_stream[sh].append(r["path"])
    seen_pairs = set()
    for paths in by_sha.values():
        for b in paths[1:]:
            out.append(dict(kind="identical_file", a=paths[0], b=b, overlap_s=None))
            seen_pairs.add(tuple(sorted((paths[0], b))))
    for paths in by_stream.values():
        for b in paths[1:]:
            pair = tuple(sorted((paths[0], b)))
            if pair not in seen_pairs:
                out.append(dict(kind="same_stream_remuxed", a=paths[0], b=b, overlap_s=None))
                seen_pairs.add(pair)
    store = {r["path"]: r["dec"]["samples"] for r in records
             if (r.get("dec") or {}).get("samples") and len(r["dec"]["samples"].get("h", []))}
    hist = {}
    for key, s in (history or {}).items():
        if key not in store:
            hist["history:" + key] = s
    allstore = {**store, **hist}
    durations = {r["path"]: (r.get("dec") or {}).get("decoded_duration_s") or 0 for r in records}
    for key, s in hist.items():
        durations[key] = len(s["h"]) / SAMPLE_FPS
    for a, b in _band_candidates(allstore):
        if a.startswith("history:") and b.startswith("history:"):
            continue
        pair = tuple(sorted((a, b)))
        if pair in seen_pairs:
            continue
        res = align(allstore[a], allstore[b])
        if res and res["norm_diff"] <= DUP_NORM_DIFF and res["density"] >= 0.6 and \
                not a.startswith("history:") and not b.startswith("history:"):
            res = refine(a, b, res)
        if debug is not None and res:
            debug.append(dict(a=a, b=b, verified=_verified(res), **res))
        if not _verified(res):
            continue
        da, db = durations.get(a, 0), durations.get(b, 0)
        shorter = max(1e-9, min(da, db))
        longer = max(1e-9, max(da, db))
        frac = res["overlap_s"] / shorter
        if res["overlap_s"] >= 0.8 * longer:
            kind = "near_duplicate"          # the same footage, re-encoded
        elif frac >= 0.8:
            kind = "trimmed_copy"            # the shorter file is a cut of the longer one
        elif res["overlap_s"] >= 10 or frac >= 0.3:
            kind = "partial_overlap"
        else:
            continue
        seen_pairs.add(pair)
        if kind == "trimmed_copy" and da < db:   # orient so that b is the shorter (the cut)
            a, b = b, a
            res = dict(res, start_a=res["start_b"], start_b=res["start_a"], offset_s=-res["offset_s"])
        out.append(dict(kind=kind, a=a, b=b, overlap_s=round(res["overlap_s"], 1), share_of_shorter=round(frac, 2),
                        offset_s=res["offset_s"], start_a=res["start_a"], start_b=res["start_b"],
                        mirrored=res["mirrored"], evidence=dict(density=res["density"], norm_diff=res["norm_diff"])))
    for path, s in store.items():
        res = align(s, s, same_file=True)
        if debug is not None and res:
            debug.append(dict(a=path, b=path, verified=_verified(res), loop=True, **res))
        if _verified(res) and res["overlap_s"] >= 10:
            out.append(dict(kind="loop_within_file", a=path, b=path, overlap_s=round(res["overlap_s"], 1),
                            start_a=res["start_a"], start_b=res["start_b"], offset_s=res["offset_s"],
                            mirrored=res["mirrored"], evidence=dict(density=res["density"], norm_diff=res["norm_diff"])))
    return out


STRONG_DUP = ("identical_file", "same_stream_remuxed", "near_duplicate")


def consolidate_duplicates(raw: list[dict], records: list[dict]) -> list[dict]:
    """Group copies into clusters and report each copy once, against one original per cluster: a file
    from an earlier batch if there is one, else the longest file (ties: first by path). Partial overlaps
    are kept once per pair of clusters. Pairs come out as a = original/earlier, b = copy, with start_a and
    start_b on each file's own timeline."""
    dur = {r["path"]: (r.get("dec") or {}).get("decoded_duration_s") or 0.0 for r in records}
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d in raw:
        if d["kind"] in STRONG_DUP:
            parent[find(d["a"])] = find(d["b"])
    members: dict = defaultdict(list)
    for x in list(parent):
        members[find(x)].append(x)

    def canon(group):
        hist = sorted(g for g in group if g.startswith("history:"))
        if hist:
            return hist[0]
        return sorted(group, key=lambda g: (-round(dur.get(g, 0.0), 1), g))[0]

    def oriented(d, orig, copy):
        """d re-expressed with a = orig side, b = copy side."""
        e = dict(d)
        if d["a"] == orig and d["b"] == copy:
            return e
        e["a"], e["b"] = d["b"], d["a"]
        e["start_a"], e["start_b"] = d.get("start_b"), d.get("start_a")
        if e.get("offset_s") is not None:
            e["offset_s"] = -e["offset_s"]
        return e

    rank = {"identical_file": 0, "same_stream_remuxed": 1, "near_duplicate": 2}
    out: list[dict] = []
    for group in members.values():
        if len(group) < 2:
            continue
        orig = canon(group)
        for m in sorted(group):
            if m == orig or m.startswith("history:"):
                continue
            edges = [d for d in raw if d["kind"] in STRONG_DUP and {d["a"], d["b"]} == {orig, m}]
            via = None
            if not edges:
                edges = [d for d in raw if d["kind"] in STRONG_DUP and m in (d["a"], d["b"])]
                via = None
            edge = sorted(edges, key=lambda d: (rank[d["kind"]], -(d.get("overlap_s") or 0)))[0]
            other = edge["a"] if edge["b"] == m else edge["b"]
            e = oriented(edge, other, m)
            if other != orig:
                via = other
            e["a"] = orig
            e["via"] = via
            if e["kind"] == "near_duplicate" and e.get("start_b") is None:
                e["start_b"] = 0.0
            out.append(e)
    reported = {e["b"] for e in out}
    canonical = {canon(g) for g in members.values() if len(g) > 1}
    best_trim: dict = {}
    for d in raw:
        if d["kind"] != "trimmed_copy" or d["b"].startswith("history:") or find(d["a"]) == find(d["b"]):
            continue
        if d["b"] in reported:
            continue
        cur = best_trim.get(d["b"])
        # prefer an original from an earlier batch, then the longest original
        key = (d["a"].startswith("history:"), d["a"] in canonical or find(d["a"]) == d["a"],
               dur.get(d["a"], 0.0), d["overlap_s"], d["a"] < d["b"])
        if cur is None or key > cur[0]:
            best_trim[d["b"]] = (key, d)
    out += [d for _, d in best_trim.values()]
    trimmed = set(best_trim)
    root = {x: find(x) for x in parent}
    for b, (_, d) in best_trim.items():          # a cut belongs to its original's family
        root[b] = root.get(d["a"], find(d["a"]))

    def family(x):
        return root.get(x, x)

    best_partial: dict = {}
    for d in raw:
        if d["kind"] == "trimmed_copy" and d["b"] in trimmed and best_trim[d["b"]][1] is not d:
            continue
        if d["kind"] != "partial_overlap":
            continue
        ra, rb = family(d["a"]), family(d["b"])
        if ra == rb:
            continue
        key = tuple(sorted((ra, rb)))
        if key not in best_partial or d["overlap_s"] > best_partial[key]["overlap_s"]:
            best_partial[key] = d
    for d in best_partial.values():
        # the copy is the side that isn't from an earlier batch; otherwise the later path
        if d["b"].startswith("history:") or (not d["a"].startswith("history:") and d["a"] > d["b"]):
            d = oriented(d, d["b"], d["a"])
        out.append(d)
    out += [d for d in raw if d["kind"] == "loop_within_file"]
    return out


def load_store(path: str | Path | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    out = {}
    for key, s in data.get("files", {}).items():
        out[key] = {"t": np.array(s["t"]), "h": np.array([int(x, 16) for x in s["h"]], dtype=np.uint64),
                    "hf": np.array([int(x, 16) for x in s["hf"]], dtype=np.uint64), "q": np.array(s["q"]),
                    "thumb": np.array(s["thumb"], dtype=np.uint8).reshape(-1, 18, 32)}
    return out


def save_store(path: str | Path, records: list[dict], previous: dict) -> None:
    files = {}
    for key, s in previous.items():
        files[key] = {"t": s["t"].round(2).tolist(), "h": [f"{int(x):016x}" for x in s["h"]],
                      "hf": [f"{int(x):016x}" for x in s["hf"]], "q": s["q"].round(1).tolist(),
                      "thumb": s["thumb"].astype(int).ravel().tolist()}
    for r in records:
        s = (r.get("dec") or {}).get("samples")
        if not s or not len(s.get("h", [])):
            continue
        key = f"{r.get('sha256', '')[:16]}:{r['name']}"
        files[key] = {"t": s["t"].round(2).tolist(), "h": [f"{int(x):016x}" for x in s["h"]],
                      "hf": [f"{int(x):016x}" for x in s["hf"]], "q": s["q"].round(1).tolist(),
                      "thumb": s["thumb"].astype(int).ravel().tolist()}
    Path(path).write_text(json.dumps({"version": 1, "sample_fps": SAMPLE_FPS, "files": files}))


# --------------------------------------------------------------------------------------------- #
# Batch report


def _robust_outliers(values: dict[str, float], z: float = 3.5) -> set[str]:
    keys = [k for k, v in values.items() if v is not None and np.isfinite(v)]
    if len(keys) < 8:
        return set()
    x = np.array([values[k] for k in keys], dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad == 0:
        return set()
    return {k for k, v in zip(keys, x) if abs(v - med) / mad > z}


def check_video_batch(paths: list[str], *, contributors_csv: str | None = None, by_folder: bool = False,
                      spec: Spec | None = None, store: str | None = None, workers: int | None = None,
                      progress=None, debug: bool = False) -> dict:
    spec = spec or Spec()
    files, root = find_videos(paths)
    mapping = load_contributors(contributors_csv) if contributors_csv else None
    if not files:
        raise ValueError("no video files found (looked for " + ", ".join(sorted(VIDEO_EXTENSIONS)) + ")")
    workers = workers or min(4, os.cpu_count() or 1)
    records: list[dict] = []
    if workers > 1 and len(files) > 1:
        import multiprocessing as mp

        # spawn, not fork: the decoder threads of the parent must not be copied into the workers
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for k, rec in enumerate(ex.map(measure_file, [str(f) for f in files])):
                records.append(rec)
                if progress:
                    progress(k + 1, len(files), rec["name"])
    else:
        for k, f in enumerate(files):
            records.append(measure_file(str(f)))
            if progress:
                progress(k + 1, len(files), f.name)
    for r in records:
        r["contributor"] = contributor_of(Path(r["path"]), root, mapping, by_folder)

    # typical sharpness per camera (contributor and resolution): the median over its files of each file's
    # median, so a few blurry files or blurry stretches don't set the bar
    groups: dict = defaultdict(lambda: {"lap": [], "ratio": []})
    gkey = {}
    for r in records:
        dec, m = r.get("dec") or {}, r.get("meta") or {}
        gkey[r["path"]] = (r["contributor"], m.get("width"), m.get("height"))
        lap, lapr = dec.get("lap"), dec.get("lapr")
        if lap is not None and len(lap) >= 5:
            groups[gkey[r["path"]]]["lap"].append(float(np.median(lap)))
            if lapr is not None and np.isfinite(lapr).any():
                groups[gkey[r["path"]]]["ratio"].append(float(np.nanmedian(lapr)))
    bases = {k: {"lap": float(np.median(v["lap"])) if v["lap"] else 0.0,
                 "ratio": float(np.median(v["ratio"])) if v["ratio"] else 0.0} for k, v in groups.items()}
    sharp_baseline = float(np.median([b["lap"] for b in bases.values()])) if bases else 0.0

    files_out = []
    for r in records:
        issues, acct = file_issues(r, spec, bases.get(gkey[r["path"]]))
        files_out.append({"record": r, "issues": issues, "acct": acct})

    # batch outliers (bitrate per pixel, duration) and re-encode hints
    bpp = {}
    for fo in files_out:
        m = fo["record"].get("meta") or {}
        if m.get("bit_rate") and m.get("width") and m.get("height") and (m.get("fps") or 0) > 0:
            bpp[fo["record"]["path"]] = m["bit_rate"] / (m["width"] * m["height"] * m["fps"])
    low_bpp = _robust_outliers(bpp)
    med_bpp = float(np.median(list(bpp.values()))) if bpp else None
    fps_counts = Counter(round((fo["record"].get("meta") or {}).get("fps") or 0, 2) for fo in files_out)
    res_counts = Counter(((fo["record"].get("meta") or {}).get("width"), (fo["record"].get("meta") or {}).get("height"))
                         for fo in files_out)
    common_fps = fps_counts.most_common(1)[0][0] if fps_counts else None
    common_res = res_counts.most_common(1)[0][0] if res_counts else None
    for fo in files_out:
        p = fo["record"]["path"]
        m = fo["record"].get("meta") or {}
        if p in low_bpp and med_bpp and bpp[p] < med_bpp:
            fo["issues"].append(Issue("video.low_bitrate", INFO,
                                      f"bitrate per pixel {bpp[p]:.3f} vs batch median {med_bpp:.3f} "
                                      "(heavily re-compressed: possibly downloaded or re-encoded, not camera-original)"))
        if not spec.fps and len(files_out) >= 5 and common_fps and m.get("fps") and \
                abs(round(m["fps"], 2) - common_fps) > 0.5 and fps_counts[common_fps] >= 0.6 * len(files_out):
            fo["issues"].append(Issue("video.unusual_fps", INFO, f"{m['fps']:.2f} fps; most files are {common_fps:g} fps"))
        if not (spec.min_width or spec.min_height) and len(files_out) >= 5 and common_res and \
                (m.get("width"), m.get("height")) != common_res and res_counts[common_res] >= 0.6 * len(files_out):
            fo["issues"].append(Issue("video.unusual_resolution", INFO,
                                      f"{m.get('width')}x{m.get('height')}; most files are {common_res[0]}x{common_res[1]}"))

    # Frames with nothing moving look alike across unrelated files (and within one), so they don't count
    # as duplicate evidence: give them zero detail before matching. (Flat frames are already skipped.)
    for fo in files_out:
        sm = (fo["record"].get("dec") or {}).get("samples")
        if not sm or not len(sm.get("t", [])):
            continue
        quiet = fo["acct"].get("idle_spans", [])
        if quiet:
            mask = np.array([any(a <= x < b for a, b in quiet) for x in sm["t"]], dtype=bool)
            sm["q_all"] = sm["q"].copy()
            sm["q"] = np.where(mask, 0.0, sm["q"])
    history = load_store(store)
    dbg_pairs: list | None = [] if debug else None
    dups = consolidate_duplicates(find_duplicates(records, history, dbg_pairs), records)
    contrib = {r["path"]: r["contributor"] for r in records}
    by_path = {fo["record"]["path"]: fo for fo in files_out}
    for d in dups:
        ca, cb = contrib.get(d["a"], "(earlier batch)"), contrib.get(d["b"], "(earlier batch)")
        d["contributor_a"], d["contributor_b"] = ca, cb
        d["cross_contributor"] = ca != cb and d["kind"] != "loop_within_file"
        target = d["b"]             # a = original (or the earlier file), b = the copy
        other_name = d["a"].replace("history:", "earlier batch: ")
        who = f" from another contributor ({ca})" if d["cross_contributor"] else ""
        mirrored = " (mirrored)" if d.get("mirrored") else ""
        if d["kind"] == "loop_within_file":
            sev, sec = WARNING, d["overlap_s"]
            msg = (f"footage repeats within the file: {d['overlap_s']:.0f} s starting at {d['start_b']:.0f} s "
                   f"copies {d['start_a']:.0f} s{mirrored}")
            span = [(d["start_b"], d["start_b"] + d["overlap_s"])]
        elif d["kind"] in ("identical_file", "same_stream_remuxed"):
            sec = by_path[target]["acct"]["decoded_s"] if target in by_path else 0.0
            sev = CRITICAL
            what = "identical file" if d["kind"] == "identical_file" else "the same recording in a different container"
            msg = f"{what} as {Path(other_name).name}{who}"
            span = [(0.0, sec)]
        elif d["kind"] == "near_duplicate":
            sev, sec = CRITICAL, d["overlap_s"]
            msg = f"re-encoded copy of {Path(other_name).name}{who} ({d['overlap_s']:.0f} s){mirrored}"
            span = [(d["start_b"], d["start_b"] + d["overlap_s"])]
        elif d["kind"] == "trimmed_copy":
            sev, sec = CRITICAL, d["overlap_s"]
            msg = (f"cut from {Path(other_name).name}{who}: {d['overlap_s']:.0f} s starting at "
                   f"{d['start_a']:.0f} s there{mirrored}")
            span = [(d["start_b"], d["start_b"] + d["overlap_s"])]
        else:
            sev, sec = WARNING, d["overlap_s"]
            msg = (f"{d['overlap_s']:.0f} s overlap with {Path(other_name).name}{who} "
                   f"(from {d['start_b']:.0f} s here, {d['start_a']:.0f} s there){mirrored}")
            span = [(d["start_b"], d["start_b"] + d["overlap_s"])]
        if d.get("via"):
            msg += f" (matched through {Path(d['via']).name})"
        if target in by_path:
            fo = by_path[target]
            fo["issues"].append(Issue(f"video.{d['kind']}", sev, msg, seconds=sec, spans=span))
            fo["acct"].setdefault("dup_spans", []).extend(span)

    # final usable accounting with duplicates
    for fo in files_out:
        a = fo["acct"]
        if a.get("dup_spans"):
            bad = _union(a.get("bad_spans", []), a["dup_spans"])
            dup_only = _span_len(_union(a["dup_spans"]))
            a["usable_s"] = 0.0 if any(i.check in ("video.identical_file", "video.same_stream_remuxed") for i in fo["issues"]) \
                else max(0.0, a.get("footage_s", a["decoded_s"]) - _span_len(bad))
            a["unusable"]["duplicate"] = dup_only
        if any(i.severity == CRITICAL and i.check in ("video.unreadable", "video.corrupt_frames") for i in fo["issues"]):
            a["usable_s"] = 0.0

    # per-contributor scorecard
    cards: dict = defaultdict(lambda: {"files": 0, "hours": 0.0, "usable_hours": 0.0, "critical_files": 0,
                                       "warning_files": 0, "unusable_s": Counter(), "cross_contributor_duplicates": 0})
    for fo in files_out:
        c = cards[fo["record"]["contributor"]]
        c["files"] += 1
        c["hours"] += (fo["acct"].get("footage_s", fo["acct"].get("decoded_s")) or fo["acct"].get("header_s") or 0) / 3600
        c["usable_hours"] += (fo["acct"].get("usable_s") or 0) / 3600
        sevs = {i.severity for i in fo["issues"]}
        c["critical_files"] += CRITICAL in sevs
        c["warning_files"] += CRITICAL not in sevs and WARNING in sevs
        for k, v in fo["acct"].get("unusable", {}).items():
            c["unusable_s"][k] += v
    for d in dups:
        if d.get("cross_contributor"):
            for side in ("contributor_a", "contributor_b"):
                if d[side] in cards:
                    cards[d[side]]["cross_contributor_duplicates"] += 1
    batch_h = sum(c["hours"] for c in cards.values())
    batch_u = sum(c["usable_hours"] for c in cards.values())
    prior = batch_u / batch_h if batch_h else 0.0
    scorecard = []
    for name, c in cards.items():
        share = c["usable_hours"] / c["hours"] if c["hours"] else 0.0
        k = 0.25  # hours of pseudo-data: small contributors are pulled toward the batch rate when ranking
        shrunk = (c["usable_hours"] + k * prior) / (c["hours"] + k) if c["hours"] + k else 0.0
        scorecard.append({"contributor": name, "files": c["files"], "hours": round(c["hours"], 3),
                          "usable_hours": round(c["usable_hours"], 3), "usable_share": round(share, 3),
                          "ranking_score": round(shrunk, 3), "critical_files": c["critical_files"],
                          "warning_files": c["warning_files"],
                          "cross_contributor_duplicates": c["cross_contributor_duplicates"],
                          "unusable_minutes": {k2: round(v / 60, 1) for k2, v in c["unusable_s"].most_common()}})
    scorecard.sort(key=lambda r: r["ranking_score"])

    totals = Counter()
    for fo in files_out:
        for k, v in fo["acct"].get("unusable", {}).items():
            totals[k] += v
    sev_count = Counter()
    for fo in files_out:
        sevs = {i.severity for i in fo["issues"]}
        sev_count["critical" if CRITICAL in sevs else "warning" if WARNING in sevs else "ok"] += 1
    return {
        "tool": "rdq check-video",
        "summary": {"files": len(files_out), "hours": round(batch_h, 3), "usable_hours": round(batch_u, 3),
                    "usable_share": round(prior, 3), "files_critical": sev_count["critical"],
                    "files_warning": sev_count["warning"], "files_ok": sev_count["ok"],
                    "unusable_minutes": {k: round(v / 60, 1) for k, v in totals.most_common()},
                    "duplicates": Counter(d["kind"] for d in dups),
                    "cross_contributor_duplicates": sum(1 for d in dups if d.get("cross_contributor")),
                    "contributors": len(cards), "sharpness_baseline": round(sharp_baseline, 1)},
        "spec": {k: v for k, v in spec.__dict__.items() if v is not None},
        "scorecard": scorecard,
        "duplicates": dups,
        "files": [{
            "path": fo["record"]["path"], "contributor": fo["record"]["contributor"],
            "duration_s": round(fo["acct"].get("footage_s", fo["acct"].get("decoded_s")) or 0, 2),
            "usable_s": round(fo["acct"].get("usable_s") or 0, 2),
            "status": ("critical" if any(i.severity == CRITICAL for i in fo["issues"]) else
                       "warning" if any(i.severity == WARNING for i in fo["issues"]) else "ok"),
            "meta": {k: v for k, v in (fo["record"].get("meta") or {}).items()},
            "timing": {k: v for k, v in (fo["record"].get("ts") or {}).items() if k not in ("gap_spans", "stream_hash")},
            "issues": [i.to_dict() for i in sorted(fo["issues"], key=lambda i: (["critical", "warning", "info"].index(i.severity), -i.seconds))],
        } for fo in files_out],
        **({"debug": _debug_info(files_out, dbg_pairs, sharp_baseline, bases, gkey)} if debug else {}),
        "_records": records,
        "_store": store,
        "_history": history,
    }


def _debug_info(files_out: list, pairs: list, sharp_baseline: float, bases: dict, gkey: dict) -> dict:
    """Raw measurements behind the findings, for tuning thresholds on new footage."""
    def num(x):
        return round(float(x), 3) if isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) else x

    files = {}
    for fo in files_out:
        dec = fo["record"].get("dec") or {}
        lap = dec.get("lap")
        sm = dec.get("samples") or {}
        chg = change_1s(sm) if sm.get("t") is not None and len(sm.get("t", [])) else np.array([])
        files[fo["record"]["path"]] = {
            "lap_p10": num(np.percentile(lap, 10)) if lap is not None and len(lap) else None,
            "lap_median": num(np.median(lap)) if lap is not None and len(lap) else None,
            "lap_p90": num(np.percentile(lap, 90)) if lap is not None and len(lap) else None,
            "change_1s_median": num(np.nanmedian(chg)) if np.isfinite(chg).any() else None,
            "change_1s_p10": num(np.nanpercentile(chg, 10)) if np.isfinite(chg).any() else None,
            "ratio_p10": num(np.nanpercentile(dec["lapr"], 10)) if dec.get("lapr") is not None and np.isfinite(dec["lapr"]).any() else None,
            "ratio_median": num(np.nanmedian(dec["lapr"])) if dec.get("lapr") is not None and np.isfinite(dec["lapr"]).any() else None,
            "camera_lap": num((bases.get(gkey.get(fo["record"]["path"])) or {}).get("lap", 0)),
            "camera_ratio": num((bases.get(gkey.get(fo["record"]["path"])) or {}).get("ratio", 0)),
        }
    return {"sharpness_baseline": num(sharp_baseline), "blur_threshold": num(BLUR_REL * sharp_baseline),
            "files": files, "pairs": [{k: num(v) for k, v in d.items()} for d in (pairs or [])]}


def write_outputs(report: dict, json_path: str | None = None, html_path: str | None = None,
                  csv_path: str | None = None) -> None:
    from .cli import _json_safe

    if report.get("_store"):
        save_store(report["_store"], report["_records"], report.get("_history") or {})
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    if json_path:
        Path(json_path).write_text(json.dumps(_json_safe(clean), indent=2, default=str))
    if html_path:
        Path(html_path).write_text(render_html(clean))
    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "contributor", "status", "duration_s", "usable_s", "issues"])
            for f in clean["files"]:
                w.writerow([f["path"], f["contributor"], f["status"], f["duration_s"], f["usable_s"],
                            "; ".join(f"{i['check']}: {i['message']}" for i in f["issues"])])


def format_terminal(report: dict, color: bool = False) -> str:
    def c(s, code):
        return f"\033[{code}m{s}\033[0m" if color else s

    s = report["summary"]
    lines = [c("rdq check-video", "1"),
             f"  {s['files']} files, {s['hours']:.2f} h of footage from {s['contributors']} contributor(s)",
             f"  usable: {s['usable_hours']:.2f} h ({s['usable_share']:.0%})   "
             f"files: {c(str(s['files_critical']) + ' critical', '31')}, {c(str(s['files_warning']) + ' warning', '33')}, "
             f"{s['files_ok']} ok"]
    if s["unusable_minutes"]:
        lines.append("  unusable minutes by reason: " + ", ".join(f"{k} {v:g}" for k, v in s["unusable_minutes"].items()))
    if report["duplicates"]:
        kinds = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in s["duplicates"].items())
        lines.append(f"  duplicates: {kinds}" + (f" ({s['cross_contributor_duplicates']} across contributors)"
                                                 if s["cross_contributor_duplicates"] else ""))
    if len(report["scorecard"]) > 1 or report["scorecard"][0]["contributor"] not in ("(all)",):
        lines += ["", c("  Contributors (worst first)", "1"),
                  f"  {'contributor':<24}{'files':>6}{'hours':>8}{'usable':>9}  main loss"]
        for r in report["scorecard"][:25]:
            loss = next(iter(r["unusable_minutes"].items()), None)
            lines.append(f"  {r['contributor'][:23]:<24}{r['files']:>6}{r['hours']:>8.2f}{r['usable_share']:>9.0%}  "
                         + (f"{loss[0]} {loss[1]:g} min" if loss else "-")
                         + (f", {r['cross_contributor_duplicates']} cross-contributor dup" if r["cross_contributor_duplicates"] else ""))
    flagged = [f for f in report["files"] if f["status"] != "ok"]
    if flagged:
        lines += ["", c("  Files with findings", "1")]
        for f in flagged[:40]:
            tag = c("CRIT", "31") if f["status"] == "critical" else c("WARN", "33")
            lines.append(f"  {tag} {Path(f['path']).name}  ({f['usable_s']:.0f}/{f['duration_s']:.0f} s usable)")
            for i in f["issues"][:4]:
                if i["severity"] != "info":
                    lines.append(f"       - {i['message']}")
        if len(flagged) > 40:
            lines.append(f"  … {len(flagged) - 40} more in the JSON/HTML report")
    lines += ["", "  Thresholds are calibrated on test clips; check a few flagged files by eye before rejecting work."]
    return "\n".join(lines)


def render_html(report: dict) -> str:
    import html as H

    s = report["summary"]

    def esc(x):
        return H.escape(str(x))

    def pct(x):
        return f"{x:.0%}"

    rows = "".join(
        f"<tr><td>{esc(r['contributor'])}</td><td class=n>{r['files']}</td><td class=n>{r['hours']:.2f}</td>"
        f"<td class=n>{r['usable_hours']:.2f}</td><td class=n><b>{pct(r['usable_share'])}</b></td>"
        f"<td class=n>{r['critical_files']}</td><td class=n>{r['cross_contributor_duplicates']}</td>"
        f"<td>{esc(', '.join(f'{k} {v:g} min' for k, v in list(r['unusable_minutes'].items())[:4]) or '-')}</td></tr>"
        for r in report["scorecard"])
    dup_rows = "".join(
        f"<tr><td>{esc(d['kind'].replace('_', ' '))}</td><td>{esc(Path(d['a']).name)}<br><small>{esc(d.get('contributor_a', ''))}</small></td>"
        f"<td>{esc(Path(d['b']).name)}<br><small>{esc(d.get('contributor_b', ''))}</small></td>"
        f"<td class=n>{'' if d.get('overlap_s') is None else esc(d['overlap_s'])}</td>"
        f"<td>{'yes' if d.get('cross_contributor') else ''}</td></tr>" for d in report["duplicates"])
    file_rows = []
    for f in report["files"]:
        issues = "".join(f"<li class={i['severity']}>{esc(i['message'])}</li>" for i in f["issues"]) or "<li class=ok>no findings</li>"
        m = f["meta"]
        file_rows.append(
            f"<tr class={f['status']}><td><b>{esc(Path(f['path']).name)}</b><br><small>{esc(f['contributor'])} · "
            f"{esc(m.get('width'))}x{esc(m.get('height'))} · {esc(round(m.get('fps') or 0, 2))} fps · {esc(m.get('codec'))}</small></td>"
            f"<td class=n>{f['usable_s']:.0f} / {f['duration_s']:.0f} s</td><td><ul>{issues}</ul></td></tr>")
    unusable = ", ".join(f"{k} {v:g} min" for k, v in s["unusable_minutes"].items()) or "none"
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>rdq video report</title><style>
:root{{--bg:#fbfaf8;--fg:#1d1f23;--muted:#62666e;--line:#e3e1dc;--card:#fff;--crit:#b42318;--warn:#b54708;--ok:#067647}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141518;--fg:#ececec;--muted:#a2a6ad;--line:#2b2d33;--card:#1c1d21;--crit:#f97066;--warn:#fdb022;--ok:#47cd89}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);margin:0;padding:24px 16px}}
main{{max-width:1100px;margin:0 auto}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:28px 0 8px}}
.muted{{color:var(--muted)}} .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}} .tile b{{display:block;font-size:22px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}} th{{font-size:13px;color:var(--muted)}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}} ul{{margin:0;padding-left:18px}} li.critical{{color:var(--crit)}}
li.warning{{color:var(--warn)}} li.info{{color:var(--muted)}} li.ok{{color:var(--ok)}} .wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Video delivery report</h1><div class=muted>rdq check-video · {s['files']} files · {s['contributors']} contributor(s)</div>
<div class=tiles><div class=tile><b>{s['hours']:.2f} h</b>footage</div><div class=tile><b>{s['usable_hours']:.2f} h</b>usable ({pct(s['usable_share'])})</div>
<div class=tile><b>{s['files_critical']}</b>files with critical findings</div><div class=tile><b>{s['cross_contributor_duplicates']}</b>duplicates across contributors</div></div>
<p>Unusable time by reason: {esc(unusable)}.</p>
<h2>Contributors, worst first</h2><div class=wrap><table><tr><th>Contributor</th><th>Files</th><th>Hours</th><th>Usable h</th><th>Usable</th><th>Critical files</th><th>Cross-contrib. dups</th><th>Main losses</th></tr>{rows}</table></div>
<h2>Duplicates and overlaps</h2>{'<div class=wrap><table><tr><th>Kind</th><th>File A</th><th>File B (copy)</th><th>Overlap s</th><th>Across contributors</th></tr>' + dup_rows + '</table></div>' if dup_rows else '<p class=muted>None found.</p>'}
<h2>Files</h2><div class=wrap><table><tr><th>File</th><th>Usable</th><th>Findings</th></tr>{''.join(file_rows)}</table></div>
<p class=muted>Thresholds are calibrated on test clips. Check a few flagged files by eye before using a finding to reject work.</p>
</main></body></html>"""
