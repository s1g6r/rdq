"""Synthetic camera footage with known defects, for tests and the check-video benchmark.

A clip is a textured scene seen by a slowly moving camera, with a few objects moving in it and
sensor noise on every frame. Defects are injected at known times:

    freeze   the same frame repeated (stalled camera)
    black    a black frame (lens cap, failed exposure)
    covered  a flat, featureless frame (hand or sleeve over the lens)
    blur     defocus (Gaussian blur) while the camera is still
    idle     camera and objects stop; only sensor noise changes
    drop     frames missing from the timestamps (a gap)

Copies for the duplicate checks (re-encode, trim, mirror, splice) are made with the ffmpeg CLI.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ClipSpec:
    seconds: float = 30.0
    fps: int = 30
    width: int = 640
    height: int = 360
    seed: int = 0
    defects: dict = field(default_factory=dict)   # name -> (start_s, end_s); "drop" -> (start_s, end_s)
    crf: int = 23
    vfr_from: float | None = None                  # halve the frame rate from this time on
    scene_seed: int | None = None                  # same value = same scene (another take in the same place)
    static_camera: bool = False                    # fixed camera: only the objects move


def _texture(rng, h, w):
    import cv2

    base = rng.random((h // 8, w // 8)).astype(np.float32)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 2)
    tex = 0.65 * base + 0.35 * fine
    tex = (tex - tex.min()) / (tex.max() - tex.min() + 1e-9)
    return (40 + 180 * tex).astype(np.float32)


def render_clip(path: str | Path, spec: ClipSpec) -> Path:
    """Write an H.264 MP4 with the requested defects."""
    import av
    import cv2

    rng = np.random.default_rng(spec.seed)
    W, H = spec.width, spec.height
    if spec.scene_seed is None:
        world = _texture(rng, H + 200, W + 400)
    else:
        world = _texture(np.random.default_rng(spec.scene_seed), H + 200, W + 400)
    n_obj = 4
    obj = [dict(r=int(rng.integers(18, 40)), c=float(rng.uniform(60, 230)),
                px=rng.uniform(0.1, 0.9), py=rng.uniform(0.2, 0.8), ax=rng.uniform(0.05, 0.25),
                ay=rng.uniform(0.05, 0.2), fx=rng.uniform(0.05, 0.3), fy=rng.uniform(0.05, 0.3),
                ph=rng.uniform(0, 6.28)) for _ in range(n_obj)]
    d = spec.defects

    def active(name, t):
        span = d.get(name)
        return span is not None and span[0] <= t < span[1]

    out = av.open(str(path), "w")
    st = out.add_stream("libx264", rate=spec.fps)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": str(spec.crf), "preset": "veryfast"}
    st.time_base = __import__("fractions").Fraction(1, spec.fps)
    n = int(spec.seconds * spec.fps)
    frozen = None
    idle_state = None
    for i in range(n):
        t = i / spec.fps
        if active("drop", t):
            continue
        if spec.vfr_from is not None and t >= spec.vfr_from and i % 2:
            continue
        if active("idle", t):
            if idle_state is None:
                idle_state = t
            tt = idle_state
        else:
            idle_state = None
            tt = t
        if spec.static_camera:
            cam_x, cam_y = 200, 100
        else:
            cam_x = 200 + 150 * np.sin(0.21 * tt + spec.seed)
            cam_y = 100 + 60 * np.sin(0.13 * tt + 2 * spec.seed)
        frame = world[int(cam_y):int(cam_y) + H, int(cam_x):int(cam_x) + W].copy()
        for o in obj:
            x = int(W * (o["px"] + o["ax"] * np.sin(2 * np.pi * o["fx"] * tt + o["ph"])))
            y = int(H * (o["py"] + o["ay"] * np.sin(2 * np.pi * o["fy"] * tt + 1.3 * o["ph"])))
            cv2.circle(frame, (x, y), o["r"], o["c"], -1)
            cv2.circle(frame, (x, y), o["r"] // 2, 255 - o["c"], -1)
        if active("blur", t):
            frame = cv2.GaussianBlur(frame, (0, 0), 3.5)
        frame = frame + rng.normal(0, 2.0, frame.shape).astype(np.float32)
        if active("black", t):
            frame = np.full((H, W), 6.0, np.float32) + rng.normal(0, 1.0, (H, W)).astype(np.float32)
        if active("covered", t):
            frame = np.full((H, W), 150.0, np.float32) + np.linspace(-3, 3, W, dtype=np.float32)[None, :]
        img = np.clip(frame, 0, 255).astype(np.uint8)
        if active("freeze", t):
            if frozen is None:
                frozen = img
            img = frozen
        else:
            frozen = None
        rgb = np.repeat(img[:, :, None], 3, axis=2)
        vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        vf.pts = i
        for pkt in st.encode(vf):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    return Path(path)


def _ffmpeg(args: list[str]) -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is required to make the duplicate variants")
    subprocess.run([exe, "-v", "error", "-y", *args], check=True)


def reencode(src, dst, width=None, crf=30, fps=None, mirror=False, brightness=0.0, trim=None, container=None):
    """A re-encoded copy: smaller, lower quality, optionally mirrored, brighter/darker, trimmed or at another fps."""
    vf = []
    if width:
        vf.append(f"scale={width}:-2")
    if mirror:
        vf.append("hflip")
    if brightness:
        vf.append(f"eq=brightness={brightness}")
    if fps:
        vf.append(f"fps={fps}")
    args = []
    if trim:
        args += ["-ss", str(trim[0]), "-to", str(trim[1])]
    args += ["-i", str(src)]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-an", str(dst)]
    _ffmpeg(args)
    return Path(dst)


def remux(src, dst):
    """Same coded stream, different container (file hash changes, frames don't)."""
    _ffmpeg(["-i", str(src), "-c", "copy", str(dst)])
    return Path(dst)


def splice(a, a_range, b, b_range, dst):
    """b[b_range] appended to a[a_range], re-encoded: a partial overlap with both sources."""
    _ffmpeg(["-ss", str(a_range[0]), "-to", str(a_range[1]), "-i", str(a),
             "-ss", str(b_range[0]), "-to", str(b_range[1]), "-i", str(b),
             "-filter_complex", "[0:v]scale=640:360,setsar=1[x];[1:v]scale=640:360,setsar=1[y];[x][y]concat=n=2:v=1[v]",
             "-map", "[v]", "-c:v", "libx264", "-crf", "24", "-preset", "veryfast", str(dst)])
    return Path(dst)


def truncate(src, dst, keep=0.45):
    data = Path(src).read_bytes()
    Path(dst).write_bytes(data[: int(len(data) * keep)])
    return Path(dst)


def corrupt_bytes(src, dst, n=40, size=64, seed=0):
    rng = np.random.default_rng(seed)
    data = bytearray(Path(src).read_bytes())
    lo, hi = len(data) // 5, len(data) * 4 // 5
    for _ in range(n):
        o = int(rng.integers(lo, hi - size))
        data[o:o + size] = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
    Path(dst).write_bytes(bytes(data))
    return Path(dst)


def faststart(src, dst):
    _ffmpeg(["-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)])
    return Path(dst)
