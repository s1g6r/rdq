"""Check suspected duplicate episodes: are the robot data identical, and do the frames match?

    python experiments/dup_verify.py owner/name A-B,C-D [times]
Prints per pair: lengths, max |difference| of state and action over the common length, whether timestamps match,
and a contact sheet (head camera of A over B) as base64 JPEG chunks.
"""
import base64
import json
import sys

import av
import cv2
import numpy as np
from huggingface_hub import snapshot_download

from rdq.dataset import load_dataset
from rdq.video import _video_file

repo, pairs = sys.argv[1], [tuple(int(x) for x in p.split("-")) for p in sys.argv[2].split(",")]
times = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,7,10").split(",")]
root = snapshot_download(repo, repo_type="dataset", allow_patterns=["meta/*", "meta/**", "data/**", "README.md", "videos/**"])
ds = load_dataset(root)
cam = ds.video_keys[0]


def frames(e):
    path, a, b = _video_file(ds, cam, ds.episodes[e])
    want, got = [a + t for t in times], {}
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            ts = float(fr.pts * st.time_base)
            for w in want:
                if w not in got and ts >= w - 1e-3:
                    got[w] = cv2.resize(fr.to_ndarray(format="bgr24"), (160, 120))
            if ts > want[-1] + 1:
                break
    return np.hstack([got.get(w, np.zeros((120, 160, 3), np.uint8)) for w in want])


rows = []
for a, b in pairs:
    ea, eb = ds.episodes[a], ds.episodes[b]
    out = {"pair": [a, b], "len": [ea.length, eb.length], "tasks": [ds.tasks.get(int(ea.task_index[0])) if ea.task_index is not None else None,
                                                                 ds.tasks.get(int(eb.task_index[0])) if eb.task_index is not None else None]}
    for key in (ds.state_key, ds.action_key):
        if key and key in ea.features and key in eb.features:
            n = min(len(ea.features[key]), len(eb.features[key]))
            d = np.abs(ea.features[key][:n] - eb.features[key][:n])
            out[key] = {"max_abs_diff": float(np.nanmax(d)), "range": float(np.nanmax(np.ptp(ea.features[key], axis=0)))}
    out["meta"] = {k: [str(ea.meta.get(k)), str(eb.meta.get(k))] for k in ("length", "dataset_from_index", "tasks") if k in ea.meta}
    print(f"::notice title=dv-{a}-{b}::" + json.dumps(out)[:3800])
    row = np.vstack([frames(a), frames(b)])
    cv2.putText(row, f"ep{a} (top) vs ep{b} (bottom) {cam.split('.')[-1]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    rows.append(row)
img = np.vstack(rows)
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
b64 = base64.b64encode(buf.tobytes()).decode()
print("::notice title=sheet-meta::" + f"{img.shape} {len(b64)}")
for i in range(0, len(b64), 3800):
    print(f"::notice title=sheet-{i // 3800:03d}::" + b64[i:i + 3800])
