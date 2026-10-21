"""Contact sheet of a few frames per camera for chosen episodes, printed as base64 JPEG chunks (CI annotations).

    python experiments/frame_sheet.py owner/name EP[,EP] t1,t2,...
"""
import base64
import sys

import av
import cv2
import numpy as np
from huggingface_hub import snapshot_download

from rdq.dataset import load_dataset
from rdq.video import _video_file

ds_id, eps, times = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], [float(x) for x in sys.argv[3].split(",")]
root = snapshot_download(ds_id, repo_type="dataset", allow_patterns=["meta/*", "data/*", "README.md", "videos/**"])
ds = load_dataset(root)
rows = []
for e in eps:
    for cam in ds.video_keys:
        path, a, b = _video_file(ds, cam, ds.episodes[e])
        want = [a + t for t in times]
        got = {}
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            for fr in c.decode(st):
                ts = float(fr.pts * st.time_base)
                for w in want:
                    if w not in got and ts >= w - 1e-3:
                        got[w] = fr.to_ndarray(format="bgr24")
                if len(got) == len(want) or ts > want[-1] + 1:
                    break
        tiles = [cv2.resize(got.get(w, np.zeros((128, 128, 3), np.uint8)), (128, 128)) for w in want]
        row = np.hstack(tiles)
        cv2.putText(row, f"ep{e} {cam.split('.')[-1]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        rows.append(row)
img = np.vstack(rows)
ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
b = base64.b64encode(buf.tobytes()).decode()
print("::notice title=sheet-meta::" + f"{img.shape} {len(b)} times={times}")
for i in range(0, len(b), 3800):
    print(f"::notice title=sheet-{i // 3800:03d}::" + b[i:i + 3800])
