"""Why a frozen stretch is judged 'moving': per flagged episode, the robot's range during the stretch relative to
its whole-episode range (per channel too), and whether another camera kept changing.

    python experiments/frozen_diag.py owner/name [camera]
"""
import json
import os
import sys

import numpy as np

from rdq import rawvideo as rv
from rdq.checks import _extent
from rdq.dataset import load_dataset
from huggingface_hub import snapshot_download
from rdq.video import _video_file

ds_id = sys.argv[1]
root = snapshot_download(ds_id, repo_type="dataset", allow_patterns=["meta/*", "data/*", "README.md", "videos/*", "videos/**"])
ds = load_dataset(root)
key = ds.state_key or ds.action_key
print("::notice title=keys::" + json.dumps({"state": ds.state_key, "action": ds.action_key, "cams": ds.video_keys,
                                          "names": (ds.features.get(key) or {}).get("names")})[:3000])
spans = {}
for cam in ds.video_keys:
    recs = {}
    for ep in sorted(ds.episodes.values(), key=lambda e: e.index)[:60]:
        path, a, b = _video_file(ds, cam, ep)
        if path is None or not path.exists():
            continue
        m = recs.setdefault(str(path), rv.measure_file(str(path)))
        r = rv.segment_record(m, a, b, f"ep{ep.index}")
        rep = rv.check_video_batch([], records=[r], workers=1)
        for i in rep["files"][0]["issues"]:
            if i["check"] == "video.frozen":
                spans.setdefault(cam, {})[ep.index] = i.get("spans") or []
rows = []
cam0 = next(iter(spans), None)
for e, sp in list((spans.get(cam0) or {}).items())[:12]:
    ep = ds.episodes[e]
    x = np.asarray(ep.features[key], float)
    t = np.asarray(ep.timestamps, float) - float(ep.timestamps[0])
    whole = _extent(x)
    per = []
    for a, b in sp:
        msk = (t >= a) & (t < b)
        seg = x[msk]
        per.append({"span": [round(a, 1), round(b, 1)], "n": int(msk.sum()), "ratio": round(_extent(seg) / max(whole, 1e-12), 3),
                    "chan_ptp": [round(float(v), 4) for v in np.ptp(seg, axis=0)] if len(seg) else [],
                    "whole_ptp": [round(float(v), 4) for v in np.ptp(x, axis=0)]})
    others = {c: (spans.get(c) or {}).get(e) for c in spans if c != cam0}
    rows.append({"ep": e, "cam": cam0, "spans": per[:3], "other_cam_frozen_spans": others})
for i, r in enumerate(rows):
    print(f"::notice title=fd-{i}::" + json.dumps(r)[:3800])
