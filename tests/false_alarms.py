"""Run every check on many clean synthetic datasets and fail on any warning or critical finding.

    python tests/false_alarms.py --runs 60
"""

from __future__ import annotations

import argparse
import collections
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdq import synthetic as syn  # noqa: E402
from rdq.checks import run_checks  # noqa: E402
from rdq.dataset import load_dataset  # noqa: E402


def main(runs: int) -> int:
    alarms = collections.Counter()
    for seed in range(runs):
        for n, length in ((10, 120), (40, 300)):
            with tempfile.TemporaryDirectory() as d:
                eps = syn.make_episodes(n_episodes=n, length=length, seed=seed, lag=1 + seed % 4)
                ds = load_dataset(syn.write_dataset(d, eps))
                for f in run_checks(ds):
                    if f.severity in ("critical", "warning"):
                        alarms[(f.severity, f.check)] += 1
    total = runs * 2
    if alarms:
        for k, v in sorted(alarms.items()):
            print(f"{v:4d}  {k[0]:8s} {k[1]}")
        print(f"FAIL: false alarms on clean data ({total} datasets)")
        return 1
    print(f"OK: no warning or critical findings on {total} clean datasets")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=60)
    sys.exit(main(p.parse_args().runs))
