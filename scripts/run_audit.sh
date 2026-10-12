#!/bin/bash
# Runs the public audit of LeRobot datasets on Hugging Face.
#   Stage 1: licence + metadata scan of the most-downloaded LeRobot datasets (small files only)
#   Stage 2: full quality checks on a sample (downloads tables, never videos, deletes after)
# Safe to stop with Ctrl-C and run again: both stages resume where they left off.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

PY=""
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then PY="$cand"; break; fi
  fi
done
if [ -z "$PY" ]; then echo "Need Python 3.9 or newer (brew install python)"; exit 1; fi

if [ ! -x .venv/bin/rdq ]; then
  echo "Setting up (one time)…"
  "$PY" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q .
fi

SCAN_LIMIT="${SCAN_LIMIT:-3000}"
AUDIT_N="${AUDIT_N:-150}"

# caffeinate keeps the Mac awake while this runs.
caffeinate -i bash -c "
  .venv/bin/rdq scan --limit $SCAN_LIMIT --out results &&
  .venv/bin/rdq audit --from-scan results --n $AUDIT_N --max-mb 250 --out results
" 2>&1 | tee -a results/run.log

echo ""
echo "Done. Results are in ~/robodata/results — tell Claude it finished."
