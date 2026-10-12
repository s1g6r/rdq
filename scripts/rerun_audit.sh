#!/bin/bash
# Second audit pass with the corrected checks, on a larger sample (300 datasets).
set -euo pipefail
cd "$(dirname "$0")/.."
caffeinate -i .venv/bin/rdq audit --from-scan results --n 300 --max-mb 250 --out results/audit3 2>&1 | grep -v "Fetching" | tee -a results/audit3.log
echo ""
echo "Done. The second audit finished."
