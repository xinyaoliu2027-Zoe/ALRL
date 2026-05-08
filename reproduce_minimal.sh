#!/usr/bin/env bash
#
# reproduce_minimal.sh — 17-min smoke test of the ALRL pipeline.
#
# Runs ONE configuration (Baseline) on ONE seed (0) over the full
# 7-task stream, just to verify your environment can run the code.
# After this completes, run `bash reproduce_all.sh` for the full
# 5-seed × 8-config sweep used in the paper.
#
# Wall-clock time: ~15-20 min on a CPU-only Apple-M-class laptop.
#

set -euo pipefail

log() { printf '\033[1;36m[smoke]\033[0m %s\n' "$*"; }

log "ALRL smoke test starting at $(date)"

if [[ ! -f "forgetting_prevention_comparison.py" ]]; then
  echo "[fail] forgetting_prevention_comparison.py not found." >&2
  echo "       Run this script from the fyp_code directory." >&2
  exit 1
fi

# Verify core deps
log "Step 1/2 — Checking dependencies"
python -c "import torch, sklearn, numpy" \
  || { echo "[fail] Missing deps. Run: pip install -r requirements.txt"; exit 1; }
log "  Core deps OK"

# Run a single config x single seed
log "Step 2/2 — Running Baseline + seed=0 (estimated 15-20 min)"
python -u forgetting_prevention_comparison.py \
    --seeds 0 --configs 0 --skip-analyze 2>&1 \
  | grep --line-buffered -v "Intel MKL WARNING\|Intel oneAPI"

log "==============================================================="
log "Smoke test complete at $(date)"
log "If you see a 'Lifelong Learning Performance Summary' block above"
log "with 7 tasks, the pipeline is working correctly."
log ""
log "Next: run 'bash reproduce_all.sh' for the full reproduction."
log "==============================================================="
