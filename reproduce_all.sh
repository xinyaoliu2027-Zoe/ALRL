#!/usr/bin/env bash
#
# reproduce_all.sh — one-command end-to-end reproduction of the
# ALRL paper.
#
#   1. Verifies the python env and installs missing dependencies
#   2. Runs the full 5-seed × 8-config × 7-task experimental sweep
#   3. Aggregates per-seed pickles into mean ± std tables
#   4. Regenerates all paper figures
#
# Wall-clock time: ~4–5 hours on a CPU-only Apple-M-class laptop.
# Run `bash reproduce_minimal.sh` first if you want a 17-min smoke
# test before committing to the full run.
#
# Usage:
#   bash reproduce_all.sh                 # full reproduction
#   bash reproduce_all.sh --seeds 0       # run a single seed
#   bash reproduce_all.sh --skip-deps     # skip pip install step
#
# Environment variables:
#   ANALYSIS_DIR  — where aggregate_seeds.py / make_figures.py live
#                   (default: $HOME/Documents/Claude/Projects/ALML)
#

set -euo pipefail

# ---- Configurable paths ----------------------------------------------
ANALYSIS_DIR="${ANALYSIS_DIR:-$HOME/Documents/Claude/Projects/ALML}"
SEEDS_DEFAULT="0 1 2 3 4"
LOG_FILE="run_log_$(date +%Y%m%d_%H%M%S).txt"

# ---- Parse arguments -------------------------------------------------
SEEDS="$SEEDS_DEFAULT"
SKIP_DEPS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      shift
      SEEDS=""
      while [[ $# -gt 0 && ! "$1" == --* ]]; do
        SEEDS="$SEEDS $1"
        shift
      done
      ;;
    --skip-deps)
      SKIP_DEPS=1
      shift
      ;;
    -h|--help)
      sed -n '2,/^$/p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done
SEEDS="$(echo "$SEEDS" | xargs)"  # trim

# ---- Helpers ---------------------------------------------------------
log()  { printf '\033[1;36m[reproduce]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[fail]\033[0m %s\n' "$*"; exit 1; }

# ---- Sanity check ----------------------------------------------------
log "ALRL one-command reproduction starting at $(date)"
log "Working directory: $(pwd)"
log "Seeds: $SEEDS"
log "Analysis directory: $ANALYSIS_DIR"
log "Log file: $LOG_FILE"

if [[ ! -f "forgetting_prevention_comparison.py" ]]; then
  fail "forgetting_prevention_comparison.py not found. Run this script from the fyp_code directory."
fi

if [[ ! -d "$ANALYSIS_DIR" ]]; then
  warn "Analysis directory $ANALYSIS_DIR does not exist."
  warn "Will skip aggregate / figure steps. Set ANALYSIS_DIR=... to enable them."
  HAS_ANALYSIS=0
else
  HAS_ANALYSIS=1
fi

# ---- Step 1. Install dependencies -----------------------------------
if [[ "$SKIP_DEPS" -eq 0 ]]; then
  log "Step 1/4 — Installing Python dependencies"
  if ! python -c "import torch, sklearn, numpy, matplotlib, pandas" 2>/dev/null; then
    log "  Some core deps missing. Running 'pip install -r requirements.txt' ..."
    pip install -q -r requirements.txt
  else
    log "  Core deps present. Verifying optional ones ..."
    pip install -q --upgrade xlrd openpyxl pandas scipy 2>/dev/null || true
  fi
else
  log "Step 1/4 — Skipping dependency installation (--skip-deps)"
fi

# ---- Step 2. Multi-seed sweep ---------------------------------------
log "Step 2/4 — Running multi-seed experimental sweep"
log "  Total runs: $(echo $SEEDS | wc -w) seeds × 8 configurations = $(($(echo $SEEDS | wc -w) * 8))"
log "  Estimated time: ~$(echo $SEEDS | wc -w | xargs -I{} expr {} \* 50) min"

# Use caffeinate on macOS to prevent sleep
CAFFEINATE_CMD=""
if [[ "$(uname)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1; then
  CAFFEINATE_CMD="caffeinate -i"
  log "  caffeinate detected — system sleep will be prevented"
fi

# Filter out Intel MKL warnings (noise on Intel Macs)
$CAFFEINATE_CMD python -u forgetting_prevention_comparison.py \
    --seeds $SEEDS --skip-analyze 2>&1 \
  | grep --line-buffered -v "Intel MKL WARNING\|Intel oneAPI" \
  | tee "$LOG_FILE"

log "  Sweep complete. Results in: experiment_results/"

# ---- Step 3. Aggregate seeds ----------------------------------------
if [[ "$HAS_ANALYSIS" -eq 1 ]]; then
  log "Step 3/4 — Aggregating per-seed results into mean ± std tables"
  python "$ANALYSIS_DIR/aggregate_seeds.py" --fyp "$(pwd)"
  log "  Aggregated outputs in: $ANALYSIS_DIR/aggregated/"
else
  log "Step 3/4 — Skipped (analysis directory not found)"
fi

# ---- Step 4. Generate figures ---------------------------------------
if [[ "$HAS_ANALYSIS" -eq 1 ]]; then
  log "Step 4/4 — Regenerating publication figures"
  python "$ANALYSIS_DIR/make_figures.py" --fyp "$(pwd)"
  log "  Figures in: $ANALYSIS_DIR/figures/"
else
  log "Step 4/4 — Skipped (analysis directory not found)"
fi

# ---- Summary ---------------------------------------------------------
log "==============================================================="
log "Reproduction complete at $(date)"
log "Result pickles : experiment_results/"
log "Aggregated tables: $ANALYSIS_DIR/aggregated/"
log "Paper figures   : $ANALYSIS_DIR/figures/"
log "Run log         : $LOG_FILE"
log "==============================================================="
