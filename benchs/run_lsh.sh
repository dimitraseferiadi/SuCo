#!/usr/bin/env bash
# run_all_benchmarks.sh
# Runs the full SuCo benchmark suite across SIFT1M, GIST1M, Deep1?,
# SIFT10M, and SpaceV10M.
#
# Usage:
#   ./benchs/run_all_benchmarks.sh [options]
#
# Options:
#   --data-dir DIR       Root data directory (default: data/)
#   --index-dir DIR      Where to cache built index files (default: data/indices/)
#   --log-dir DIR        Where to write the log file (default: logs/)
#   --nb NB              Deep1? dataset size: 1000000, 10000000, 100000000 (default: 1000000)
#   --sift10m-mat PATH   Path to SIFT10Mfeatures.mat (default: data/SIFT10M/SIFT10Mfeatures.mat)
#   --spacev10m-dir DIR  Directory with SpaceV10M .i8bin files (default: data/spacev10m/)
#   --make-figures       Generate figures from the benchmark log after run
#   --linux-log PATH     Optional Linux bench_all log used for cross-platform plots
#   --dry-run            Print commands without running them
#   --sift-only          Run only SIFT1M benchmarks
#   --gist-only          Run only GIST1M benchmarks
#   --deep-only          Run only Deep1? benchmarks
#   --sift10m-only       Run only SIFT10M benchmarks
#   --spacev10m-only     Run only SpaceV10M benchmarks
#   -h, --help           Show this help and exit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- defaults ----------------------------------------------------------------
DATA_DIR="$ROOT_DIR/data/"
INDEX_DIR="$ROOT_DIR/data/indices/"
LOG_DIR="$ROOT_DIR/logs/"
DEEP_NB=1000000
SIFT10M_MAT="$ROOT_DIR/data/SIFT10M/SIFT10Mfeatures.mat"
SPACEV10M_DIR="$ROOT_DIR/data/spacev10m/"
DRY_RUN=0
MAKE_FIGURES=0
LINUX_LOG_COMPARE=""
RUN_SIFT=1
RUN_GIST=1
RUN_DEEP=1
RUN_SIFT10M=1
RUN_SPACEV10M=1

# ---- parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)       DATA_DIR="$2";       shift 2 ;;
        --index-dir)      INDEX_DIR="$2";      shift 2 ;;
        --log-dir)        LOG_DIR="$2";        shift 2 ;;
        --nb)             DEEP_NB="$2";        shift 2 ;;
        --sift10m-mat)    SIFT10M_MAT="$2";    shift 2 ;;
        --spacev10m-dir)  SPACEV10M_DIR="$2";  shift 2 ;;
        --make-figures)   MAKE_FIGURES=1;       shift   ;;
        --linux-log)      LINUX_LOG_COMPARE="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=1;           shift   ;;
        --sift-only)      RUN_GIST=0; RUN_DEEP=0; RUN_SIFT10M=0; RUN_SPACEV10M=0; shift ;;
        --gist-only)      RUN_SIFT=0; RUN_DEEP=0; RUN_SIFT10M=0; RUN_SPACEV10M=0; shift ;;
        --deep-only)      RUN_SIFT=0; RUN_GIST=0; RUN_SIFT10M=0; RUN_SPACEV10M=0; shift ;;
        --sift10m-only)   RUN_SIFT=0; RUN_GIST=0; RUN_DEEP=0; RUN_SPACEV10M=0; shift ;;
        --spacev10m-only) RUN_SIFT=0; RUN_GIST=0; RUN_DEEP=0; RUN_SIFT10M=0;  shift ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- set up paths ------------------------------------------------------------
VENV="$ROOT_DIR/.venv/bin/activate"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$INDEX_DIR" "$LOG_DIR"
    LOG_FILE="$LOG_DIR/bench_all_${TIMESTAMP}.log"
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "Log: $LOG_FILE"
fi

# ---- activate venv -----------------------------------------------------------
# shellcheck disable=SC1090
source "$VENV"

PYTHON="python"

# ---- helpers -----------------------------------------------------------------
section() {
    echo ""
    echo "##########################################################################"
    echo "##  $*"
    echo "##########################################################################"
}

run_bench() {
    echo ""
    echo ">>> $*"
    if [[ $DRY_RUN -eq 1 ]]; then
        return
    fi
    local start
    start=$(date +%s)
    "$@"
    local elapsed=$(( $(date +%s) - start ))
    echo "(done in ${elapsed}s)"
}

# ---- header ------------------------------------------------------------------
section "SuCo Full Benchmark Suite  —  $(date)"
echo "  data-dir     : $DATA_DIR"
echo "  index-dir    : $INDEX_DIR"
echo "  deep-nb      : $DEEP_NB"
echo "  sift10m-mat  : $SIFT10M_MAT"
echo "  spacev10m-dir: $SPACEV10M_DIR"
echo "  make-figures : $MAKE_FIGURES"
if [[ -n "$LINUX_LOG_COMPARE" ]]; then
    echo "  linux-log    : $LINUX_LOG_COMPARE"
fi
[[ $DRY_RUN -eq 1 ]] && echo "  *** DRY RUN — commands will not execute ***"

# ---- SIFT1M ------------------------------------------------------------------
if [[ $RUN_SIFT -eq 1 ]]; then
    section "SIFT1M  (d=128, nb=1M, nq=10K)"
    SIFT_BASE=("$PYTHON" "$SCRIPT_DIR/bench_suco_sift1m.py"
               --data-dir "$DATA_DIR"
               --index-path "$INDEX_DIR/sift1m.idx")

    run_bench "${SIFT_BASE[@]}" --lsh-sweep
fi

# ---- GIST1M ------------------------------------------------------------------
if [[ $RUN_GIST -eq 1 ]]; then
    section "GIST1M  (d=960, nb=1M, nq=1K)"
    GIST_BASE=("$PYTHON" "$SCRIPT_DIR/bench_suco_gist1m.py"
               --data-dir "$DATA_DIR"
               --index-path "$INDEX_DIR/gist1m.idx")

    run_bench "${GIST_BASE[@]}" --lsh-sweep
fi

# ---- Deep1? ------------------------------------------------------------------
if [[ $RUN_DEEP -eq 1 ]]; then
    case "$DEEP_NB" in
        1000000)   DEEP_TAG="Deep1M"   ;;
        10000000)  DEEP_TAG="Deep10M"  ;;
        100000000) DEEP_TAG="Deep100M" ;;
        *)         DEEP_TAG="Deep${DEEP_NB}" ;;
    esac
    section "${DEEP_TAG}  (d=96, nb=${DEEP_NB})"
    DEEP_BASE=("$PYTHON" "$SCRIPT_DIR/bench_suco_deep1b.py"
               --data-dir "$DATA_DIR"
               --nb "$DEEP_NB"
               --index-path "$INDEX_DIR/deep_nb${DEEP_NB}.idx")

    run_bench "${DEEP_BASE[@]}" --lsh-sweep
fi

# ---- SIFT10M -----------------------------------------------------------------
if [[ $RUN_SIFT10M -eq 1 ]]; then
    section "SIFT10M  (d=128, nb=10M)"
    SIFT10M_BASE=("$PYTHON" "$SCRIPT_DIR/bench_suco_sift10m.py"
                  --mat-path   "$SIFT10M_MAT"
                  --gt-path    "$INDEX_DIR/sift10m_gt.npy"
                  --index-path "$INDEX_DIR/sift10m.idx")

    run_bench "${SIFT10M_BASE[@]}" --lsh-sweep
fi

# ---- SpaceV10M ---------------------------------------------------------------
if [[ $RUN_SPACEV10M -eq 1 ]]; then
    section "SpaceV10M  (d=100, nb=10M)"
    SPACEV10M_BASE=("$PYTHON" "$SCRIPT_DIR/bench_suco_spacev10m.py"
                    --data-dir   "$SPACEV10M_DIR"
                    --gt-path    "$INDEX_DIR/spacev10m_gt.npy"
                    --index-path "$INDEX_DIR/spacev10m.idx")

    run_bench "${SPACEV10M_BASE[@]}" --lsh-sweep
fi

# ---- footer ------------------------------------------------------------------
section "All benchmarks finished  —  $(date)"

if [[ $DRY_RUN -eq 0 && $MAKE_FIGURES -eq 1 ]]; then
    section "Generate figures from logs"
    PLOT_CMD=("$PYTHON" "$SCRIPT_DIR/plot_benchmarks_from_logs.py" --mac-log "$LOG_FILE")
    if [[ -n "$LINUX_LOG_COMPARE" ]]; then
        PLOT_CMD+=(--linux-log "$LINUX_LOG_COMPARE")
    fi
    run_bench "${PLOT_CMD[@]}"
fi
