#!/bin/bash
#
# run_all_router_benchmarks.sh
#
# Runs the router-training benchmark suite (SuCo / SHG / CSPG with paper
# defaults) across all 11 datasets, writes one results_<dataset>.json per
# dataset, and saves indices to INDEX_DIR for reuse.
#
# Usage:
#   chmod +x benchs/run_all_router_benchmarks.sh
#   ./benchs/run_all_router_benchmarks.sh                    # all 11 datasets
#   ./benchs/run_all_router_benchmarks.sh sift1m             # one dataset
#   ./benchs/run_all_router_benchmarks.sh sift1m gist1m deep1m   # several
#
# Subset of benchmarks / indexes via env vars:
#   BENCHMARKS="construction recall_k10 recall_k20" \
#     ./benchs/run_all_router_benchmarks.sh sift1m
#
#   INDEX_TYPES="suco shg" \
#     ./benchs/run_all_router_benchmarks.sh openai1m
#
# Available datasets (positional args):
#   sift1m sift10m gist1m deep1m deep10m spacev10m
#   msong enron openai1m msturing10m uqv
#
# Available index types:
#   suco shg cspg hnsw32 hnsw48
#
# Environment variables:
#   DATA_DIR     — dataset root                  (default: /Users/dhm/Documents/data)
#   INDEX_DIR    — saved indices                 (default: /Users/dhm/Documents/indices)
#   OUTPUT_DIR   — result JSONs                  (default: benchs/results_router)
#   BENCHMARKS   — space-separated               (default: all)
#   INDEX_TYPES  — suco/shg/cspg/hnsw32/hnsw48   (default: "suco shg cspg hnsw32 hnsw48")
#
# Auto-prep: when deep1m or deep10m is in the dataset list, the script invokes
# prepare_deep1m.py to write deep1b/base.fvecs and deep1b/learn.fvecs (only if
# they are missing or smaller than required). Skip otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-/Users/dhm/Documents/data}"
INDEX_DIR="${INDEX_DIR:-/Users/dhm/Documents/indices}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_router}"
BENCHMARKS="${BENCHMARKS:-all}"
INDEX_TYPES="${INDEX_TYPES:-suco shg cspg hnsw32 hnsw48}"

BENCH_SCRIPT="${SCRIPT_DIR}/bench_router_paper.py"

ALL_DATASETS=(
    sift1m sift10m gist1m
    deep1m deep10m
    spacev10m
    msong enron openai1m
    msturing10m uqv
)

read -r -a BENCHMARK_ARGS  <<< "${BENCHMARKS}"
read -r -a INDEX_TYPE_ARGS <<< "${INDEX_TYPES}"

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

# Validate dataset names early.
for ds in "${DATASETS[@]}"; do
    found=0
    for valid in "${ALL_DATASETS[@]}"; do
        if [ "$ds" = "$valid" ]; then found=1; break; fi
    done
    if [ $found -eq 0 ]; then
        echo "Unknown dataset: '$ds'"
        echo "Valid choices: ${ALL_DATASETS[*]}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Auto-prepare deep1b/base.fvecs + learn.fvecs when deep* datasets are selected.
# Idempotent: skips if files already exist with sufficient size.
# Each fvecs row for d=96 occupies 4 + 96*4 = 388 bytes.
# ---------------------------------------------------------------------------
DEEP_FVECS_ROW_BYTES=388
file_size() {
    if stat -f%z "$1" >/dev/null 2>&1; then stat -f%z "$1"
    else stat -c%s "$1"; fi
}
prepare_deep_if_needed() {
    local nb_req=$1
    local nt_req=$2
    local deep_dir="${DATA_DIR%/}/deep1b"
    local base_file="${deep_dir}/base.fvecs"
    local learn_file="${deep_dir}/learn.fvecs"
    local need_base=$(( nb_req * DEEP_FVECS_ROW_BYTES ))
    local need_learn=$(( nt_req * DEEP_FVECS_ROW_BYTES ))

    local have_base=0; local have_learn=0
    [ -f "$base_file" ]  && have_base=$(file_size "$base_file")
    [ -f "$learn_file" ] && have_learn=$(file_size "$learn_file")

    if [ "$have_base" -ge "$need_base" ] && [ "$have_learn" -ge "$need_learn" ]; then
        echo "Deep base.fvecs / learn.fvecs already prepared (≥${nb_req} base, ≥${nt_req} learn)"
        return 0
    fi

    if [ ! -f "${deep_dir}/base00" ] || [ ! -f "${deep_dir}/learn00" ]; then
        echo "ERROR: ${deep_dir}/base00 or learn00 missing — required to build base.fvecs / learn.fvecs"
        echo "Download with benchs/downloadDeep1B.py first."
        exit 1
    fi

    echo ""
    echo "############################################################"
    echo "# Auto-prepare deep1b: nb=${nb_req}, nt=${nt_req}"
    echo "############################################################"
    python3 "${SCRIPT_DIR}/prepare_deep1m.py" \
        --data-dir "${DATA_DIR}" \
        --nb "${nb_req}" --nt "${nt_req}"
}

need_nb=0; need_nt=0
for ds in "${DATASETS[@]}"; do
    case "$ds" in
        deep1m)  if [ $need_nb -lt 1000000  ];  then need_nb=1000000;  need_nt=500000;  fi ;;
        deep10m) if [ $need_nb -lt 10000000 ];  then need_nb=10000000; need_nt=1000000; fi ;;
    esac
done
if [ $need_nb -gt 0 ]; then
    prepare_deep_if_needed "$need_nb" "$need_nt"
fi

echo "============================================================"
echo "Router Benchmark Suite (SuCo / SHG / CSPG, paper defaults)"
echo "============================================================"
echo "  Data dir:    ${DATA_DIR}"
echo "  Index dir:   ${INDEX_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Benchmarks:  ${BENCHMARK_ARGS[*]}"
echo "  Index types: ${INDEX_TYPE_ARGS[*]}"
echo "============================================================"

mkdir -p "${INDEX_DIR}" "${OUTPUT_DIR}"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# Dataset: ${ds}"
    echo "############################################################"

    LOG_FILE="${OUTPUT_DIR}/log_router_${ds}.txt"

    python3 "${BENCH_SCRIPT}" \
        --data-dir   "${DATA_DIR}" \
        --index-dir  "${INDEX_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset    "${ds}" \
        --benchmark  "${BENCHMARK_ARGS[@]}" \
        --index-type "${INDEX_TYPE_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "Done: ${ds}  (log: ${LOG_FILE})"
done

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "  Results: ${OUTPUT_DIR}/results_*.json"
echo "  Logs:    ${OUTPUT_DIR}/log_router_*.txt"
echo "============================================================"
