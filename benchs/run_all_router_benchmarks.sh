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
#   ./benchs/run_all_router_benchmarks.sh
#
#   # Single dataset:
#   ./benchs/run_all_router_benchmarks.sh sift1m
#
#   # Multiple datasets:
#   ./benchs/run_all_router_benchmarks.sh sift1m gist1m deep1m
#
#   # Specific benchmarks / index types via env:
#   BENCHMARKS="construction recall_k10 recall_k20" \
#     ./benchs/run_all_router_benchmarks.sh sift1m
#
#   INDEX_TYPES="suco shg" \
#     ./benchs/run_all_router_benchmarks.sh openai1m
#
# Environment variables:
#   DATA_DIR     — dataset root        (default: /Users/dhm/Documents/data)
#   INDEX_DIR    — saved indices       (default: /Users/dhm/Documents/indices)
#   OUTPUT_DIR   — result JSONs        (default: benchs/results_router)
#   BENCHMARKS   — space-separated     (default: all)
#   INDEX_TYPES  — suco / shg / cspg   (default: "suco shg cspg")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-/Users/dhm/Documents/data}"
INDEX_DIR="${INDEX_DIR:-/Users/dhm/Documents/indices}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_router}"
BENCHMARKS="${BENCHMARKS:-all}"
INDEX_TYPES="${INDEX_TYPES:-suco shg cspg}"

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
