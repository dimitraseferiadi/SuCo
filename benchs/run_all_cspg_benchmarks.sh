#!/bin/bash
#
# run_all_cspg_benchmarks.sh
#
# Runs the full CSPG benchmark suite and generates paper-style plots.
# Mirrors the structure of run_all_shg_benchmarks.sh.
#
# Usage:
#   chmod +x benchs/run_all_cspg_benchmarks.sh
#   ./benchs/run_all_cspg_benchmarks.sh
#
#   # Single dataset:
#   ./benchs/run_all_cspg_benchmarks.sh gist1m
#
#   # Multiple datasets, only specific benchmarks:
#   BENCHMARKS="construction recall_k10 recall_k20" \
#     ./benchs/run_all_cspg_benchmarks.sh sift1m deep1m
#
# Environment variables:
#   DATA_DIR    — path to dataset root   (default: /Users/dhm/Documents/data)
#   INDEX_DIR   — path for saved indices  (default: /Users/dhm/Documents/indices)
#   OUTPUT_DIR  — path for result JSON    (default: benchs/results_cspg)
#   BENCHMARKS  — space-separated list    (default: all)
#   CSPG_M      — num_partitions          (default: 2)
#   CSPG_LAMBDA — routing ratio           (default: 0.5)
#   CSPG_M_ARG  — M (HNSW degree)         (default: 32)
#   CSPG_EFC    — efConstruction          (default: 128)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-/Users/dhm/Documents/data}"
INDEX_DIR="${INDEX_DIR:-/Users/dhm/Documents/indices}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results_cspg}"
BENCHMARKS="${BENCHMARKS:-all}"
CSPG_M="${CSPG_M:-2}"
CSPG_LAMBDA="${CSPG_LAMBDA:-0.5}"
CSPG_M_ARG="${CSPG_M_ARG:-32}"
CSPG_EFC="${CSPG_EFC:-128}"

BENCH_SCRIPT="${SCRIPT_DIR}/bench_cspg_paper.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_cspg_paper.py"

ALL_DATASETS=(sift1m deep1m gist1m sift10m)

if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

echo "============================================================"
echo "CSPG Paper Benchmark Suite"
echo "============================================================"
echo "  Data dir:    ${DATA_DIR}"
echo "  Index dir:   ${INDEX_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Datasets:    ${DATASETS[*]}"
echo "  Benchmarks:  ${BENCHMARKS}"
echo "  CSPG params: M=${CSPG_M_ARG}, efC=${CSPG_EFC}, m=${CSPG_M}, λ=${CSPG_LAMBDA}"
echo "============================================================"

mkdir -p "${INDEX_DIR}" "${OUTPUT_DIR}"

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# Dataset: ${ds}"
    echo "############################################################"

    LOG_FILE="${OUTPUT_DIR}/log_cspg_${ds}.txt"

    # shellcheck disable=SC2086
    python3 "${BENCH_SCRIPT}" \
        --data-dir   "${DATA_DIR}" \
        --index-dir  "${INDEX_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset    "${ds}" \
        --benchmark  ${BENCHMARKS} \
        --M          "${CSPG_M_ARG}" \
        --efc        "${CSPG_EFC}" \
        --m          "${CSPG_M}" \
        --lam        "${CSPG_LAMBDA}" \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "Done: ${ds}  (log: ${LOG_FILE})"
done

echo ""
echo "############################################################"
echo "# Generating plots"
echo "############################################################"

python3 "${PLOT_SCRIPT}" \
    --results-dir "${OUTPUT_DIR}" \
    --output-dir  "${OUTPUT_DIR}/plots"

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "  Results: ${OUTPUT_DIR}/results_cspg_*.json"
echo "  Plots:   ${OUTPUT_DIR}/plots/"
echo "  Logs:    ${OUTPUT_DIR}/log_cspg_*.txt"
echo "============================================================"