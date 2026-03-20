#!/bin/bash
#
# run_all_shg_benchmarks.sh
#
# Runs the full SHG paper benchmark suite across all datasets and
# generates paper-style plots.
#
# Usage:
#   chmod +x benchs/run_all_shg_benchmarks.sh
#   ./benchs/run_all_shg_benchmarks.sh
#
# Or run individual datasets:
#   ./benchs/run_all_shg_benchmarks.sh gist1m
#   ./benchs/run_all_shg_benchmarks.sh enron msong
#
# Environment variables:
#   DATA_DIR    - path to datasets    (default: /Users/dhm/Documents/data)
#   INDEX_DIR   - path to save indices (default: /Users/dhm/Documents/indices)
#   OUTPUT_DIR  - path for results     (default: benchs/results)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATA_DIR="${DATA_DIR:-/Users/dhm/Documents/data}"
INDEX_DIR="${INDEX_DIR:-/Users/dhm/Documents/indices}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

BENCH_SCRIPT="${SCRIPT_DIR}/bench_shg_paper.py"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_shg_paper.py"

# All datasets from the paper
ALL_DATASETS=(openai enron gist1m msong uqv msturing10m)

# Use provided datasets or all
if [ $# -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=("${ALL_DATASETS[@]}")
fi

echo "============================================================"
echo "SHG Paper Benchmark Suite"
echo "============================================================"
echo "Data dir:    ${DATA_DIR}"
echo "Index dir:   ${INDEX_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Datasets:    ${DATASETS[*]}"
echo "============================================================"
echo ""

mkdir -p "${INDEX_DIR}" "${OUTPUT_DIR}"

# Run benchmarks for each dataset
for ds in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# Running benchmarks for: ${ds}"
    echo "############################################################"

    LOG_FILE="${OUTPUT_DIR}/log_${ds}.txt"

    python3 "${BENCH_SCRIPT}" \
        --data-dir "${DATA_DIR}" \
        --index-dir "${INDEX_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --dataset "${ds}" \
        --benchmark all \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "Benchmark for ${ds} complete. Log: ${LOG_FILE}"
done

# Generate plots
echo ""
echo "############################################################"
echo "# Generating plots"
echo "############################################################"

python3 "${PLOT_SCRIPT}" \
    --results-dir "${OUTPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}/plots"

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "  Results: ${OUTPUT_DIR}/results_*.json"
echo "  Plots:   ${OUTPUT_DIR}/plots/"
echo "  Logs:    ${OUTPUT_DIR}/log_*.txt"
echo "============================================================"
