#!/usr/bin/env bash
# ==============================================================================
# HPC Duplicate Detection Runner Script (Linux / POSIX)
# ==============================================================================
# Usage:
#   ./run_hpc.sh exp1                              # Run Experiment 1 (Band Sweep)
#   ./run_hpc.sh exp2                              # Run Experiment 2 (Retrieval Benchmark)
#   ./run_hpc.sh exp3                              # Run Experiment 3 (4-Way Latency)
#   ./run_hpc.sh all_experiments                   # Run Exp 1, 2, and 3 sequentially
#   ./run_hpc.sh all --num-samples 105000000       # Full 105M generation + all experiments
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

STAGE="${1:-all_experiments}"
shift || true

# Determine Python interpreter
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "[ERROR] Python 3 was not found in PATH. Please load your Python/Conda module."
    exit 1
fi

# Detect CPU cores
if command -v nproc &>/dev/null; then
    CORES=$(nproc)
else
    CORES=8
fi

echo "================================================================================"
echo "           HPC DUPLICATE DETECTION LAUNCHER (Scale: 6M - 105M+)"
echo "================================================================================"
echo " Working Directory: ${SCRIPT_DIR}"
echo " Python Binary:     $(which ${PYTHON})"
echo " Python Version:    $(${PYTHON} --version)"
echo " Detected Cores:    ${CORES}"
echo " Pipeline Stage:    ${STAGE}"
echo " Additional Flags:  $*"
echo "================================================================================"

# Execute requested stage via master orchestrator
exec ${PYTHON} scripts/run_hpc_pipeline.py --stage "${STAGE}" --workers "${CORES}" "$@"
