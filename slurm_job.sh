#!/usr/bin/env bash
#SBATCH --job-name=dup_det_105m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ==============================================================================
# SLURM Batch Submission Script for 105M+ Scale Duplicate Detection
# ==============================================================================
# Submit to queue with:
#   sbatch slurm_job.sh
# Check status with:
#   squeue -u $USER
# View real-time logs with:
#   tail -f logs/slurm_<JOB_ID>.out
# ==============================================================================

set -euo pipefail

# Ensure logs directory exists
mkdir -p logs reports

echo "================================================================================"
echo " Starting SLURM Job ID:    ${SLURM_JOB_ID:-LOCAL}"
echo " Compute Node:             $(hostname)"
echo " Allocation:               ${SLURM_CPUS_PER_TASK:-64} CPUs, 128 GB RAM"
echo " Submission Directory:     ${SLURM_SUBMIT_DIR:-$(pwd)}"
echo " Start Timestamp:          $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "================================================================================"

# 1. Environment Loading (customize according to your HPC cluster modules)
# Example:
# module purge
# module load python/3.11-anaconda
# source activate dup_det_env

# 2. OpenBLAS / MKL Threading Configuration
# Prevent thread over-subscription during parallel multiprocessing & sorting
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# 3. Verify Python & Dependencies
python -c "import numpy, scipy, h5py, tqdm; print('All core libraries verified!')"

# 4. Pipeline Execution Options:
#
# Option A: Run all three benchmark experiments on existing dataset
python scripts/run_hpc_pipeline.py \
    --stage all_experiments \
    --workers ${SLURM_CPUS_PER_TASK:-64} \
    --num-queries 2500 \
    --reports-dir reports

# Option B: Run full 105M generation (7M identities), signature extraction, and experiments (Uncomment if generating 105M)
# python scripts/run_hpc_pipeline.py \
#     --stage all \
#     --num-samples 105000000 \
#     --workers ${SLURM_CPUS_PER_TASK:-64} \
#     --num-queries 2500 \
#     --reports-dir reports

echo "================================================================================"
echo " Job Complete! Results saved to reports/"
echo " End Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "================================================================================"
