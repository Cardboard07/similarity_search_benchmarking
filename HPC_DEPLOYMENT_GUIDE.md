# HPC Deployment & 100M+ Scale Execution Guide

This guide details the complete deployment, hardware sizing, and experiment execution procedures for running the **Synthetic Fingerprint Generation and LSH-Band Bloom Filter Duplicate Detection Engine** at massive scale (6M to 102M+ observations) on High-Performance Computing (HPC) clusters and cloud compute instances.

---

## 1. System Architecture & Sizing for 100 Million Scale

### Scale Specifications (102M Observations)
- **Total Samples ($N$):** $102,000,000$ observations (6.8 million unique synthetic identities $\times$ 15 impressions).
- **Train/Gallery Partition ($2/3$):** $68,000,000$ reference fingerprint signatures.
- **Test/Probe Partition ($1/3$):** $34,000,000$ runtime evaluation queries.
- **Signature Length:** 64 MinHash uint32 hashes per fingerprint observation.
- **LSH Index Configuration:** $b = 16$ bands, $r = 4$ rows per band ($b \times r = 64$).
- **Similarity Metric:** Pure fraction of matching signature positions ($\mathrm{mean}(\mathrm{sig}_A == \mathrm{sig}_B)$).

### In-Memory RAM Breakdown at 68M Gallery Records
All arrays are allocated contiguously with zero duplicate copies during loading:

| Memory Component | Formula / Dimensions | Peak RAM |
| :--- | :--- | :--- |
| **Gallery Signatures Matrix** | $68,000,000 \times 64 \times 4\text{ bytes}$ (uint32) | **17.41 GB** |
| **Gallery Identity IDs** | $68,000,000 \times 4\text{ bytes}$ (int32) | **0.27 GB** |
| **LSH Inverted Sorted Keys** | 16 bands $\times 68,000,000 \times 8\text{ bytes}$ (uint64) | **8.70 GB** |
| **LSH Inverted Sorted Indices** | 16 bands $\times 68,000,000 \times 4\text{ bytes}$ (int32) | **4.35 GB** |
| **Probabilistic LSH-Band Bloom Filter** | $272\text{M unique keys},\; p = 0.001\; (\sim 14.38\text{ bits/key})$ | **1.95 GB** |
| **OS, Python Runtime & Query Buffers** | Active query candidates, working heap | **~14.50 GB** |
| **Total In-Memory Working Set** | | **~47.18 GB** |

### Recommended Hardware Node Specifications
- **Optimal Compute Node:** **128 GB RAM**, **32–64 CPU cores** (e.g., AWS `c6i.16xlarge` / `m6i.16xlarge`, GCP `c2-standard-60`, or dual-socket AMD EPYC / Intel Xeon HPC node).
- **Minimum Viable Node:** **64 GB RAM**, **16 CPU cores**.
- **Storage:** **~180 GB NVMe SSD** (fast parallel I/O for 680 HDF5 shards).

---

## 2. Environment Setup on HPC Node

Once you clone or pull the repository onto the HPC node:

```bash
# 1. Clone repository
git clone <REPO_URL> duplicate_detection
cd duplicate_detection

# 2. Create and activate a clean Python 3.11+ environment
conda create -n dup_det python=3.11 -y
conda activate dup_det

# 3. Install dependencies and install package in editable mode
pip install -r requirements.txt
pip install -e .

# 4. Verify system stability with unit tests
pytest tests/test_lsh_scale.py -v
```

All 6 core scale unit tests should pass in under 1 second.

---

## 3. The 3 Canonical Evaluation Experiments

The three recurring evaluation experiments have been standardized into standalone, numbered executable scripts under `scripts/`. Each script can be run independently with clear CLI arguments and exports machine-readable JSON reports directly to `reports/`.

### Experiment 1: LSH Band Bloom Filter Parameter Sweep
**Script:** `scripts/run_exp1_band_sweep.py`  
**Purpose:** Sweeps 9 canonical $(b,r)$ configurations over MinHash length 64 ($b \times r \le 64$) to evaluate the trade-off between search space pruning, query latency, Hit@10 %, and false negatives.

```bash
# Linux / HPC Cluster:
./run_hpc.sh exp1

# Direct Python CLI:
python scripts/run_exp1_band_sweep.py --num-queries 2500 --output reports/band_prefilter_sweep.json
```
- **Output Artifact:** `reports/band_prefilter_sweep.json`
- **Console Output:** Formatted 9-row comparison table displaying $(b,r)$, theoretical threshold $\tau = (1/b)^{1/r}$, bloom pruning ratio %, false negatives (FN), Hit@10 %, and latency per query.

---

### Experiment 2: Full-Scale Standard Retrieval Benchmark
**Script:** `scripts/run_exp2_retrieval_benchmark.py`  
**Purpose:** Computes comprehensive IR and duplicate retrieval metrics under standard operating parameters ($b=16, r=4, \tau=0.40$), including quality tier failure analysis.

```bash
# Linux / HPC Cluster:
./run_hpc.sh exp2

# Direct Python CLI:
python scripts/run_exp2_retrieval_benchmark.py --num-queries 2500 --output reports/benchmark_full_scale.json
```
- **Output Artifact:** `reports/benchmark_full_scale.json`
- **Metrics Evaluated:**
  - **Ranking Metrics:** MRR (Mean Reciprocal Rank), MAP@10 (Mean Average Precision).
  - **Recall Curve:** Hit@1, Hit@5, Hit@10, Hit@20, Hit@50 rate.
  - **Standard IR:** Precision@10, Recall@10.
  - **Efficiency:** Average candidates evaluated, search space pruning ratio %, LSH candidate recall, Bloom filter FN.
  - **Quality Breakdown:** Failure analysis across all 5 noise tiers (`q1` lowest, `q3` pristine, `q5` noisy, `q7` good, `q9` severe noise).
  - **Latency Distribution:** Mean, P50, P90, P99 query latency and QPS throughput.

---

### Experiment 3: 4-Way Architectural Latency & Throughput Benchmark
**Script:** `scripts/run_exp3_latency_4way.py`  
**Purpose:** Directly benchmarks the 4 system architectures across **Positive** (in database), **Negative** (unseen non-duplicates), and **Mixed** (50/50) workloads.

```bash
# Linux / HPC Cluster:
./run_hpc.sh exp3

# Direct Python CLI:
python scripts/run_exp3_latency_4way.py --num-queries 100 --output reports/latency_4way_results.json
```
- **Architectures Tested:**
  1. `1_only_minhash`: Brute force linear scan (MinHash distance over entire gallery).
  2. `2_lsh_and_prefilter`: Full hybrid pipeline (LSH-band Bloom filter pre-filter + LSH candidate lookup + MinHash scoring).
  3. `3_prefilter_no_lsh`: Bloom filter only (early exit on non-matches, linear scan on matches).
  4. `4_lsh_no_prefilter`: LSH candidate retrieval without Bloom filter early-rejection gating.
- **Output Artifact:** `reports/latency_4way_results.json`
- **Console Output:** Comparative throughput (QPS), speedup multipliers vs. brute force (up to 14×+ speedup on negative queries), and latency percentiles.

---

## 4. Master HPC Pipeline Orchestrator

The orchestrator `scripts/run_hpc_pipeline.py` provides a single command-line interface to coordinate generation, signature extraction, and all 3 benchmark experiments.

### Running All 3 Experiments Sequentially
```bash
./run_hpc.sh all_experiments
```
Or with direct Python:
```bash
python scripts/run_hpc_pipeline.py --stage all_experiments --num-queries 2500
```

### Running Quick Smoke Test on Small Gallery Subset
To verify an HPC node setup in seconds before launching full runs:
```bash
python scripts/run_hpc_pipeline.py --stage all_experiments --max-gallery 50000 --num-queries 50
```

### Full End-to-End Scaling Run (Generation -> Extraction -> Experiments)
To generate 102 million samples across 64 cores and run all experiments:
```bash
python scripts/run_hpc_pipeline.py \
    --stage all \
    --num-samples 102000000 \
    --workers 64 \
    --shard-size 150000 \
    --num-queries 2500
```

---

## 5. SLURM Cluster Job Submission

For clusters managed by the SLURM workload manager, a pre-configured template `slurm_job.sh` is provided in the repository root.

### Submitting the Job
```bash
sbatch slurm_job.sh
```

### Monitoring Execution
```bash
# Check queue status
squeue -u $USER

# Tail job stdout log in real-time
tail -f logs/slurm_*.out

# Inspect error log if needed
cat logs/slurm_*.err
```

### Customizing SLURM Directives
Edit `slurm_job.sh` to match your cluster's partition and allocation:
```bash
#SBATCH --cpus-per-task=64     # Request 64 CPU cores
#SBATCH --mem=128G             # Request 128 GB RAM
#SBATCH --time=12:00:00        # Maximum walltime
#SBATCH --partition=standard   # Cluster partition name
```

---

## 6. Architecture & Optimization Safeguards

1. **Multi-Threaded Parallel Band Sorting:**  
   `ScaledLSHIndex.index_signatures` distributes the 16 independent band `np.argsort` sorting operations across CPU cores using Python's `ThreadPoolExecutor`. Because NumPy argsort executes in optimized C without holding the GIL, 68M records index in under 2 minutes on a 64-core node.

2. **Zero-Copy Contiguous HDF5 Reading:**  
   `ShardedSignatureDataset.load_gallery` and `load_probes` calculate total records in advance, pre-allocate single `np.empty((N, 64), dtype=np.uint32)` destination arrays, and stream shard slices directly into memory. This eliminates intermediate Python lists and prevents the 17.4 GB duplicate memory spike during array concatenation.

3. **Numeric Shard File Sorting:**  
   All shard readers extract digits numerically (`key=lambda p: int(''.join(filter(str.isdigit, p.stem)))`), completely preventing lexicographical bugs where `dataset_1000.h5` would precede `dataset_20.h5`.

4. **Runtime MinHash Signature Computation:**  
   All probe queries are loaded as raw minutiae tuples `(points, angles, types)` and processed through `VectorizedMinHashEngine.process_observation` at query time, simulating realistic biometric deployment where probe templates are never pre-computed.

5. **Pure MinHash Similarity:**  
   Candidate ranking strictly computes the fraction of matching hash positions (`sim = mean(sig_a == sig_b)`), which is the mathematically unbiased estimator of Jaccard similarity.
