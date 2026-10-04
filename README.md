# Biometric Fingerprint Duplicate Detection System
### Large-Scale Biometric Retrieval via Bloom Filter Gating, MinHash Signatures, and Locality-Sensitive Hashing (LSH)

A high-performance biometric fingerprint duplicate detection and 1:N retrieval system. Given an arbitrary query fingerprint observation (minutiae point set), the system rapidly identifies all related observations belonging to the same identity across millions of gallery records, while short-circuiting non-existent queries in microseconds.

This document presents the complete system architecture, mathematical derivations, and empirical results across two benchmarks:
1. **Small-Scale Experiment**: 15,000 observations across 1,000 identities (JSON records).
2. **Large-Scale Experiment**: **6,000,000 observations across 400,000 identities** (40 HDF5 shards).

The large-scale benchmark consists of three phases:
- **Phase 1**: LSH hyperparameter tuning sweep on 20k identities (200k gallery).
- **Phase 2**: Full 6M standard metrics evaluation with optimal hyperparameters.
- **Phase 3**: 4-way architectural latency and accuracy comparison at full 4M gallery scale.

---

## 1. System Architecture & General Working

The end-to-end retrieval pipeline decouples biometric duplicate search into four hierarchical stages designed for sub-linear search complexity:

```
                          [ Query Fingerprint Minutiae ]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │     Bloom Filter Gating     │
                         │    (400,000 Identities)     │
                         └──────────────┬──────────────┘
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
             [ Definite Negative ]               [ Probable Positive ]
             Abort Query (< 14 µs)                         │
             Zero Search Cost                              ▼
                                             ┌───────────────────────────┐
                                             │  Vectorized MinHash Engine│
                                             │  (64 Integer Hash Signatures)
                                             └─────────────┬─────────────┘
                                                           │
                                                           ▼
                                             ┌───────────────────────────┐
                                             │   LSH Sorted-Array Index  │
                                             │   (16 Bands x 4 Rows)     │
                                             │  Prunes 99.9991% of 4M    │
                                             └─────────────┬─────────────┘
                                                           │
                                                           ▼
                                             ┌───────────────────────────┐
                                             │  Candidate Jaccard Rank   │
                                             │  Top-K Duplicate Matches  │
                                             └───────────────────────────┘
```

### Stage 1: Topological Minutiae Relative Tokenizer
Raw minutiae point sets $M = \{(x_i, y_i, \theta_i, \text{type}_i)\}_{i=1}^N$ vary under rigid body translation and rotation. To achieve geometric invariance:
- For each minutia $i$, the $k$-nearest spatial neighbors $j \in \mathcal{N}_k(i)$ are determined.
- Local topology is encoded as a discrete 5-tuple:
  $$\tau_{i,j} = \left(\left\lfloor \frac{d_{i,j}}{\Delta d} \right\rfloor, \left\lfloor \frac{\phi_{i,j}^{\text{rel}}}{\Delta \phi} \right\rfloor, \left\lfloor \frac{\Delta \theta_{i,j}}{\Delta \theta} \right\rfloor, \text{type}_i, \text{type}_j\right)$$
  where $d_{i,j} = \|(x_j - x_i, y_j - y_i)\|_2$, $\phi_{i,j}^{\text{rel}} = (\text{atan2}(y_j - y_i, x_j - x_i) - \theta_i) \pmod{360^\circ}$, and $\Delta \theta_{i,j} = (\theta_j - \theta_i) \pmod{360^\circ}$.
- This relative geometry representation is strictly invariant to global displacement and global orientation.

### Stage 2: MinHash Signature Derivation
Each token $\tau$ is mapped to a 64-bit integer $x = \text{Blake2b}(\tau)$. To compress variable-sized token sets into fixed-length signatures, a family of $D = 64$ linear congruential hash functions is applied:
$$h_d(x) = (a_d \cdot x + b_d) \pmod p, \quad d \in \{1, \dots, D\}$$
where $p = 4{,}294{,}967{,}311$ (prime modulus $2^{32} - 5$), and $a_d, b_d$ are random coefficients. The MinHash signature is:
$$s(M)_d = \min_{\tau \in \text{Tokens}(M)} h_d(\text{Blake2b}(\tau))$$
By the MinHash Theorem, the probability that two fingerprints share an identical hash value equals their Jaccard similarity:
$$P(s(M_A)_d = s(M_B)_d) = J(\text{Tokens}(M_A), \text{Tokens}(M_B))$$

To prevent 64-bit scalar multiplication overflow in vectorized execution, modular reduction is pre-applied:
$$(a_d \cdot x + b_d) \pmod p = \left(a_d \cdot (x \pmod p) + b_d\right) \pmod p$$
guaranteeing exact mathematical equivalence while operating at **0.26 ms per observation** (**30x faster** than iterative hashing).

### Stage 3: Probabilistic Bloom Filter Gating
Before querying the index or evaluating candidates, the query identity is tested against a pre-search **Bloom Filter**:
- Utilizes Kirsch-Mitzenmacher double-hashing optimization to evaluate $k=10$ hash indices:
  $$g_i(x) = \left(h_1(x) + i \cdot h_2(x)\right) \pmod m, \quad i \in \{0, \dots, k-1\}$$
- If any bit is 0, the identity is **definitively not in the database** (zero false negatives). The query aborts immediately in **$< 14\,\mu\text{s}$**, saving 100% of search compute.
- If all bits are 1, the query proceeds to candidate retrieval with a false positive rate $\le 0.1\%$.

### Stage 4: Locality-Sensitive Hashing (LSH) — Sorted-Array Index
The 64-integer signature is partitioned into $b$ bands of $r$ rows ($b \times r = 64$).
- Each band is hashed using a vectorized FNV-1a-style 64-bit hash into integer keys stored in a **sorted numpy array** (replacing Python dicts for 3.5x lower RAM and 9.4x faster indexing).
- Two fingerprints become candidates if their sub-signatures match in **at least one band**.
- The probability of candidate collision as a function of Jaccard similarity $s$ follows an S-curve:
  $$P(\text{Candidate Collision}) = 1 - (1 - s^r)^b$$
- Approximation threshold: $t \approx (1/b)^{1/r}$.

### Stage 5: Candidate Verification & Jaccard Scoring
For all unique candidates retrieved across matching band buckets, Jaccard similarity is evaluated against the query signature, producing ranked duplicate candidates.

---

## 2. Experimental Datasets

| Dimension | Small-Scale Baseline | Large-Scale (6M) |
| :--- | :--- | :--- |
| **Storage Medium** | Individual JSON files | 40 HDF5 shards (`signatures_000.h5` - `signatures_039.h5`) |
| **Total Observations** | 15,000 observations | **6,000,000 observations** |
| **Total Unique Identities** | 1,000 identities | **400,000 identities** (4 Lakh) |
| **Observations per Identity** | 15 (3 orientations × 5 qualities) | 15 (3 orientations × 5 qualities) |
| **Gallery Size (Train)** | 10,000 records | **4,000,000 records** (10 obs / identity) |
| **Probe Size (Test)** | 4,500 records | **2,000,000 records** (5 obs / identity) |
| **Signature Storage** | `signatures.json` (15.5 MB) | 40 HDF5 shards (1.10 GB, LZF compressed) |
| **Bloom Filter Capacity** | $n = 1,000$ identities (1.76 KB RAM) | **$n = 400,000$ identities (702 KB RAM)** |
| **LSH Configuration** | $b = 8$, $r = 8$ ($t \approx 0.77$) | **$b = 16$, $r = 4$ ($t = 0.50$)** |

---

## 3. Phase 1 — LSH Hyperparameter Tuning (20k Identities, 200k Gallery)

A sweep across 9 `(bands, rows)` configurations was run using 500 balanced probes on 20,000 identities (200,000 gallery records). The goal was to find the minimum candidate load that guarantees 100% recall before scaling to 6M.

| Bands (b) | Rows (r) | Threshold (t) | Avg Candidates | Hit@10 | Candidate Precision | Latency (ms) |
|:---------:|:--------:|:-------------:|:--------------:|:------:|:-------------------:|:------------:|
| 4         | 16       | 0.917         | 0.8            | 31.6%  | 31.60%              | 0.09         |
| 6         | 10       | 0.836         | 2.3            | 61.6%  | 61.60%              | 0.16         |
| 8         | 8        | 0.771         | 3.8            | 81.2%  | 81.20%              | 0.23         |
| 10        | 6        | 0.681         | 6.1            | 94.0%  | 94.00%              | 0.20         |
| 12        | 5        | 0.608         | 7.7            | 98.6%  | 98.47%              | 0.24         |
| **16**    | **4**    | **0.500**     | **10.4**       | **100.0%** | **93.73%**     | **0.28**     |
| 20        | 3        | 0.368         | 57.1           | 100.0% | 34.03%              | 0.47         |
| 32        | 2        | 0.177         | 2,593.6        | 100.0% | 0.62%               | 3.23         |
| 64        | 1        | 0.016         | 113,293.9      | 100.0% | 0.009%              | 172.64       |

### Optimal Hyperparameters Selected

| Parameter | Value | Rationale |
|:----------|:-----:|:----------|
| **Bands (b\*)** | **16** | First configuration achieving 100% recall |
| **Rows (r\*)** | **4** | b × r = 64 (uses full 64-dim signature) |
| **LSH Threshold (t)** | 0.500 | $t = (1/16)^{1/4} = 0.500$ |
| **Decision Threshold (τ\*)** | 0.40 | Empirically optimal for Jaccard scoring |
| **Avg Candidates at 200k Gallery** | 10.4 | Prunes 99.995% of search space |

The precision-vs-candidates curve shows a sharp knee at b=16/r=4: above this point, precision collapses from 93% toward noise-level values while candidates explode by orders of magnitude.

---

## 4. Phase 2 — Full 6M Standard Metrics (b=16, r=4, τ=0.40)

Evaluated **2,500 balanced probes** (500 per quality level: q1, q3, q5, q7, q9) against the full **4,000,000-record gallery**.

### LSH Index Build Performance

| Metric | Value |
|:-------|:------|
| Gallery Size | 4,000,000 records |
| Indexing Time | **25 seconds** (vs 234s with dict-based index) |
| Indexing Throughput | 162,695 records/sec |
| Peak RAM | ~2.3 GB (vs ~8 GB with dict-based index) |
| Implementation | Vectorized FNV-1a hashing + `np.argsort` + `np.searchsorted` |

### Bloom Filter Performance

| Metric | Value |
|:-------|:------|
| Queries Screened | 2,500 |
| False Negatives | **0** |
| False Negative Rate | **0.00%** |
| Avg Bloom Latency | 0.038 ms |

### Headline Retrieval Numbers

| Metric | Value |
|:-------|:------|
| **Mean Reciprocal Rank (MRR)** | **0.9988** |
| **MAP@10** | **0.9181** |
| **Hit@10** | **99.92%** |
| **LSH Candidate Recall** | **99.92%** |
| **Total False Negatives** | **2** out of 2,500 (0.08%) |
| **Bloom False Negatives** | 0 |
| **LSH False Negatives** | 2 |
| **Avg Candidates per Query** | 34.57 |
| **Search Space Pruned** | **99.9991%** (4,000,000 → 34.6 candidates) |
| **Throughput** | **1,033 QPS** |

### Precision@K, Recall@K, Hit@K

| K | Hit@K | Precision@K | Recall@K |
|:-:|:-----:|:-----------:|:--------:|
| 1 | 99.84% | 0.9984 | 0.0998 |
| 5 | 99.92% | 0.9912 | 0.4956 |
| **10** | **99.92%** | **0.9180** | **0.9180** |
| 20 | 99.92% | 0.4595 | 0.9189 |
| 50 | 99.92% | 0.1838 | 0.9189 |

> **Note on P@10 = R@10 = 0.918**: Each identity has exactly 10 gallery observations (2 orientations × 5 quality levels). At K=10, the system retrieves ~9.18 of 10 true matches on average — thus precision and recall are numerically equal. Recall plateaus at ~0.919 beyond K=10 as the candidate pool is exhausted.

### Decision Threshold Analysis (τ = 0.40)

| Metric | Value |
|:-------|:------|
| True Positives | 22,920 |
| False Positives | 48 |
| Hit Rate at Threshold | 99.92% |
| Avg Recall at Threshold | 0.9168 |
| **Avg Precision at Threshold** | **0.9990** |

### Quality-Level Breakdown (Zero-Recall Analysis)

| Quality | Probes | Zero-Recall Failures | Failure Rate | Hit@10 | P@10 | R@10 | MRR | MAP@10 | Avg Candidates |
|:-------:|:------:|:--------------------:|:------------:|:------:|:----:|:----:|:---:|:------:|:--------------:|
| q1 (Lowest) | 500 | 0 | 0.0% | 100.00% | 0.8980 | 0.8980 | 1.0000 | 0.8982 | 32.2 |
| q3 | 500 | 0 | 0.0% | 100.00% | 0.9810 | 0.9810 | 1.0000 | 0.9812 | 34.8 |
| q5 | 500 | 1 | 0.2% | 99.80% | 0.9030 | 0.9030 | 0.9970 | 0.9029 | 33.1 |
| q7 | 500 | 0 | 0.0% | 100.00% | 0.9176 | 0.9176 | 1.0000 | 0.9178 | 41.2 |
| q9 (Highest) | 500 | 1 | 0.2% | 99.80% | 0.8904 | 0.8904 | 0.9970 | 0.8904 | 31.6 |
| **Overall** | **2,500** | **2** | **0.08%** | **99.92%** | **0.9180** | **0.9180** | **0.9988** | **0.9181** | **34.6** |

> **Observation**: q3 achieves the highest precision (0.981) as medium-quality fingerprints produce the most stable minutiae distributions. q9 and q1 have marginally lower precision due to extreme quality parameter ranges introducing hash variability, but still maintain 99.8%+ hit rates with 0 complete failures at these levels.

### Latency Profile (2,500 Queries, 4M Gallery)

| Metric | Value |
|:-------|:------|
| **Avg Latency** | **0.884 ms** |
| P50 Latency | 0.725 ms |
| P90 Latency | 1.470 ms |
| **P99 Latency** | **2.830 ms** |
| Min Latency | 0.348 ms |
| Max Latency | 5.003 ms |
| **Throughput** | **1,033 QPS** |

**Per-query latency breakdown:**

| Stage | Time (ms) | % of Query Time |
|:------|:---------:|:---------------:|
| Bloom Filter Gating | 0.038 | 4.3% |
| **LSH Candidate Search** | **0.653** | **73.8%** |
| Jaccard Scoring & Ranking | 0.176 | 19.9% |

---

## 5. Phase 3 — 4-Way Architectural Comparison (b=16, r=4, 4M Gallery)

100 queries per workload tested across three query types:
- **Positive Queries**: Identity is in the database (duplicates exist).
- **Negative Queries**: Identity is not in the database (no duplicates).
- **Mixed Workload**: 50% positive / 50% negative (production simulation).

### Latency Comparison

| Configuration | Positive (In DB) | Negative (Not in DB) | Mixed (50/50) | Mixed Throughput |
|:--------------|:----------------:|:--------------------:|:-------------:|:----------------:|
| 1. Only MinHash (Brute Force) | 1,084.25 ms | 997.53 ms | 1,175.89 ms | 0.9 QPS |
| **2. LSH + Bloom Filter** | **0.63 ms** | **0.008 ms** | **0.30 ms** | **3,333 QPS** |
| 3. Bloom Filter (No LSH) | 925.52 ms | 0.009 ms | 524.33 ms | 1.9 QPS |
| 4. LSH (No Bloom Filter) | 0.57 ms | 0.384 ms | 0.59 ms | 1,695 QPS |

### Speedup vs Brute Force

| Configuration | Positive Speedup | Negative Speedup | Mixed Speedup |
|:--------------|:----------------:|:----------------:|:-------------:|
| **LSH + Bloom Filter** | **1,722×** | **124,815×** | **3,922×** |
| Bloom Filter (No LSH) | 1.2× | 110,837× | 2.2× |
| LSH (No Bloom Filter) | 1,897× | 2,598× | 1,993× |

```
  Positive Queries (In DB):
  Brute Force:         ████████████████████████████████████████ 1,084 ms
  LSH + Bloom Filter:  ▏ 0.63 ms  [1,722x Speedup]

  Negative Queries (Not in DB):
  Brute Force:         ████████████████████████████████████████ 997 ms
  LSH + Bloom Filter:  ▏ 0.008 ms [124,815x Speedup]

  Mixed Workload (50/50):
  Brute Force:         ████████████████████████████████████████ 1,175 ms
  LSH + Bloom Filter:  ▏ 0.30 ms  [3,922x Speedup]
```

### Retrieval Metrics (Positive Workload, 100 Queries)

| Configuration | Hit@10 | Precision@10 | Recall@10 | False Negatives | Avg Candidates |
|:--------------|:------:|:------------:|:---------:|:---------------:|:--------------:|
| 1. Only MinHash (Brute Force) | 100.00% | 1.0000 | 1.0000 | 0 | 4,000,000 |
| **2. LSH + Bloom Filter** | **99.00%** | **0.9070** | **0.9070** | **1** | **19.3** |
| 3. Bloom Filter (No LSH) | 100.00% | 1.0000 | 1.0000 | 0 | 4,000,000 |
| 4. LSH (No Bloom Filter) | 99.00% | 0.9070 | 0.9070 | 1 | 19.3 |

**Key Architectural Insights:**

1. **LSH is essential for positive queries**: Reduces search from 1,084 ms → 0.63 ms (1,722× speedup) while inspecting only **19.3 candidates** out of 4,000,000 (99.9995% search space reduction). The 1% miss rate (1/100) is a statistical edge case confirming the probabilistic guarantee of LSH.

2. **Bloom filter is essential for negative queries**: Rejects non-existent fingerprints in **0.008 ms** (8 microseconds) without touching the LSH index at all — a **124,815×** speedup over brute force. Without Bloom, LSH alone still requires 0.384 ms per negative query (hashing and bucket lookups even if empty).

3. **Combined pipeline dominates all scenarios**: In mixed workloads (realistic production), LSH+Bloom achieves 0.30 ms average — 3,922× faster than brute force and 1.97× faster than LSH alone, because Bloom eliminates ~50% of queries before any MinHash or LSH computation.

---

## 6. LSH Index Engineering — Sorted-Array Implementation

The `ScaledLSHIndex` was redesigned from a Python dict-based inverted index to a vectorized **sorted numpy array** implementation:

| Aspect | Dict-Based (Old) | Sorted-Array (New) | Improvement |
|:-------|:----------------:|:------------------:|:-----------:|
| Indexing Time (4M records) | 234 seconds | **25 seconds** | **9.4× faster** |
| Peak RAM | ~8 GB | **~2.3 GB** | **3.5× less** |
| Lookup Algorithm | Python dict O(1) amortized | `np.searchsorted` binary search | Vectorized, cache-friendly |
| Key Type | Python tuple/bytes | `int64` (FNV-1a-style hash) | Zero allocation per query |

The key technique: each band's `(b_0, b_1, ..., b_{r-1})` row values are combined into a single `uint64` key via XOR-shift multiplication hashing, enabling the entire inverted index to be stored as two flat `int64` numpy arrays (`sorted_keys`, `sorted_indices`) and queried with `np.searchsorted`.

---

## 7. Mathematical Collision Probability Reference

$$P(\text{Collision at similarity } s) = 1 - (1 - s^r)^b$$

| Configuration | b | r | Threshold t | P(collision) at s=0.20 | P(collision) at s=0.30 | P(collision) at s=0.50 |
|:---|:-:|:-:|:-----------:|:-----------------------:|:-----------------------:|:-----------------------:|
| Small-Scale | 8 | 8 | 0.771 | 0.002% | 0.052% | 0.4% |
| **Optimal (b=16, r=4)** | **16** | **4** | **0.500** | **2.53%** | **12.21%** | **50.0%** |
| Aggressive | 20 | 3 | 0.368 | 15.0% | 40.9% | 87.5% |
| Over-sensitive | 32 | 2 | 0.177 | 73.0% | 95.3% | 99.9% |

The b=16, r=4 configuration sits at the S-curve knee: sufficient recall for intra-identity Jaccard scores (~0.40–0.60) while rejecting uncorrelated fingerprints (s~0.10–0.20).

---

## 8. Throughput & Scalability Benchmark

| Operation | Scale | Total Time | Throughput | Peak RAM |
| :--- | :--- | :---: | :---: | :---: |
| **MinHash Extraction (40 Shards)** | 6,000,000 observations | 439.9 s (7.3 min) | 13,638 obs/sec | ~1.5 GB |
| **400k Identity Bloom Filter Build** | 400,000 identities | 4.21 s | 95,012 IDs/sec | **702 KB** |
| **LSH Index Build (4M Gallery)** | 4,000,000 records | **25 s** | **162,695 rec/sec** | **~2.3 GB** |
| **Bloom Gating (Negative Query)** | Per query | 0.008 ms | 125,000 QPS | Zero |
| **Full Pipeline (Positive Query)** | Per query (4M gallery) | 0.63 ms | 1,588 QPS | Sliced |
| **Full Pipeline (Mixed Workload)** | Per query (4M gallery) | 0.30 ms | **3,333 QPS** | Sliced |
| **Standard Benchmark (2,500 probes)** | 4M gallery | 2.42 s total | **1,033 QPS** | ~2.3 GB |

---

## 9. Project Structure & Source Code Layout

```
duplicate_detection/
├── dataset/
│   ├── dataset_000.h5 ... dataset_039.h5     # 40 HDF5 shards (6,000,000 raw samples) [EXCLUDED FROM GIT]
│   ├── manifest.json                         # Shard tracking manifest
│   ├── bloom_filter_400k.bin / .json         # 400,000 identity Bloom filter (702 KB) [EXCLUDED FROM GIT]
│   └── signatures_shards/
│       ├── signatures_000.h5 ... _039.h5     # Extracted 6M MinHash signatures [EXCLUDED FROM GIT]
├── src/
│   ├── lsh_scale/
│   │   ├── __init__.py                       # Package exports
│   │   ├── vectorized_minhash.py             # Vectorized NumPy token extraction & modular MinHash
│   │   ├── bloom_filter_6m.py                # ScaledIdentityBloomFilter & LargeBitArray
│   │   ├── lsh_index_6m.py                   # ScaledLSHIndex — sorted-array inverted index
│   │   └── hdf5_signature_extractor.py       # Sharded signature extractor & reader
│   └── storage/
│       └── hdf5_reader.py                    # HDF5 observation streaming reader
├── scripts/
│   ├── build_signatures_6m.py                # Parallel CLI signature extraction for 40 shards
│   ├── latency_comparison_6m.py              # 4-way latency comparison benchmark script
│   └── evaluate_similarity_6m.py             # Comprehensive retrieval accuracy benchmark
├── reports/
│   ├── latency_4way_results_scale.json       # Raw JSON — 4-way latency comparison (Phase 3)
│   └── benchmark_6m_best_hyperparameters.json# Raw JSON — accuracy & retrieval metrics (Phase 2)
└── tests/
    └── test_lsh_scale.py                     # Unit tests verifying mathematical equivalence
```

---

## 10. CLI Usage Guide & Reproducibility

### 1. Run Automated Unit Tests
Verifies mathematical equivalence of vectorized tokenization and modular MinHash against the baseline:
```powershell
python -m pytest tests/test_lsh_scale.py -v
```

### 2. Extract Signatures Across 40 Shards (6M Observations)
Processes all 40 HDF5 shards into LZF-compressed signature shards and constructs the 400k Bloom filter:
```powershell
python scripts/build_signatures_6m.py --num-workers 10 --batch-size 25000
```

### 3. Run 4-Way Latency Comparison Benchmark (Phase 3)
Executes the 4-way latency evaluation across Positive, Negative, and Mixed workloads:
```powershell
python scripts/latency_comparison_6m.py --max-gallery 4000000 --num-queries 100 --bands 16 --rows 4
```

### 4. Run Retrieval Accuracy & Efficiency Benchmark (Phase 2)
Computes MRR, MAP@10, Hit@K, candidate recall, zero-recall quality breakdown, and latency percentiles:
```powershell
python scripts/evaluate_similarity_6m.py --max-gallery 4000000 --num-queries 2500 --bands 16 --rows 4 --tau 0.4
```

---

## 11. Summary of Results

| Aspect | Small-Scale (10k Gallery) | **Large-Scale (4M Gallery)** |
|:-------|:-------------------------:|:----------------------------:|
| Optimal Configuration | b=8, r=8, t=0.77 | **b=16, r=4, t=0.50, τ=0.40** |
| MRR | 0.790 | **0.9988** |
| MAP@10 | 0.389 | **0.9181** |
| Hit@10 | 79.0% | **99.92%** |
| Total False Negatives | — | **2 / 2,500 (0.08%)** |
| Search Space Reduction | 99.96% | **99.9991%** |
| Avg Candidates | ~4 | **34.6** |
| Avg Query Latency | — | **0.884 ms** |
| P99 Latency | — | **2.830 ms** |
| Throughput | — | **1,033 QPS** |
| Speedup vs Brute Force (Positive) | 10.4× | **1,722×** |
| Speedup vs Brute Force (Negative) | 53,300× | **124,815×** |
| LSH Index RAM | — | **~2.3 GB** |
| LSH Index Build Time | — | **25 seconds** |
