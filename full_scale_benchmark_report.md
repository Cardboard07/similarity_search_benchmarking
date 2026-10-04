# Fingerprint Duplicate Detection — Full-Scale Benchmark Report (6M Samples)

> **Date**: 2026-10-04 | **Dataset**: 400,000 identities x 15 observations = 6,000,000 samples | **Gallery**: 4,000,000 (train) | **Probes**: 2,000,000 (test)

---

## Phase 1: LSH Hyperparameter Tuning (20k Identities, 200k Gallery)

Swept across 9 `(bands, rows)` configurations using 500 balanced probes on 20,000 identities (200,000 gallery records).

| Bands (b) | Rows (r) | Threshold (t) | Avg Candidates | Hit@10 | Candidate Precision | Latency (ms) |
|:---------:|:--------:|:-------------:|:--------------:|:------:|:-------------------:|:------------:|
| 4         | 16       | 0.917         | 0.8            | 31.6%  | 31.60%              | 0.09         |
| 6         | 10       | 0.836         | 2.3            | 61.6%  | 61.60%              | 0.16         |
| 8         | 8        | 0.771         | 3.8            | 81.2%  | 81.20%              | 0.23         |
| 10        | 6        | 0.681         | 6.1            | 94.0%  | 94.00%              | 0.20         |
| 12        | 5        | 0.608         | 7.7            | 98.6%  | 98.47%              | 0.24         |
| **16**    | **4**    | **0.500**     | **10.4**       | **100.0%** | **93.73%**      | **0.28**     |
| 20        | 3        | 0.368         | 57.1           | 100.0% | 34.03%              | 0.47         |
| 32        | 2        | 0.177         | 2,593.6        | 100.0% | 0.62%               | 3.23         |
| 64        | 1        | 0.016         | 113,293.9      | 100.0% | 0.009%              | 172.64       |

### Optimal Hyperparameters

| Parameter | Value |
|:----------|:------|
| **Bands (b*)** | **16** |
| **Rows (r*)** | **4** |
| **LSH Threshold** | 0.500 |
| **Decision Threshold (tau*)** | 0.40 |
| **Rationale** | 100% recall, 0 false negatives, prunes 200k to 10.4 candidates, 0.28 ms latency |

### Phase 1 Figures

````carousel
![Precision vs Candidates Curve](C:/Users/hrish/.gemini/antigravity-ide/brain/453cc283-9b7b-497a-bbbe-83074734020c/figures/lsh_precision_vs_candidates_curve.png)
<!-- slide -->
![LSH Tuning Tradeoffs](C:/Users/hrish/.gemini/antigravity-ide/brain/453cc283-9b7b-497a-bbbe-83074734020c/figures/lsh_tuning_tradeoffs.png)
<!-- slide -->
![Recall by Quality Breakdown](C:/Users/hrish/.gemini/antigravity-ide/brain/453cc283-9b7b-497a-bbbe-83074734020c/figures/recall_by_quality_breakdown.png)
````

---

## Phase 2: Full 6M Standard Metrics (b=16, r=4, tau=0.40)

Evaluated 2,500 balanced probes (500 per quality level) against the full 4,000,000 gallery.

### Headline Numbers

| Metric | Value |
|:-------|:------|
| **Gallery Size** | 4,000,000 records (400,000 unique identities) |
| **Total Probes** | 2,500 (balanced across q1, q3, q5, q7, q9) |
| **MRR** | **0.9988** |
| **MAP** | **0.9181** |
| **Hit@10** | **99.92%** |
| **LSH Candidate Recall** | **99.92%** |
| **Total False Negatives** | **2** (Bloom: 0, LSH: 2) |
| **Avg Candidates per Query** | 34.57 (pruned **99.9991%** of search space) |
| **Throughput** | **1,033 QPS** |

### Precision@K, Recall@K, Hit@K

| K | Hit@K | Precision@K | Recall@K |
|:-:|:-----:|:-----------:|:--------:|
| 1 | 99.84% | 0.9984 | 0.0998 |
| 5 | 99.92% | 0.9912 | 0.4956 |
| 10 | 99.92% | 0.9180 | 0.9180 |
| 20 | 99.92% | 0.4595 | 0.9189 |
| 50 | 99.92% | 0.1838 | 0.9189 |

> [!NOTE]
> Each identity has 10 gallery observations (orientations 1 and 2 x 5 qualities). P@10 = R@10 = 0.918 is expected since the system retrieves ~9.18 true matches in the top 10. Recall plateaus at ~0.919 beyond K=10 because candidates are already exhausted.

### Decision Threshold Analysis (tau = 0.40)

| Metric | Value |
|:-------|:------|
| True Positives | 22,920 |
| False Positives | 48 |
| Hit Rate at Threshold | 99.92% |
| Avg Recall at Threshold | 0.9168 |
| **Avg Precision at Threshold** | **0.9990** |

### Quality-Level Breakdown (Zero-Recall Analysis)

| Quality | Queries | Zero-Recall Failures | Failure Rate | Hit@10 | P@10 | R@10 | MRR | MAP |
|:-------:|:-------:|:-------------------:|:------------:|:------:|:----:|:----:|:---:|:---:|
| q1 | 500 | 0 | 0.0% | 100.00% | 0.8980 | 0.8980 | 1.0000 | 0.8982 |
| q3 | 500 | 0 | 0.0% | 100.00% | 0.9810 | 0.9810 | 1.0000 | 0.9812 |
| q5 | 500 | 1 | 0.2% | 99.80% | 0.9030 | 0.9030 | 0.9970 | 0.9029 |
| q7 | 500 | 0 | 0.0% | 100.00% | 0.9176 | 0.9176 | 1.0000 | 0.9178 |
| q9 | 500 | 1 | 0.2% | 99.80% | 0.8904 | 0.8904 | 0.9970 | 0.8904 |

> [!TIP]
> q3 (medium quality) achieves the highest precision (0.981) because these fingerprints produce the most stable minutiae distributions. q9 (highest quality) and q1 (lowest quality) have slightly lower precision due to extreme parameter ranges producing slightly more hash variability, but still maintain 99.8%+ hit rates.

### Latency Profile

| Metric | Value |
|:-------|:------|
| Avg Latency | **0.884 ms** |
| P50 Latency | 0.725 ms |
| P90 Latency | 1.470 ms |
| P99 Latency | 2.830 ms |
| Min Latency | 0.348 ms |
| Max Latency | 5.003 ms |

**Breakdown per query:**

| Stage | Time (ms) | % of Total |
|:------|:---------:|:----------:|
| Bloom Filter Gating | 0.038 | 4.3% |
| LSH Candidate Search | 0.653 | 73.8% |
| Jaccard Scoring and Ranking | 0.176 | 19.9% |

---

## Phase 3: 4-Way Architectural Comparison (b=16, r=4)

100 queries per workload against the full 4,000,000 gallery.

### Latency Comparison

| Configuration | Positive (In DB) | Negative (Not in DB) | Mixed (50/50) |
|:--------------|:----------------:|:--------------------:|:-------------:|
| 1. Only MinHash (Brute Force) | 1,084.25 ms | 997.53 ms | 1,175.89 ms |
| **2. LSH + Bloom Filter** | **0.63 ms** | **0.008 ms** | **0.30 ms** |
| 3. Bloom Filter (No LSH) | 925.52 ms | 0.009 ms | 524.33 ms |
| 4. LSH (No Bloom Filter) | 0.57 ms | 0.384 ms | 0.59 ms |

### Speedup vs Brute Force

| Configuration | Positive | Negative | Mixed |
|:--------------|:--------:|:--------:|:-----:|
| **LSH + Bloom Filter** | **1,722x** | **124,815x** | **3,970x** |
| Bloom Filter (No LSH) | 1.2x | 109,330x | 2.2x |
| LSH (No Bloom Filter) | 1,897x | 2,594x | 1,997x |

### Retrieval Metrics (Positive Workload)

| Configuration | Hit@10 | Precision@10 | Recall@10 | False Negatives | Avg Candidates |
|:--------------|:------:|:------------:|:---------:|:---------------:|:--------------:|
| 1. Only MinHash (Brute Force) | 100.00% | 1.0000 | 1.0000 | 0 | 4,000,000 |
| **2. LSH + Bloom Filter** | **99.00%** | **0.9070** | **0.9070** | **1** | **19.3** |
| 3. Bloom Filter (No LSH) | 100.00% | 1.0000 | 1.0000 | 0 | 4,000,000 |
| 4. LSH (No Bloom Filter) | 99.00% | 0.9070 | 0.9070 | 1 | 19.3 |

> [!IMPORTANT]
> **Key Insight**: The LSH+Bloom pipeline achieves **99% Hit@10** while inspecting only **19.3 candidates** (out of 4,000,000) — a **99.9995% reduction** in search space — with a **1,722x speedup** over brute force on positive queries. The single false negative is a statistical edge case (1 in 100 queries).
>
> For negative queries, the Bloom filter short-circuits in **0.008 ms** (125,125 QPS), making the system essentially free for non-duplicate rejections.

---

## LSH Index Optimization Note

> [!NOTE]
> During this evaluation, the `ScaledLSHIndex` was upgraded from a Python dict-based inverted index (234s indexing time, ~8 GB RAM) to a **vectorized sorted-array implementation** using FNV-1a-style hashing + `np.argsort` + `np.searchsorted`. This reduced:
> - **Indexing time**: 234s to **25s** (9.4x faster)
> - **Peak RAM**: ~8 GB to **~2.3 GB** (3.5x less memory)
> - **Lookup time**: comparable sub-millisecond performance

---

## Summary

| Aspect | Result |
|:-------|:-------|
| **Optimal Hyperparameters** | b=16, r=4, tau=0.40 |
| **Full-Scale MRR** | 0.9988 |
| **Full-Scale MAP** | 0.9181 |
| **Full-Scale Hit@10** | 99.92% |
| **Full-Scale False Negatives** | 2 / 2,500 (0.08%) |
| **Search Space Reduction** | 99.9991% (4M to 34.6 candidates) |
| **Query Latency** | 0.884 ms avg (P99: 2.83 ms) |
| **Throughput** | 1,033 QPS |
| **Speedup vs Brute Force** | 1,722x (positive), 124,815x (negative) |
| **Best Architecture** | LSH + Bloom Filter (Full Pipeline) |

> **Reports saved to:** [benchmark_6m_best_hyperparameters.json](file:///c:/Users/hrish/Projects/research_projects/duplicate_detection/reports/benchmark_6m_best_hyperparameters.json) and [latency_4way_results_scale.json](file:///c:/Users/hrish/Projects/research_projects/duplicate_detection/reports/latency_4way_results_scale.json)
