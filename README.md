# Fingerprint Duplicate Detection

Biometric fingerprint duplicate detection system using MinHash, Locality-Sensitive Hashing (LSH), and Bloom Filter pre-search gating.

See full documentation and benchmarks in [LSH/README.md](LSH/README.md).

---

## 4-Way Latency Comparison Results

| Configuration | Positive Queries (In DB) | Negative Queries (Not in DB) | Mixed Workload (50% Pos / 50% Neg) | Search Throughput (Mixed) |
| :--- | :---: | :---: | :---: | :---: |
| **1. Only MinHash (No LSH, No Bloom)** | 75.81 ms | 63.59 ms | 66.50 ms | 15.0 queries/sec |
| **2. LSH + Bloom Filter** | **7.14 ms** | **0.006 ms** | **3.55 ms** | **281.9 queries/sec** |
| **3. Bloom Filter (No LSH)** | 71.92 ms | **0.006 ms** | 35.18 ms | 28.4 queries/sec |
| **4. LSH (No Bloom Filter)** | **6.98 ms** | 0.166 ms | 3.62 ms | 276.0 queries/sec |

### Key Takeaways:
- **LSH**: Provides **~10.8x speedup** on positive queries (75.81 ms $\to$ 6.98 ms) by pruning candidate comparisons from 10,000 documents down to ~4 documents.
- **Bloom Filter**: Discards non-existent fingerprints in **0.006 ms (6 µs)**, providing **~27x speedup over LSH** and **~10,000x speedup over brute force**.
- **Combined (LSH + Bloom)**: Delivers lowest mixed workload latency (**3.55 ms**) and highest throughput (**281.9 QPS**) with **0.00% False Negatives**.

---

## Project Structure
- `LSH/main/bloom_filter.py`: Custom `BitArray` and `BloomFilter` implementation.
- `LSH/main/lsh.py`: LSH index partitioning and candidate retrieval.
- `LSH/main/similarity.py`: Interactive query retrieval with Bloom Filter pre-check.
- `LSH/main/evaluate_similarity.py`: Evaluation benchmark harness.
- `LSH/scripts/latency_comparison_4ways.py`: 4-way latency benchmark script.
- `LSH/BLOOM_FILTER_PARAMETERS.md`: Full mathematical parameter derivation and documentation.
- `LSH/latency_4way_results.json`: Raw JSON metrics output.
