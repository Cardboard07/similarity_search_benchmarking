# Bloom Filter Parameters & System Documentation
**Project:** Fingerprint Duplicate Detection and Retrieval  
**Module:** Locality-Sensitive Hashing (LSH) & Probabilistic Pre-Search Filtering  

---

## 1. Executive Summary

In biometric identification and duplicate detection systems, full nearest-neighbor searches (MinHash signature generation, LSH band bucket lookups, and candidate Jaccard scoring) are computationally expensive. 

To eliminate unnecessary computational overhead on negative queries (fingerprints or identities that do not exist in the database), we introduced a **Bloom Filter pre-search gating layer**. 

Before committing to the LSH index lookup and Jaccard similarity scoring:
1. The query fingerprint is screened by the **Bloom Filter**.
2. If the filter returns `False` (**Definite Negative**), the query is aborted immediately with **zero search cost**.
3. If the filter returns `True` (**Probable Positive**), the query proceeds to LSH candidate retrieval and similarity ranking.

---

## 2. Complete Parameter Specification

The Bloom Filter layer provides three specialized configurations based on the granularity of filtering required:

| Parameter | Identity Filter (Default Gating) | Record Filter (Exact Duplicate Check) | Band Filter (LSH Sub-signature Gating) |
| :--- | :--- | :--- | :--- |
| **Purpose** | Check if fingerprint identity / subject is registered | Check if exact fingerprint file / impression exists | Check if query shares any LSH band buckets with DB |
| **Dataset Elements ($n$)** | **1,000** unique identities | **10,000** training records | **53,224** unique (band, signature) tuples |
| **Target False Positive Rate ($p$)** | **0.001** ($0.1\%$) | **0.001** ($0.1\%$) | **0.001** ($0.1\%$) |
| **Bit Array Size ($m$)** | **14,378 bits** | **143,776 bits** | **765,233 bits** |
| **Memory Footprint (Bytes)** | **1,798 bytes** | **17,972 bytes** | **95,655 bytes** |
| **Memory Footprint (KB)** | **1.76 KB** | **17.55 KB** | **93.41 KB** |
| **Number of Hash Functions ($k$)** | **10** | **10** | **10** |
| **Bits per Element ($m / n$)** | **14.38 bits/element** | **14.38 bits/element** | **14.38 bits/element** |
| **Bits Set to 1 (Ones Count)** | 7,234 | 72,173 | 383,599 |
| **Bit Array Fill Ratio** | **50.31%** (optimal $\approx 50\%$) | **50.20%** (optimal $\approx 50\%$) | **50.13%** (optimal $\approx 50\%$) |
| **Theoretical FPR** | **0.001039** ($0.104\%$) | **0.001016** ($0.102\%$) | **0.001002** ($0.100\%$) |
| **Empirical FPR (on 10,000 queries)** | **0.0094** | < 0.002 | < 0.002 |
| **False Negative Rate** | **0.00%** (Guaranteed 0) | **0.00%** (Guaranteed 0) | **0.00%** (Guaranteed 0) |

---

## 3. Mathematical Formulations & Derivations

### 3.1 Optimal Bit Array Sizing ($m$)
Given $n$ expected elements to insert and a maximum acceptable false positive probability $p$, the minimum number of bits $m$ required is derived from the collision probability:

$$p \approx \left(1 - e^{-k n / m}\right)^k$$

Minimizing $p$ with respect to $k$ yields $k = \frac{m}{n} \ln 2$. Substituting this back into the formula yields the optimal bit array size:

$$m = -\frac{n \ln p}{(\ln 2)^2} \approx -1.442695 \cdot n \log_2(p)$$

For $p = 0.001$:
$$\frac{m}{n} = -\frac{\ln(0.001)}{(\ln 2)^2} = \frac{6.907755}{0.480453} \approx 14.3776 \text{ bits/element}$$

- For $n = 1,000$ identities: $m = \lceil 1,000 \times 14.3776 \rceil = \mathbf{14,378\text{ bits}}$ ($\approx 1.76\text{ KB}$)
- For $n = 10,000$ records: $m = \lceil 10,000 \times 14.3776 \rceil = \mathbf{143,776\text{ bits}}$ ($\approx 17.55\text{ KB}$)
- For $n = 53,224$ bands: $m = \lceil 53,224 \times 14.3776 \rceil = \mathbf{765,233\text{ bits}}$ ($\approx 93.41\text{ KB}$)

### 3.2 Optimal Number of Hash Functions ($k$)
The optimal number of independent hash functions that minimizes the false positive rate for a given ratio $m/n$:

$$k = \frac{m}{n} \ln 2 = -\log_2(p)$$

For $p = 0.001$:
$$k = -\log_2(0.001) \approx 9.96578 \implies \mathbf{k = 10\text{ hash functions}}$$

### 3.3 Hashing Scheme: Kirsch-Mitzenmacher Optimization
Instead of evaluating $k=10$ separate cryptographic hash functions (which would be slow), we utilize the **Kirsch-Mitzenmacher double-hashing technique**. This theorem proves that $k$ independent hash functions can be simulated using only two independent base hash values $h_1(x)$ and $h_2(x)$ without asymptotic loss in the false positive rate:

$$h_i(x) = \left(h_1(x) + i \cdot h_2(x)\right) \pmod m, \quad \text{for } i \in \{0, 1, \dots, k-1\}$$

- **$h_1(x)$**: Generated via `hashlib.blake2b` (digest size 8 bytes, salt `b"bloom_h1"`).
- **$h_2(x)$**: Generated via `hashlib.blake2b` (digest size 8 bytes, salt `b"bloom_h2"`).
- If $h_2(x) = 0$, it is set to $1$ to ensure full cycle traversal.

---

## 4. Architecture & Implementation

### 4.1 Custom `BitArray` Class
Located in [bloom_filter.py](file:///c:/Users/hrish/Projects/research_projects/duplicate_detection/LSH/main/bloom_filter.py):
- Backed by Python's native `bytearray` (1 byte = 8 bits).
- Zero external C-extension dependencies (ensuring cross-platform portability on Windows).
- **Bitwise Indexing**:
  - Byte index: `byte_idx = index >> 3` (equivalent to `index // 8`)
  - Bit mask: `bit_mask = 1 << (index & 7)` (equivalent to `1 << (index % 8)`)
- **Operations**:
  - `set(index)`: `self._array[byte_idx] |= bit_mask`
  - `get(index)`: `bool(self._array[byte_idx] & bit_mask)`
  - `count_ones()`: Uses fast population count via bit-level popcount.
  - Binary serialization: `to_bytes()`, `from_bytes()`, `save()`, `load()`.

### 4.2 Application Integration Points
1. **[similarity.py](file:///c:/Users/hrish/Projects/research_projects/duplicate_detection/LSH/main/similarity.py)**:
   - Automatically screens single query fingerprints.
   - If not in database:
     ```
     [Bloom Filter Check] Result: NOT IN DATABASE
                          Detail: Identity '99999' definitively not found in database.
     [Bloom Filter Pruning] Skipped LSH search & Jaccard scoring (zero search cost).
     ```
   - If in database:
     ```
     [Bloom Filter Check] Result: IN DATABASE
                          Detail: Identity '1' probably in database.
     [Search Committed] Querying LSH candidate index...
     ```

2. **[evaluate_similarity.py](file:///c:/Users/hrish/Projects/research_projects/duplicate_detection/LSH/main/evaluate_similarity.py)**:
   - Full benchmarking harness with automated Bloom filter gating, timing breakdowns, and diagnostic telemetry.

---

## 5. Performance Benchmarks

### 5.1 Latency Breakdown Comparison

| Pipeline Stage | Without Bloom Filter | With Bloom Filter (Positive Hit) | With Bloom Filter (Negative Query) |
| :--- | :--- | :--- | :--- |
| **Bloom Filter Gating** | 0.00 ms | **0.02 ms** (20 µs) | **0.008 ms** (8 µs) |
| **MinHash Signature** | 7.51 ms | 7.51 ms | **0.00 ms** (Skipped) |
| **LSH Candidate Lookup** | 0.02 ms | 0.02 ms | **0.00 ms** (Skipped) |
| **Jaccard Ranking** | 0.05 ms | 0.05 ms | **0.00 ms** (Skipped) |
| **Total Query Latency** | **7.58 ms** | **7.60 ms** | **0.008 ms** |
| **Speedup on Negatives** | Baseline (1x) | ~1.0x | **~950x faster** |

### 5.2 Key Takeaways
- **Zero False Negatives**: 100% of all valid ground-truth database queries pass through the Bloom Filter without disruption.
- **Negligible Overhead**: Adding the Bloom Filter costs less than 25 microseconds ($0.025\text{ ms}$) per query, which is $< 0.3\%$ of total pipeline latency.
- **Massive Savings on Non-Duplicates**: Any non-existent fingerprint query is discarded in under 10 microseconds, saving $> 99.8\%$ of compute time.
- **Ultra-Lightweight Footprint**: All three filters combined consume less than $115\text{ KB}$ of RAM.

---

## 6. Empirical 4-Way Latency Comparison

Benchmark performed over 100 queries per configuration evaluating:
1. **Only MinHash (No LSH, No Bloom)**: Brute-force linear scan against all 10,000 database signatures.
2. **LSH + Bloom Filter**: Full production pipeline with Bloom gating and LSH band candidate lookup.
3. **Bloom Filter (No LSH)**: Bloom filter gating followed by linear scan on positive admission.
4. **LSH (No Bloom Filter)**: LSH band index lookup without Bloom filter pre-checking.

### Benchmark Results Table

| Configuration | Positive Queries (In DB) | Negative Queries (Out of DB) | Mixed Workload (50% Pos / 50% Neg) | Search Throughput (Mixed) |
| :--- | :--- | :--- | :--- | :--- |
| **[1] Only MinHash (No LSH, No Bloom)** | 75.81 ms / query | 63.59 ms / query | 66.50 ms / query | 15.0 QPS |
| **[2] LSH + Bloom Filter** | **7.14 ms / query** | **0.006 ms / query** | **3.55 ms / query** | **281.9 QPS** |
| **[3] Bloom Filter (No LSH)** | 71.92 ms / query | **0.006 ms / query** | 35.18 ms / query | 28.4 QPS |
| **[4] LSH (No Bloom Filter)** | **6.98 ms / query** | 0.166 ms / query | 3.62 ms / query | 276.0 QPS |

### Insights:
- **Impact of LSH**: Reduces comparison time from **75.8 ms** down to **7.0 ms** on positive queries (**~10.8x speedup**), because candidate search replaces scanning 10,000 documents with evaluating only ~4 candidate documents.
- **Impact of Bloom Filter on Negatives**: Drops latency from **63.6 ms** (brute force) and **0.17 ms** (LSH) down to **0.006 ms** (6 microseconds), a **~10,000x speedup** over brute force and **~27x speedup** over LSH alone.
- **Best Overall Synergy**: **LSH + Bloom Filter** achieves the lowest average query time (**3.55 ms**) and highest throughput (**282 queries/sec**) under mixed workloads.

