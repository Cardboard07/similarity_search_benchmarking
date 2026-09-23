"""
4-Way Latency Comparison Benchmark at Scale.

Replicates the 4 architectural configurations from LSH/README.md:
  1. Only MinHash (No LSH, No Bloom) - Brute force linear scan
  2. LSH + Bloom Filter - Full production pipeline
  3. Bloom Filter (No LSH) - Bloom gating with linear scan on positive admission
  4. LSH (No Bloom Filter) - LSH candidate index without Bloom pre-search gating
Across Positive, Negative, and Mixed workloads.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from src.lsh_scale.bloom_filter_6m import ScaledIdentityBloomFilter
from src.lsh_scale.lsh_index_6m import ScaledLSHIndex
from src.lsh_scale.hdf5_signature_extractor import ShardedSignatureDataset


def run_latency_comparison(
    shards_dir: str,
    bloom_path: str,
    num_queries: int = 100,
    bands: int = 8,
    max_gallery_records: int = 100000,
    output_path: str = "reports/latency_4way_results_scale.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print("        4-WAY LATENCY COMPARISON BENCHMARK AT SCALE")
    print("=" * 70)

    # 1. Load Bloom Filter
    print(f"Loading Bloom Filter from {bloom_path}...")
    bloom_mgr = ScaledIdentityBloomFilter.load(bloom_path)
    print(f"Bloom Filter loaded: capacity={bloom_mgr.capacity:,}, k={bloom_mgr.k}, bits={bloom_mgr.m:,}")

    # 2. Load Gallery Dataset
    print(f"\nLoading Gallery Signatures from {shards_dir}...")
    dataset = ShardedSignatureDataset(shards_dir)
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    print(f"Gallery loaded: {num_gallery:,} records (Identities: {len(np.unique(gallery_ids)):,})")

    # 3. Build LSH Index over Gallery
    print(f"\nBuilding LSH Index (bands={bands}, rows={64 // bands}) over {num_gallery:,} records...")
    t_idx0 = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands)
    lsh_index.index_signatures(gallery_sigs)
    t_idx1 = time.perf_counter()
    print(f"LSH Index built in {t_idx1 - t_idx0:.2f} seconds.")

    # 4. Prepare Positive and Negative Queries
    print(f"\nLoading {num_queries} Positive Probe Queries...")
    probe_sigs, probe_ids, probe_oris, probe_quals = dataset.load_probes(max_records=num_queries)
    actual_pos_queries = len(probe_sigs)
    print(f"Loaded {actual_pos_queries} positive probe queries.")

    positive_queries = [
        {"identity": int(probe_ids[i]), "signatures": probe_sigs[i]}
        for i in range(actual_pos_queries)
    ]

    print(f"Generating {num_queries} Negative (Unseen Identity) Queries...")
    max_u32 = int(np.iinfo(np.uint32).max)
    negative_queries = [
        {
            "identity": int(9000000 + i),
            "signatures": np.random.randint(0, max_u32, size=64, dtype=np.uint32)
        }
        for i in range(actual_pos_queries)
    ]

    half = actual_pos_queries // 2
    mixed_queries = positive_queries[:half] + negative_queries[:half]

    workloads = [
        ("Positive Queries (In Database)", positive_queries),
        ("Negative Queries (Not In Database)", negative_queries),
        ("Mixed Workload (50% Positive / 50% Negative)", mixed_queries)
    ]

    all_results = {
        "metadata": {
            "gallery_size": num_gallery,
            "queries_per_workload": actual_pos_queries,
            "bands": bands,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "workloads": {}
    }

    for workload_name, queries in workloads:
        print("\n" + "=" * 70)
        print(f">>> Running Workload: {workload_name} ({len(queries)} queries) <<<")
        print("=" * 70)
        workload_res = {}

        # ------------------------------------------------------------------
        # Mode 1: Only MinHash (No LSH, No Bloom) - Brute Force Linear Scan
        # ------------------------------------------------------------------
        print(f"Running Mode 1: Only MinHash (Linear Scan against {num_gallery:,} docs)...")
        t0 = time.perf_counter()
        for q in queries:
            q_sig = q["signatures"]
            # Vectorized Jaccard similarity across all gallery signatures
            sims = np.mean(gallery_sigs == q_sig, axis=1)
            # Top-10 candidates
            top_indices = np.argpartition(sims, -10)[-10:]
        t1 = time.perf_counter()

        m1_sec = t1 - t0
        m1_ms = (m1_sec / len(queries)) * 1000.0
        m1_qps = len(queries) / m1_sec if m1_sec > 0 else 0.0
        workload_res["1_only_minhash"] = {
            "name": "Only MinHash (No LSH, No Bloom)",
            "avg_latency_ms": round(m1_ms, 3),
            "total_sec": round(m1_sec, 3),
            "throughput_qps": round(m1_qps, 1)
        }
        print(f"  [1] Only MinHash (No LSH, No Bloom) : {m1_ms:8.2f} ms / query ({m1_qps:6.1f} QPS)")

        # ------------------------------------------------------------------
        # Mode 2: LSH + Bloom Filter (Production Pipeline)
        # ------------------------------------------------------------------
        print("Running Mode 2: LSH + Bloom Filter (Full Pipeline)...")
        t0 = time.perf_counter()
        for q in queries:
            # 1. Bloom Filter screening
            if not bloom_mgr.contains(q["identity"]):
                continue  # Zero-cost short-circuit!

            # 2. LSH candidate retrieval
            cands = lsh_index.find_candidates(q["signatures"])

            # 3. Candidate scoring
            scores = ScaledLSHIndex.score_candidates_vectorized(
                cands, q["signatures"], gallery_sigs, k=10, use_set_jaccard=False
            )
        t1 = time.perf_counter()

        m2_sec = t1 - t0
        m2_ms = (m2_sec / len(queries)) * 1000.0
        m2_qps = len(queries) / m2_sec if m2_sec > 0 else 0.0
        workload_res["2_lsh_and_bloom"] = {
            "name": "LSH + Bloom Filter",
            "avg_latency_ms": round(m2_ms, 3),
            "total_sec": round(m2_sec, 3),
            "throughput_qps": round(m2_qps, 1)
        }
        print(f"  [2] LSH + Bloom Filter               : {m2_ms:8.2f} ms / query ({m2_qps:6.1f} QPS)")

        # ------------------------------------------------------------------
        # Mode 3: Bloom Filter (No LSH) - Linear scan on positive admission
        # ------------------------------------------------------------------
        print("Running Mode 3: Bloom Filter (No LSH)...")
        t0 = time.perf_counter()
        for q in queries:
            # 1. Bloom Filter screening
            if not bloom_mgr.contains(q["identity"]):
                continue  # Short-circuit on negative!

            # Fallback to brute-force linear scan on positive admission
            sims = np.mean(gallery_sigs == q["signatures"], axis=1)
            top_indices = np.argpartition(sims, -10)[-10:]
        t1 = time.perf_counter()

        m3_sec = t1 - t0
        m3_ms = (m3_sec / len(queries)) * 1000.0
        m3_qps = len(queries) / m3_sec if m3_sec > 0 else 0.0
        workload_res["3_bloom_no_lsh"] = {
            "name": "Bloom Filter (No LSH)",
            "avg_latency_ms": round(m3_ms, 3),
            "total_sec": round(m3_sec, 3),
            "throughput_qps": round(m3_qps, 1)
        }
        print(f"  [3] Bloom Filter (No LSH)            : {m3_ms:8.2f} ms / query ({m3_qps:6.1f} QPS)")

        # ------------------------------------------------------------------
        # Mode 4: LSH (No Bloom Filter)
        # ------------------------------------------------------------------
        print("Running Mode 4: LSH (No Bloom Filter)...")
        t0 = time.perf_counter()
        for q in queries:
            # LSH candidate lookup directly
            cands = lsh_index.find_candidates(q["signatures"])
            scores = ScaledLSHIndex.score_candidates_vectorized(
                cands, q["signatures"], gallery_sigs, k=10, use_set_jaccard=False
            )
        t1 = time.perf_counter()

        m4_sec = t1 - t0
        m4_ms = (m4_sec / len(queries)) * 1000.0
        m4_qps = len(queries) / m4_sec if m4_sec > 0 else 0.0
        workload_res["4_lsh_no_bloom"] = {
            "name": "LSH (No Bloom Filter)",
            "avg_latency_ms": round(m4_ms, 3),
            "total_sec": round(m4_sec, 3),
            "throughput_qps": round(m4_qps, 1)
        }
        print(f"  [4] LSH (No Bloom Filter)            : {m4_ms:8.2f} ms / query ({m4_qps:6.1f} QPS)")

        all_results["workloads"][workload_name] = workload_res

    # Print Summary Table
    print("\n" + "=" * 80)
    print("                     4-WAY LATENCY COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'Positive (In DB)':<16} | {'Negative (Out DB)':<17} | {'Mixed (50/50)':<13}")
    print("-" * 88)

    pos_res = all_results["workloads"]["Positive Queries (In Database)"]
    neg_res = all_results["workloads"]["Negative Queries (Not In Database)"]
    mix_res = all_results["workloads"]["Mixed Workload (50% Positive / 50% Negative)"]

    for k in ["1_only_minhash", "2_lsh_and_bloom", "3_bloom_no_lsh", "4_lsh_no_bloom"]:
        c_name = pos_res[k]["name"]
        p_lat = f"{pos_res[k]['avg_latency_ms']:.2f} ms"
        n_lat = f"{neg_res[k]['avg_latency_ms']:.4f} ms"
        m_lat = f"{mix_res[k]['avg_latency_ms']:.2f} ms"
        print(f"{c_name:<35} | {p_lat:<16} | {n_lat:<17} | {m_lat:<13}")
    print("=" * 88)

    # Save to JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved raw JSON benchmark results to {out_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-way latency comparison at scale.")
    parser.add_argument(
        "--shards-dir",
        type=str,
        default=str(workspace_root / "dataset" / "signatures_shards"),
        help="Directory containing extracted signatures_*.h5 shards."
    )
    parser.add_argument(
        "--bloom-path",
        type=str,
        default=str(workspace_root / "dataset" / "bloom_filter_400k"),
        help="Path prefix to 400k Bloom filter (.bin/.json)."
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries per workload."
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=8,
        help="Number of LSH bands."
    )
    parser.add_argument(
        "--max-gallery",
        type=int,
        default=100000,
        help="Max gallery records to index."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/latency_4way_results_scale.json",
        help="Output JSON path."
    )

    args = parser.parse_args()
    run_latency_comparison(
        shards_dir=args.shards_dir,
        bloom_path=args.bloom_path,
        num_queries=args.num_queries,
        bands=args.bands,
        max_gallery_records=args.max_gallery,
        output_path=args.output
    )
