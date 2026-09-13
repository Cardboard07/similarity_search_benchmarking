import os
import sys
import time
import json
import glob
from pathlib import Path
from typing import List, Dict, Any

# Ensure main directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main")))

from lsh import find_cand, dicts, write_f, data as train_data
from min_hash import minhash_signature, hash_funcs
from similarity import jaccard, score_jaccard, get_bloom_filter


def run_latency_test(num_queries: int = 100, include_negatives: bool = True):
    print("=" * 70)
    print(f"   4-WAY LATENCY COMPARISON BENCHMARK ({num_queries} queries per mode)")
    print("=" * 70)

    # 1. Load Bloom filter
    print("Loading Bloom filter...")
    bloom_mgr = get_bloom_filter()

    # 2. Prepare test queries
    test_dir = Path(__file__).resolve().parent.parent.parent / "fingerprint_dataset" / "parsed_data" / "test"
    test_files = sorted(glob.glob(str(test_dir / "*.json")))[:num_queries]

    # Pre-load raw queries
    positive_queries = []
    for tf in test_files:
        with open(tf, "r", encoding="utf-8") as f:
            raw = json.load(f)
        positive_queries.append({
            "file_name": raw.get("file_name", os.path.basename(tf)),
            "relative_tokens": raw["relative_tokens"]
        })

    # Prepare negative queries (identities not in DB, e.g. 5000+)
    negative_queries = [
        {
            "file_name": f"q1_{9000 + i}_1.json",
            "relative_tokens": [[1, 2, 3, 4, 5], [10, 20, 30, 1, 1]]
        }
        for i in range(num_queries)
    ]

    workloads = [
        ("Positive Queries (In Database)", positive_queries),
    ]
    if include_negatives:
        # 50/50 mixed workload
        half = num_queries // 2
        mixed_queries = positive_queries[:half] + negative_queries[:half]
        workloads.append(("Negative Queries (Not In Database)", negative_queries))
        workloads.append(("Mixed Workload (50% Positive / 50% Negative)", mixed_queries))

    all_results = {}

    for workload_name, queries in workloads:
        print(f"\n>>> Running Workload: {workload_name} ({len(queries)} queries) <<<")
        workload_results = {}

        # ----------------------------------------------------
        # Mode 1: Only MinHash (No LSH, No Bloom) - Linear Scan
        # ----------------------------------------------------
        t0 = time.perf_counter()
        for q in queries:
            sig = minhash_signature(q["relative_tokens"], hash_funcs)
            doc_rec = {"file_name": q["file_name"], "signatures": sig}
            # Linear scan over ALL 10,000 database signatures
            similarities = [
                (td["file_name"], jaccard(td["signatures"], sig))
                for td in train_data
            ]
            top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        t1 = time.perf_counter()
        mode1_time_ms = ((t1 - t0) / len(queries)) * 1000
        mode1_total_sec = t1 - t0
        workload_results["1_only_minhash"] = {
            "name": "Only MinHash (No LSH, No Bloom)",
            "avg_latency_ms": mode1_time_ms,
            "total_sec": mode1_total_sec,
            "throughput_qps": len(queries) / mode1_total_sec
        }
        print(f"  [1] Only MinHash (No LSH, No Bloom) : {mode1_time_ms:8.2f} ms / query")

        # ----------------------------------------------------
        # Mode 2: LSH and Bloom Filter
        # ----------------------------------------------------
        t0 = time.perf_counter()
        for q in queries:
            is_in_db, _ = bloom_mgr.check_fingerprint(q, mode="identity")
            if not is_in_db:
                continue  # Short-circuit without search!
            sig = minhash_signature(q["relative_tokens"], hash_funcs)
            doc_rec = {"file_name": q["file_name"], "signatures": sig}
            candidates = find_cand(dicts, doc_rec)
            top_matches = score_jaccard(candidates, doc_rec, k=10)
        t1 = time.perf_counter()
        mode2_time_ms = ((t1 - t0) / len(queries)) * 1000
        mode2_total_sec = t1 - t0
        workload_results["2_lsh_and_bloom"] = {
            "name": "LSH + Bloom Filter",
            "avg_latency_ms": mode2_time_ms,
            "total_sec": mode2_total_sec,
            "throughput_qps": len(queries) / mode2_total_sec
        }
        print(f"  [2] LSH + Bloom Filter               : {mode2_time_ms:8.2f} ms / query")

        # ----------------------------------------------------
        # Mode 3: Bloom Filter but No LSH (Linear Scan on positive)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        for q in queries:
            is_in_db, _ = bloom_mgr.check_fingerprint(q, mode="identity")
            if not is_in_db:
                continue  # Short-circuit without search!
            sig = minhash_signature(q["relative_tokens"], hash_funcs)
            doc_rec = {"file_name": q["file_name"], "signatures": sig}
            similarities = [
                (td["file_name"], jaccard(td["signatures"], sig))
                for td in train_data
            ]
            top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
        t1 = time.perf_counter()
        mode3_time_ms = ((t1 - t0) / len(queries)) * 1000
        mode3_total_sec = t1 - t0
        workload_results["3_bloom_no_lsh"] = {
            "name": "Bloom Filter (No LSH)",
            "avg_latency_ms": mode3_time_ms,
            "total_sec": mode3_total_sec,
            "throughput_qps": len(queries) / mode3_total_sec
        }
        print(f"  [3] Bloom Filter (No LSH)            : {mode3_time_ms:8.2f} ms / query")

        # ----------------------------------------------------
        # Mode 4: LSH but No Bloom Filter
        # ----------------------------------------------------
        t0 = time.perf_counter()
        for q in queries:
            sig = minhash_signature(q["relative_tokens"], hash_funcs)
            doc_rec = {"file_name": q["file_name"], "signatures": sig}
            candidates = find_cand(dicts, doc_rec)
            top_matches = score_jaccard(candidates, doc_rec, k=10)
        t1 = time.perf_counter()
        mode4_time_ms = ((t1 - t0) / len(queries)) * 1000
        mode4_total_sec = t1 - t0
        workload_results["4_lsh_no_bloom"] = {
            "name": "LSH (No Bloom Filter)",
            "avg_latency_ms": mode4_time_ms,
            "total_sec": mode4_total_sec,
            "throughput_qps": len(queries) / mode4_total_sec
        }
        print(f"  [4] LSH (No Bloom Filter)            : {mode4_time_ms:8.2f} ms / query")

        all_results[workload_name] = workload_results

    return all_results


if __name__ == "__main__":
    results = run_latency_test(num_queries=100, include_negatives=True)
    out_path = Path(__file__).resolve().parent.parent / "latency_4way_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved benchmark results to {out_path}")
