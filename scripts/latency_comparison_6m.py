"""
4-Way Architecture Comparison Benchmark at Scale (6M Data Points).

Evaluates the 4 architectural configurations from LSH/README.md:
  1. Only MinHash (No LSH, No Bloom) - Brute force linear scan
  2. LSH + Bloom Filter - Full production pipeline
  3. Bloom Filter (No LSH) - Bloom gating with linear scan on positive admission
  4. LSH (No Bloom Filter) - LSH candidate index without Bloom pre-search gating

Evaluates across:
  - Positive Workload (Identities in database)
  - Negative Workload (Unseen identities not in database)
  - Mixed Workload (50% Positive / 50% Negative)

Records BOTH:
  - Latency Profiles: Mean, P50, P90, P99, Throughput (QPS), Speedup vs Brute Force
  - Retrieval Accuracy Metrics: Hit@10, Precision@10, Recall@10, False Negatives, Candidates Inspected
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
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
    bands: int = 16,
    rows: int = 4,
    max_gallery_records: Optional[int] = None,
    output_path: str = "reports/latency_4way_results_scale.json"
) -> Dict[str, Any]:
    print("=" * 84)
    print("       4-WAY ARCHITECTURAL COMPARISON: METRICS & LATENCY AT SCALE")
    print("=" * 84)

    # 1. Load Bloom Filter
    print(f"Loading Bloom Filter from {bloom_path}...")
    bloom_mgr = ScaledIdentityBloomFilter.load(bloom_path)
    print(f"Bloom Filter loaded: capacity={bloom_mgr.capacity:,}, k={bloom_mgr.k}, bits={bloom_mgr.m:,}")

    # 2. Load Gallery Dataset
    gallery_desc = f"max {max_gallery_records:,}" if max_gallery_records else "all 40 shards (4,000,000)"
    print(f"\nLoading Gallery Signatures from {shards_dir} ({gallery_desc} records)...")
    dataset = ShardedSignatureDataset(shards_dir)
    t0_load = time.perf_counter()
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    print(f"Gallery loaded: {num_gallery:,} records in {time.perf_counter() - t0_load:.2f}s.")

    # Build identity -> list of gallery record indices
    print("Building ground-truth identity index...")
    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        ident = int(ident)
        if ident not in gt_map:
            gt_map[ident] = []
        gt_map[ident].append(idx)
    print(f"Indexed {len(gt_map):,} unique gallery identities.")

    # 3. Build LSH Index over Gallery
    print(f"\nBuilding LSH Index (bands={bands}, rows={rows}) over {num_gallery:,} records...")
    t_idx0 = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands, rows=rows)
    lsh_index.index_signatures(gallery_sigs)
    t_idx1 = time.perf_counter()
    indexing_time = t_idx1 - t_idx0
    print(f"LSH Index built in {indexing_time:.2f} seconds ({num_gallery/indexing_time:,.0f} records/s).")

    # 4. Prepare Positive and Negative Queries
    print(f"\nLoading {num_queries} Positive Probe Queries...")
    probe_sigs, probe_ids, probe_oris, probe_quals = dataset.load_probes(max_records=num_queries)
    actual_pos_queries = len(probe_sigs)
    print(f"Loaded {actual_pos_queries} positive probe queries.")

    positive_queries = [
        {"identity": int(probe_ids[i]), "signatures": probe_sigs[i], "in_db": True}
        for i in range(actual_pos_queries)
    ]

    print(f"Generating {num_queries} Negative (Unseen Identity) Queries...")
    max_u32 = int(np.iinfo(np.uint32).max)
    negative_queries = [
        {
            "identity": int(9000000 + i),
            "signatures": np.random.randint(0, max_u32, size=64, dtype=np.uint32),
            "in_db": False
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
            "rows": rows,
            "theoretical_threshold": round((1.0 / bands) ** (1.0 / rows), 4),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "workloads": {}
    }

    for workload_name, queries in workloads:
        print("\n" + "=" * 84)
        print(f">>> Running Workload: {workload_name} ({len(queries)} queries) <<<")
        print("=" * 84)
        workload_res = {}

        # ------------------------------------------------------------------
        # Mode 1: Only MinHash (No LSH, No Bloom) - Brute Force Linear Scan
        # ------------------------------------------------------------------
        print(f"Running Mode 1: Only MinHash (Linear Scan against {num_gallery:,} docs)...")
        latencies_m1 = []
        hits_m1 = 0
        prec_m1 = 0.0
        rec_m1 = 0.0
        fn_m1 = 0
        cands_m1 = 0
        eval_count = 0

        for q in queries:
            t0 = time.perf_counter()
            q_sig = q["signatures"]
            # Vectorized Jaccard similarity across all gallery signatures
            sims = np.mean(gallery_sigs == q_sig, axis=1)
            # Top-10 candidates
            top_indices = np.argpartition(sims, -10)[-10:]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
            t1 = time.perf_counter()

            latencies_m1.append((t1 - t0) * 1000.0)
            cands_m1 += num_gallery

            if q["in_db"]:
                eval_count += 1
                q_id = q["identity"]
                gt = gt_map.get(q_id, [])
                match_count = sum(1 for idx in top_indices if gallery_ids[idx] == q_id)
                if match_count > 0:
                    hits_m1 += 1
                else:
                    fn_m1 += 1
                prec_m1 += match_count / 10.0
                if len(gt) > 0:
                    rec_m1 += match_count / len(gt)

        m1_arr = np.array(latencies_m1)
        m1_sec = sum(latencies_m1) / 1000.0
        m1_ms = float(np.mean(m1_arr))
        m1_qps = len(queries) / m1_sec if m1_sec > 0 else 0.0
        workload_res["1_only_minhash"] = {
            "name": "Only MinHash (No LSH, No Bloom)",
            "avg_latency_ms": round(m1_ms, 3),
            "p50_latency_ms": round(float(np.median(m1_arr)), 3),
            "p99_latency_ms": round(float(np.percentile(m1_arr, 99)), 3),
            "total_sec": round(m1_sec, 3),
            "throughput_qps": round(m1_qps, 1),
            "avg_candidates": num_gallery,
            "hit_at_10_pct": round((hits_m1 / eval_count * 100.0), 2) if eval_count > 0 else None,
            "precision_at_10": round(prec_m1 / eval_count, 4) if eval_count > 0 else None,
            "recall_at_10": round(rec_m1 / eval_count, 4) if eval_count > 0 else None,
            "false_negatives": fn_m1 if eval_count > 0 else 0
        }
        print(f"  [1] Only MinHash (No LSH, No Bloom) : {m1_ms:8.2f} ms / query ({m1_qps:6.1f} QPS) | Hit@10: {workload_res['1_only_minhash']['hit_at_10_pct']}%")

        # ------------------------------------------------------------------
        # Mode 2: LSH + Bloom Filter (Production Pipeline)
        # ------------------------------------------------------------------
        print("Running Mode 2: LSH + Bloom Filter (Full Pipeline)...")
        latencies_m2 = []
        hits_m2 = 0
        prec_m2 = 0.0
        rec_m2 = 0.0
        fn_m2 = 0
        cands_m2 = 0

        for q in queries:
            t0 = time.perf_counter()
            # 1. Bloom Filter screening
            if not bloom_mgr.contains(q["identity"]):
                t1 = time.perf_counter()
                latencies_m2.append((t1 - t0) * 1000.0)
                if q["in_db"]:
                    fn_m2 += 1
                continue

            # 2. LSH candidate retrieval
            cands = lsh_index.find_candidates(q["signatures"])
            cands_m2 += len(cands)

            # 3. Candidate scoring
            scores = ScaledLSHIndex.score_candidates_vectorized(
                cands, q["signatures"], gallery_sigs, k=10, use_set_jaccard=False
            )
            t1 = time.perf_counter()
            latencies_m2.append((t1 - t0) * 1000.0)

            if q["in_db"]:
                q_id = q["identity"]
                gt = gt_map.get(q_id, [])
                match_count = sum(1 for c_idx, _ in scores if gallery_ids[c_idx] == q_id)
                if match_count > 0:
                    hits_m2 += 1
                else:
                    fn_m2 += 1
                prec_m2 += match_count / 10.0
                if len(gt) > 0:
                    rec_m2 += match_count / len(gt)

        m2_arr = np.array(latencies_m2)
        m2_sec = sum(latencies_m2) / 1000.0
        m2_ms = float(np.mean(m2_arr))
        m2_qps = len(queries) / m2_sec if m2_sec > 0 else 0.0
        workload_res["2_lsh_and_bloom"] = {
            "name": "LSH + Bloom Filter",
            "avg_latency_ms": round(m2_ms, 3),
            "p50_latency_ms": round(float(np.median(m2_arr)), 3),
            "p99_latency_ms": round(float(np.percentile(m2_arr, 99)), 3),
            "total_sec": round(m2_sec, 3),
            "throughput_qps": round(m2_qps, 1),
            "avg_candidates": round(cands_m2 / len(queries), 1),
            "hit_at_10_pct": round((hits_m2 / eval_count * 100.0), 2) if eval_count > 0 else None,
            "precision_at_10": round(prec_m2 / eval_count, 4) if eval_count > 0 else None,
            "recall_at_10": round(rec_m2 / eval_count, 4) if eval_count > 0 else None,
            "false_negatives": fn_m2 if eval_count > 0 else 0,
            "speedup_vs_minhash": round(m1_ms / m2_ms, 1) if m2_ms > 0 else 0.0
        }
        print(f"  [2] LSH + Bloom Filter               : {m2_ms:8.2f} ms / query ({m2_qps:6.1f} QPS) | Hit@10: {workload_res['2_lsh_and_bloom']['hit_at_10_pct']}% | Speedup: {workload_res['2_lsh_and_bloom']['speedup_vs_minhash']}x")

        # ------------------------------------------------------------------
        # Mode 3: Bloom Filter (No LSH) - Linear scan on positive admission
        # ------------------------------------------------------------------
        print("Running Mode 3: Bloom Filter (No LSH)...")
        latencies_m3 = []
        hits_m3 = 0
        prec_m3 = 0.0
        rec_m3 = 0.0
        fn_m3 = 0
        cands_m3 = 0

        for q in queries:
            t0 = time.perf_counter()
            # 1. Bloom Filter screening
            if not bloom_mgr.contains(q["identity"]):
                t1 = time.perf_counter()
                latencies_m3.append((t1 - t0) * 1000.0)
                if q["in_db"]:
                    fn_m3 += 1
                continue

            # Linear scan fallback
            cands_m3 += num_gallery
            sims = np.mean(gallery_sigs == q["signatures"], axis=1)
            top_indices = np.argpartition(sims, -10)[-10:]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
            t1 = time.perf_counter()
            latencies_m3.append((t1 - t0) * 1000.0)

            if q["in_db"]:
                q_id = q["identity"]
                gt = gt_map.get(q_id, [])
                match_count = sum(1 for idx in top_indices if gallery_ids[idx] == q_id)
                if match_count > 0:
                    hits_m3 += 1
                else:
                    fn_m3 += 1
                prec_m3 += match_count / 10.0
                if len(gt) > 0:
                    rec_m3 += match_count / len(gt)

        m3_arr = np.array(latencies_m3)
        m3_sec = sum(latencies_m3) / 1000.0
        m3_ms = float(np.mean(m3_arr))
        m3_qps = len(queries) / m3_sec if m3_sec > 0 else 0.0
        workload_res["3_bloom_no_lsh"] = {
            "name": "Bloom Filter (No LSH)",
            "avg_latency_ms": round(m3_ms, 3),
            "p50_latency_ms": round(float(np.median(m3_arr)), 3),
            "p99_latency_ms": round(float(np.percentile(m3_arr, 99)), 3),
            "total_sec": round(m3_sec, 3),
            "throughput_qps": round(m3_qps, 1),
            "avg_candidates": round(cands_m3 / len(queries), 1),
            "hit_at_10_pct": round((hits_m3 / eval_count * 100.0), 2) if eval_count > 0 else None,
            "precision_at_10": round(prec_m3 / eval_count, 4) if eval_count > 0 else None,
            "recall_at_10": round(rec_m3 / eval_count, 4) if eval_count > 0 else None,
            "false_negatives": fn_m3 if eval_count > 0 else 0,
            "speedup_vs_minhash": round(m1_ms / m3_ms, 1) if m3_ms > 0 else 0.0
        }
        print(f"  [3] Bloom Filter (No LSH)            : {m3_ms:8.2f} ms / query ({m3_qps:6.1f} QPS) | Hit@10: {workload_res['3_bloom_no_lsh']['hit_at_10_pct']}% | Speedup: {workload_res['3_bloom_no_lsh']['speedup_vs_minhash']}x")

        # ------------------------------------------------------------------
        # Mode 4: LSH (No Bloom Filter)
        # ------------------------------------------------------------------
        print("Running Mode 4: LSH (No Bloom Filter)...")
        latencies_m4 = []
        hits_m4 = 0
        prec_m4 = 0.0
        rec_m4 = 0.0
        fn_m4 = 0
        cands_m4 = 0

        for q in queries:
            t0 = time.perf_counter()
            cands = lsh_index.find_candidates(q["signatures"])
            cands_m4 += len(cands)
            scores = ScaledLSHIndex.score_candidates_vectorized(
                cands, q["signatures"], gallery_sigs, k=10, use_set_jaccard=False
            )
            t1 = time.perf_counter()
            latencies_m4.append((t1 - t0) * 1000.0)

            if q["in_db"]:
                q_id = q["identity"]
                gt = gt_map.get(q_id, [])
                match_count = sum(1 for c_idx, _ in scores if gallery_ids[c_idx] == q_id)
                if match_count > 0:
                    hits_m4 += 1
                else:
                    fn_m4 += 1
                prec_m4 += match_count / 10.0
                if len(gt) > 0:
                    rec_m4 += match_count / len(gt)

        m4_arr = np.array(latencies_m4)
        m4_sec = sum(latencies_m4) / 1000.0
        m4_ms = float(np.mean(m4_arr))
        m4_qps = len(queries) / m4_sec if m4_sec > 0 else 0.0
        workload_res["4_lsh_no_bloom"] = {
            "name": "LSH (No Bloom Filter)",
            "avg_latency_ms": round(m4_ms, 3),
            "p50_latency_ms": round(float(np.median(m4_arr)), 3),
            "p99_latency_ms": round(float(np.percentile(m4_arr, 99)), 3),
            "total_sec": round(m4_sec, 3),
            "throughput_qps": round(m4_qps, 1),
            "avg_candidates": round(cands_m4 / len(queries), 1),
            "hit_at_10_pct": round((hits_m4 / eval_count * 100.0), 2) if eval_count > 0 else None,
            "precision_at_10": round(prec_m4 / eval_count, 4) if eval_count > 0 else None,
            "recall_at_10": round(rec_m4 / eval_count, 4) if eval_count > 0 else None,
            "false_negatives": fn_m4 if eval_count > 0 else 0,
            "speedup_vs_minhash": round(m1_ms / m4_ms, 1) if m4_ms > 0 else 0.0
        }
        print(f"  [4] LSH (No Bloom Filter)            : {m4_ms:8.2f} ms / query ({m4_qps:6.1f} QPS) | Hit@10: {workload_res['4_lsh_no_bloom']['hit_at_10_pct']}% | Speedup: {workload_res['4_lsh_no_bloom']['speedup_vs_minhash']}x")

        all_results["workloads"][workload_name] = workload_res

    # Print Summary Tables
    print("\n" + "=" * 92)
    print("                     4-WAY LATENCY COMPARISON SUMMARY TABLE")
    print("=" * 92)
    print(f"{'Configuration':<35} | {'Positive (In DB)':<16} | {'Negative (Out DB)':<17} | {'Mixed (50/50)':<14}")
    print("-" * 92)

    pos_res = all_results["workloads"]["Positive Queries (In Database)"]
    neg_res = all_results["workloads"]["Negative Queries (Not In Database)"]
    mix_res = all_results["workloads"]["Mixed Workload (50% Positive / 50% Negative)"]

    for k in ["1_only_minhash", "2_lsh_and_bloom", "3_bloom_no_lsh", "4_lsh_no_bloom"]:
        c_name = pos_res[k]["name"]
        p_lat = f"{pos_res[k]['avg_latency_ms']:.2f} ms"
        n_lat = f"{neg_res[k]['avg_latency_ms']:.4f} ms"
        m_lat = f"{mix_res[k]['avg_latency_ms']:.2f} ms"
        print(f"{c_name:<35} | {p_lat:<16} | {n_lat:<17} | {m_lat:<14}")
    print("=" * 92)

    print("\n" + "=" * 92)
    print("                     4-WAY RETRIEVAL METRICS SUMMARY TABLE (POSITIVE WORKLOAD)")
    print("=" * 92)
    print(f"{'Configuration':<35} | {'Hit@10':<10} | {'Recall@10':<12} | {'Precision@10':<14} | {'False Negatives':<15}")
    print("-" * 92)
    for k in ["1_only_minhash", "2_lsh_and_bloom", "3_bloom_no_lsh", "4_lsh_no_bloom"]:
        c_name = pos_res[k]["name"]
        h10 = f"{pos_res[k]['hit_at_10_pct']:.2f}%"
        r10 = f"{pos_res[k]['recall_at_10']:.4f}"
        p10 = f"{pos_res[k]['precision_at_10']:.4f}"
        fn = f"{pos_res[k]['false_negatives']}"
        print(f"{c_name:<35} | {h10:<10} | {r10:<12} | {p10:<14} | {fn:<15}")
    print("=" * 92)

    # Save to JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved raw JSON benchmark results to {out_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 4-way latency and metrics comparison at scale.")
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
        default=16,
        help="Number of LSH bands."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of rows per LSH band."
    )
    parser.add_argument(
        "--max-gallery",
        type=int,
        default=None,
        help="Max gallery records to index (None = all 4,000,000 records)."
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
        rows=args.rows,
        max_gallery_records=args.max_gallery,
        output_path=args.output
    )
