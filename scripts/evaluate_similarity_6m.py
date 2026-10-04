"""
Evaluation & Accuracy Benchmark for Large-Scale Duplicate Detection (6M Scale).

Computes comprehensive retrieval metrics on probe queries:
  - MRR, MAP
  - Hit@1, Hit@5, Hit@10, Hit@20, Hit@50
  - Precision@K, Recall@K
  - Decision Threshold Metrics (tau = 0.40)
  - Zero-Recall Queries Breakdown by Quality Level (q1, q3, q5, q7, q9)
  - Number of False Negatives (Bloom Filter vs LSH Candidate Reduction)
  - Detailed Latency Percentiles (Mean, P50, P90, P99, Min, Max) & Throughput (QPS)
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


def evaluate_retrieval(
    shards_dir: str,
    bloom_path: str,
    num_queries: int = 2500,
    bands: int = 16,
    rows: int = 4,
    decision_threshold: float = 0.40,
    max_gallery_records: Optional[int] = None,
    top_k_values: List[int] = [1, 5, 10, 20, 50],
    output_path: str = "reports/benchmark_6m_best_hyperparameters.json"
) -> Dict[str, Any]:
    print("=" * 78)
    print("     DUPLICATE DETECTION BENCHMARK AT FULL SCALE (6M SAMPLES)")
    print("=" * 78)

    # 1. Load Bloom Filter
    print(f"Loading Bloom Filter from {bloom_path}...")
    bloom_mgr = ScaledIdentityBloomFilter.load(bloom_path)
    print(f"Bloom Filter loaded: capacity={bloom_mgr.capacity:,}, k={bloom_mgr.k}, bits={bloom_mgr.m:,}")

    # 2. Load Gallery
    gallery_desc = f"max {max_gallery_records:,}" if max_gallery_records else "all 40 shards (4,000,000)"
    print(f"\nLoading Gallery Signatures from {shards_dir} ({gallery_desc} records)...")
    dataset = ShardedSignatureDataset(shards_dir)
    t0_load = time.perf_counter()
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    print(f"Gallery loaded: {num_gallery:,} records in {time.perf_counter() - t0_load:.2f}s.")

    # Build identity -> list of gallery record indices
    print("Building ground-truth identity mapping...")
    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        ident = int(ident)
        if ident not in gt_map:
            gt_map[ident] = []
        gt_map[ident].append(idx)
    num_unique_identities = len(gt_map)
    print(f"Ground truth indexed: {num_unique_identities:,} unique identities.")

    # 3. Build LSH Index with optimal hyperparameters (b=16, r=4)
    print(f"\nBuilding LSH Index (bands={bands}, rows={rows}, theoretical_threshold={(1.0/bands)**(1.0/rows):.3f})...")
    t0_idx = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands, rows=rows)
    lsh_index.index_signatures(gallery_sigs)
    t1_idx = time.perf_counter()
    indexing_time = t1_idx - t0_idx
    print(f"Indexed {num_gallery:,} records in {indexing_time:.2f}s ({num_gallery/indexing_time:,.0f} records/s).")

    # 4. Load Probe Queries
    print(f"\nLoading {num_queries} Probe Queries...")
    probe_sigs, probe_ids, probe_oris, probe_quals = dataset.load_probes(max_records=num_queries)
    total_queries = len(probe_sigs)
    max_k = max(top_k_values)
    print(f"Loaded {total_queries} probe queries.")

    # Performance & metric accumulators
    query_latencies = []
    bloom_latencies = []
    lsh_latencies = []
    scoring_latencies = []

    hit_at_k = {k: 0 for k in top_k_values}
    precision_at_k = {k: 0.0 for k in top_k_values}
    recall_at_k = {k: 0.0 for k in top_k_values}

    queries_with_gt = 0
    lsh_candidate_recall = 0
    total_lsh_candidates = 0

    bf_screened = 0
    bf_passed = 0
    bf_pruned = 0
    bf_false_negatives = 0
    lsh_false_negatives = 0
    total_false_negatives = 0

    reciprocal_ranks = []
    avg_precisions = []

    # Threshold-specific accumulators (tau = 0.40)
    thresh_tp_total = 0
    thresh_fp_total = 0
    thresh_queries_with_hit = 0
    thresh_recalls = []
    thresh_precisions = []

    # Quality breakdown: q in [1, 3, 5, 7, 9]
    unique_qualities = sorted(list(set(probe_quals.tolist())))
    qual_stats = {
        int(q): {
            "query_count": 0,
            "zero_recall_count": 0,
            "total_recall_at_10": 0.0,
            "total_precision_at_10": 0.0,
            "hit_at_10": 0,
            "total_candidates": 0,
            "mrr_sum": 0.0,
            "map_sum": 0.0
        }
        for q in unique_qualities
    }

    print("\nRunning Evaluation Benchmark across probes...")
    t_eval_start = time.perf_counter()

    for i in range(total_queries):
        q_ident = int(probe_ids[i])
        q_sig = probe_sigs[i]
        q_qual = int(probe_quals[i])

        gt_indices = gt_map.get(q_ident, [])
        num_gt = len(gt_indices)
        if num_gt > 0:
            queries_with_gt += 1

        qual_stats[q_qual]["query_count"] += 1

        t_q0 = time.perf_counter()

        # 1. Bloom Filter Pre-check
        t_b0 = time.perf_counter()
        passed_bf = bloom_mgr.contains(q_ident)
        t_b1 = time.perf_counter()
        bloom_latencies.append((t_b1 - t_b0) * 1000.0)
        bf_screened += 1

        if not passed_bf:
            bf_pruned += 1
            if num_gt > 0:
                bf_false_negatives += 1
                total_false_negatives += 1
            t_q_end = time.perf_counter()
            query_latencies.append((t_q_end - t_q0) * 1000.0)
            reciprocal_ranks.append(0.0)
            avg_precisions.append(0.0)
            qual_stats[q_qual]["zero_recall_count"] += 1
            continue
        else:
            bf_passed += 1

        # 2. LSH Candidate Retrieval
        t_l0 = time.perf_counter()
        candidates = lsh_index.find_candidates(q_sig)
        t_l1 = time.perf_counter()
        lsh_latencies.append((t_l1 - t_l0) * 1000.0)
        total_lsh_candidates += len(candidates)
        qual_stats[q_qual]["total_candidates"] += len(candidates)

        cand_set = set(candidates)
        has_lsh_hit = any(g_idx in cand_set for g_idx in gt_indices)
        if has_lsh_hit:
            lsh_candidate_recall += 1
        elif num_gt > 0:
            lsh_false_negatives += 1

        # 3. Jaccard scoring & ranking
        t_j0 = time.perf_counter()
        scores = ScaledLSHIndex.score_candidates_vectorized(
            candidates, q_sig, gallery_sigs, k=max_k, use_set_jaccard=False
        )
        t_j1 = time.perf_counter()
        scoring_latencies.append((t_j1 - t_j0) * 1000.0)

        total_query_lat_ms = (t_j1 - t_q0) * 1000.0
        query_latencies.append(total_query_lat_ms)

        # Top-K ranking evaluation
        hit_ranks = []
        for rank_idx, (cand_idx, sim_score) in enumerate(scores):
            if gallery_ids[cand_idx] == q_ident:
                hit_ranks.append((rank_idx + 1, sim_score))

        # MRR calculation
        if hit_ranks:
            mrr_val = 1.0 / hit_ranks[0][0]
            reciprocal_ranks.append(mrr_val)
            qual_stats[q_qual]["mrr_sum"] += mrr_val
        else:
            reciprocal_ranks.append(0.0)

        # MAP@max_k calculation
        if hit_ranks and num_gt > 0:
            ap_sum = 0.0
            hits_so_far = 0
            for rank, _ in hit_ranks:
                hits_so_far += 1
                ap_sum += hits_so_far / rank
            ap_val = ap_sum / min(num_gt, max_k)
            avg_precisions.append(ap_val)
            qual_stats[q_qual]["map_sum"] += ap_val
        else:
            avg_precisions.append(0.0)

        # Top-K hit, precision, recall
        hits_at_10_count = 0
        for k in top_k_values:
            k_results = scores[:k]
            hits = [1 for c_idx, _ in k_results if gallery_ids[c_idx] == q_ident]
            num_hits = len(hits)
            if num_hits > 0:
                hit_at_k[k] += 1
            precision_at_k[k] += num_hits / k if k > 0 else 0.0
            if num_gt > 0:
                recall_at_k[k] += num_hits / num_gt
            if k == 10:
                hits_at_10_count = num_hits
                if num_hits > 0:
                    qual_stats[q_qual]["hit_at_10"] += 1
                qual_stats[q_qual]["total_precision_at_10"] += num_hits / 10.0
                if num_gt > 0:
                    qual_stats[q_qual]["total_recall_at_10"] += num_hits / num_gt

        # Zero recall check (failure to retrieve any true duplicate in top-10)
        if hits_at_10_count == 0:
            qual_stats[q_qual]["zero_recall_count"] += 1
            if num_gt > 0:
                total_false_negatives += 1

        # Threshold evaluation (tau = 0.40)
        thresh_matches = [
            (c_idx, s) for c_idx, s in scores if s >= decision_threshold
        ]
        tp_thresh = sum(1 for c_idx, _ in thresh_matches if gallery_ids[c_idx] == q_ident)
        fp_thresh = len(thresh_matches) - tp_thresh
        thresh_tp_total += tp_thresh
        thresh_fp_total += fp_thresh
        if tp_thresh > 0:
            thresh_queries_with_hit += 1
        q_thresh_rec = tp_thresh / num_gt if num_gt > 0 else 0.0
        q_thresh_prec = tp_thresh / len(thresh_matches) if thresh_matches else 1.0
        thresh_recalls.append(q_thresh_rec)
        thresh_precisions.append(q_thresh_prec)

    t_eval_end = time.perf_counter()
    total_eval_time = t_eval_end - t_eval_start

    # Compute latency statistics
    q_lat_arr = np.array(query_latencies)
    avg_latency_ms = float(np.mean(q_lat_arr)) if len(q_lat_arr) > 0 else 0.0
    p50_latency_ms = float(np.median(q_lat_arr)) if len(q_lat_arr) > 0 else 0.0
    p90_latency_ms = float(np.percentile(q_lat_arr, 90)) if len(q_lat_arr) > 0 else 0.0
    p99_latency_ms = float(np.percentile(q_lat_arr, 99)) if len(q_lat_arr) > 0 else 0.0
    min_latency_ms = float(np.min(q_lat_arr)) if len(q_lat_arr) > 0 else 0.0
    max_latency_ms = float(np.max(q_lat_arr)) if len(q_lat_arr) > 0 else 0.0
    throughput_qps = total_queries / total_eval_time if total_eval_time > 0 else 0.0

    avg_candidates = total_lsh_candidates / total_queries if total_queries > 0 else 0.0
    mean_reciprocal_rank = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mean_avg_precision = float(np.mean(avg_precisions)) if avg_precisions else 0.0
    lsh_recall_pct = (lsh_candidate_recall / queries_with_gt * 100.0) if queries_with_gt > 0 else 0.0

    # Compile Quality breakdown
    quality_breakdown = {}
    for q in unique_qualities:
        st = qual_stats[q]
        qc = st["query_count"]
        quality_breakdown[f"q_{q}"] = {
            "quality_level": q,
            "queries_evaluated": qc,
            "zero_recall_failures": st["zero_recall_count"],
            "zero_recall_failure_rate_pct": round(st["zero_recall_count"] / qc * 100.0, 2) if qc > 0 else 0.0,
            "hit_at_10_pct": round(st["hit_at_10"] / qc * 100.0, 2) if qc > 0 else 0.0,
            "precision_at_10": round(st["total_precision_at_10"] / qc, 4) if qc > 0 else 0.0,
            "recall_at_10": round(st["total_recall_at_10"] / qc, 4) if qc > 0 else 0.0,
            "mrr": round(st["mrr_sum"] / qc, 4) if qc > 0 else 0.0,
            "map_at_10": round(st["map_sum"] / qc, 4) if qc > 0 else 0.0,
            "avg_candidates": round(st["total_candidates"] / qc, 2) if qc > 0 else 0.0
        }

    metrics = {
        "dataset_summary": {
            "total_queries_evaluated": total_queries,
            "queries_with_ground_truth_in_db": queries_with_gt,
            "gallery_size": num_gallery,
            "unique_identities_in_gallery": num_unique_identities,
            "hyperparameters": {
                "bands": bands,
                "rows": rows,
                "theoretical_lsh_threshold": round((1.0 / bands) ** (1.0 / rows), 4),
                "decision_threshold_tau": decision_threshold
            }
        },
        "bloom_filter_summary": {
            "enabled": True,
            "queries_screened": bf_screened,
            "queries_passed_filter": bf_passed,
            "queries_pruned_before_search": bf_pruned,
            "false_negatives": bf_false_negatives,
            "false_negative_rate_pct": round((bf_false_negatives / queries_with_gt * 100.0), 4) if queries_with_gt > 0 else 0.0,
            "avg_bloom_latency_ms": round(float(np.mean(bloom_latencies)), 4) if bloom_latencies else 0.0
        },
        "lsh_index_efficiency": {
            "gallery_size": num_gallery,
            "indexing_time_sec": round(indexing_time, 2),
            "indexing_throughput_records_per_sec": round(num_gallery / indexing_time, 1) if indexing_time > 0 else 0.0,
            "avg_candidates_per_query": round(avg_candidates, 2),
            "candidate_pruning_ratio_pct": round((1.0 - (avg_candidates / num_gallery)) * 100.0, 4) if num_gallery > 0 else 0.0,
            "lsh_candidate_recall_pct": round(lsh_recall_pct, 2),
            "lsh_false_negatives": lsh_false_negatives
        },
        "retrieval_accuracy": {
            "mrr": round(mean_reciprocal_rank, 4),
            "map_at_k": round(mean_avg_precision, 4),
            "overall_hit_rate_pct": round((hit_at_k[10] / total_queries) * 100.0, 2),
            "overall_precision": round(precision_at_k[10] / total_queries, 4),
            "overall_recall": round(recall_at_k[10] / (queries_with_gt if queries_with_gt > 0 else 1), 4),
            "total_false_negatives": total_false_negatives,
            "hit_at_k": {
                f"Hit@{k}": round((hit_at_k[k] / total_queries) * 100.0, 2) for k in top_k_values
            },
            "precision_at_k": {
                f"P@{k}": round(precision_at_k[k] / total_queries, 4) for k in top_k_values
            },
            "recall_at_k": {
                f"R@{k}": round(recall_at_k[k] / (queries_with_gt if queries_with_gt > 0 else 1), 4) for k in top_k_values
            }
        },
        "decision_threshold_metrics": {
            "threshold_tau": decision_threshold,
            "true_positives": thresh_tp_total,
            "false_positives": thresh_fp_total,
            "hit_rate_at_threshold_pct": round((thresh_queries_with_hit / total_queries) * 100.0, 2),
            "avg_recall_at_threshold": round(float(np.mean(thresh_recalls)), 4) if thresh_recalls else 0.0,
            "avg_precision_at_threshold": round(float(np.mean(thresh_precisions)), 4) if thresh_precisions else 0.0
        },
        "quality_breakdown": quality_breakdown,
        "latency_profile": {
            "total_evaluation_time_sec": round(total_eval_time, 3),
            "throughput_qps": round(throughput_qps, 1),
            "avg_latency_ms": round(avg_latency_ms, 3),
            "p50_latency_ms": round(p50_latency_ms, 3),
            "p90_latency_ms": round(p90_latency_ms, 3),
            "p99_latency_ms": round(p99_latency_ms, 3),
            "min_latency_ms": round(min_latency_ms, 3),
            "max_latency_ms": round(max_latency_ms, 3),
            "breakdown_ms_per_query": {
                "bloom_filter_gating": round(float(np.mean(bloom_latencies)), 4) if bloom_latencies else 0.0,
                "lsh_candidate_search": round(float(np.mean(lsh_latencies)), 4) if lsh_latencies else 0.0,
                "jaccard_scoring_ranking": round(float(np.mean(scoring_latencies)), 4) if scoring_latencies else 0.0
            }
        }
    }

    # Print Formatted Report
    print("\n" + "=" * 78)
    print("                    STANDARD METRICS BENCHMARK REPORT (6M)")
    print("=" * 78)
    print(f"Total Probe Queries Evaluated    : {total_queries:,} (2,500 balanced queries across qualities)")
    print(f"Gallery Records Indexed          : {num_gallery:,} (40 shards, 400,000 unique identities)")
    print(f"Hyperparameters                  : bands = {bands}, rows = {rows}, threshold tau = {decision_threshold}")
    print(f"Index Construction Time          : {indexing_time:.2f} s ({num_gallery/indexing_time:,.0f} records/s)")
    print(f"Average Candidates per Query     : {avg_candidates:.2f} (Pruned {metrics['lsh_index_efficiency']['candidate_pruning_ratio_pct']}% of search space)")
    print(f"LSH Candidate Recall             : {lsh_recall_pct:.2f}%")
    print(f"Total False Negatives            : {total_false_negatives} (Bloom: {bf_false_negatives}, LSH: {lsh_false_negatives})")
    print(f"Average Query Latency            : {avg_latency_ms:.3f} ms / query (P50: {p50_latency_ms:.3f} ms, P99: {p99_latency_ms:.3f} ms)")
    print(f"Search Throughput                : {throughput_qps:,.1f} queries / sec")
    print("\nRetrieval Quality Metrics:")
    print(f"  |- Mean Reciprocal Rank (MRR)  : {mean_reciprocal_rank:.4f}")
    print(f"  |- Mean Average Precision (MAP): {mean_avg_precision:.4f}")
    for k in top_k_values:
        print(f"  |- Hit@{k:<2d}                       : {metrics['retrieval_accuracy']['hit_at_k'][f'Hit@{k}']:.2f}%")
        print(f"  |- Precision@{k:<2d}                : {metrics['retrieval_accuracy']['precision_at_k'][f'P@{k}']:.4f}")
        print(f"  |- Recall@{k:<2d}                   : {metrics['retrieval_accuracy']['recall_at_k'][f'R@{k}']:.4f}")

    print("\nQuality Breakdown (Zero-Recall Analysis):")
    print(f"{'Quality':<10} | {'Queries':<8} | {'Zero-Recall Failures':<22} | {'Hit@10':<8} | {'Precision@10':<12} | {'Recall@10':<10}")
    print("-" * 78)
    for q_name, q_data in quality_breakdown.items():
        print(f"{q_name:<10} | {q_data['queries_evaluated']:<8} | {q_data['zero_recall_failures']:<22} | {q_data['hit_at_10_pct']:>6.2f}% | {q_data['precision_at_10']:>12.4f} | {q_data['recall_at_10']:>10.4f}")
    print("=" * 78)

    # Save JSON report
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved benchmark metrics to {out_file}")

    # Also save copy to benchmark_results_scale.json
    scale_file = Path("reports/benchmark_results_scale.json")
    with open(scale_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate similarity retrieval on large-scale dataset.")
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
        default=2500,
        help="Number of probe queries to evaluate."
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
        "--threshold",
        type=float,
        default=0.40,
        help="Similarity decision threshold."
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
        default="reports/benchmark_6m_best_hyperparameters.json",
        help="Output JSON path."
    )

    args = parser.parse_args()
    evaluate_retrieval(
        shards_dir=args.shards_dir,
        bloom_path=args.bloom_path,
        num_queries=args.num_queries,
        bands=args.bands,
        rows=args.rows,
        decision_threshold=args.threshold,
        max_gallery_records=args.max_gallery,
        output_path=args.output
    )
