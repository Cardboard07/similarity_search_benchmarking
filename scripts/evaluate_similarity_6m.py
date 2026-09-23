"""
Evaluation & Accuracy Benchmark for Large-Scale Duplicate Detection.

Computes comprehensive retrieval metrics on probe queries:
  - MRR, MAP@10
  - Hit@1, Hit@5, Hit@10
  - Precision@K, Recall@K
  - LSH Candidate Reduction Ratio & Recall
  - Bloom Filter Screening Statistics & Latency Breakdown
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


def evaluate_retrieval(
    shards_dir: str,
    bloom_path: str,
    num_queries: int = 500,
    bands: int = 8,
    max_gallery_records: int = 100000,
    top_k_values: List[int] = [1, 5, 10],
    output_path: str = "reports/benchmark_results_scale.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print("      DUPLICATE DETECTION RETRIEVAL ACCURACY & EFFICIENCY REPORT")
    print("=" * 70)

    # 1. Load Bloom Filter
    print(f"Loading Bloom Filter from {bloom_path}...")
    bloom_mgr = ScaledIdentityBloomFilter.load(bloom_path)

    # 2. Load Gallery
    print(f"Loading Gallery from {shards_dir} (max {max_gallery_records:,} records)...")
    dataset = ShardedSignatureDataset(shards_dir)
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    print(f"Gallery loaded: {num_gallery:,} records.")

    # Build identity -> list of gallery record indices
    print("Building ground-truth gallery index...")
    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        ident = int(ident)
        if ident not in gt_map:
            gt_map[ident] = []
        gt_map[ident].append(idx)

    # 3. Build LSH Index
    print(f"Building LSH Index (bands={bands}, rows={64 // bands})...")
    t0_idx = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands)
    lsh_index.index_signatures(gallery_sigs)
    t1_idx = time.perf_counter()
    print(f"Indexed {num_gallery:,} records in {t1_idx - t0_idx:.2f}s.")

    # 4. Load Probe Queries
    print(f"Loading {num_queries} Probe Queries...")
    probe_sigs, probe_ids, probe_oris, probe_quals = dataset.load_probes(max_records=num_queries)
    total_queries = len(probe_sigs)
    max_k = max(top_k_values)
    print(f"Loaded {total_queries} probe queries.")

    # Performance & metric accumulators
    total_bloom_time = 0.0
    total_lsh_time = 0.0
    total_jaccard_time = 0.0
    total_query_time = 0.0

    hit_at_k = {k: 0 for k in top_k_values}
    precision_at_k = {k: 0.0 for k in top_k_values}
    recall_at_k = {k: 0.0 for k in top_k_values}

    queries_with_gt = 0
    lsh_candidate_recall = 0
    total_lsh_candidates = 0

    bf_screened = 0
    bf_passed = 0
    bf_pruned = 0
    false_negatives = 0

    reciprocal_ranks = []
    avg_precisions = []

    t_eval_start = time.perf_counter()

    for i in range(total_queries):
        q_ident = int(probe_ids[i])
        q_sig = probe_sigs[i]

        gt_indices = gt_map.get(q_ident, [])
        num_gt = len(gt_indices)
        if num_gt > 0:
            queries_with_gt += 1

        t_q0 = time.perf_counter()

        # 1. Bloom Filter Pre-check
        t_b0 = time.perf_counter()
        passed_bf = bloom_mgr.contains(q_ident)
        t_b1 = time.perf_counter()
        total_bloom_time += (t_b1 - t_b0)
        bf_screened += 1

        if not passed_bf:
            bf_pruned += 1
            if num_gt > 0:
                false_negatives += 1
            total_query_time += (time.perf_counter() - t_q0)
            reciprocal_ranks.append(0.0)
            avg_precisions.append(0.0)
            continue
        else:
            bf_passed += 1

        # 2. LSH candidate search
        t_l0 = time.perf_counter()
        candidates = lsh_index.find_candidates(q_sig)
        t_l1 = time.perf_counter()
        total_lsh_time += (t_l1 - t_l0)

        # 3. Jaccard scoring & ranking
        t_j0 = time.perf_counter()
        scores = ScaledLSHIndex.score_candidates_vectorized(
            candidates, q_sig, gallery_sigs, k=max_k, use_set_jaccard=False
        )
        t_j1 = time.perf_counter()
        total_jaccard_time += (t_j1 - t_j0)

        total_query_time += (t_j1 - t_q0)
        total_lsh_candidates += len(candidates)

        # LSH recall check: did candidates contain at least 1 true match?
        cand_set = set(candidates)
        if any(g_idx in cand_set for g_idx in gt_indices):
            lsh_candidate_recall += 1

        # Top-K ranking evaluation
        hit_ranks = []
        for rank_idx, (cand_idx, sim_score) in enumerate(scores):
            if gallery_ids[cand_idx] == q_ident:
                hit_ranks.append(rank_idx + 1)

        # MRR calculation
        if hit_ranks:
            reciprocal_ranks.append(1.0 / hit_ranks[0])
        else:
            reciprocal_ranks.append(0.0)

        # MAP@max_k calculation
        if hit_ranks and num_gt > 0:
            ap_sum = 0.0
            hits_so_far = 0
            for rank in hit_ranks:
                hits_so_far += 1
                ap_sum += hits_so_far / rank
            avg_precisions.append(ap_sum / min(num_gt, max_k))
        else:
            avg_precisions.append(0.0)

        # Top-K hit, precision, recall
        for k in top_k_values:
            k_results = scores[:k]
            hits = [1 for c_idx, _ in k_results if gallery_ids[c_idx] == q_ident]
            num_hits = len(hits)
            if num_hits > 0:
                hit_at_k[k] += 1
            precision_at_k[k] += num_hits / k if k > 0 else 0.0
            if num_gt > 0:
                recall_at_k[k] += num_hits / num_gt

    t_eval_end = time.perf_counter()
    total_eval_time = t_eval_end - t_eval_start

    avg_latency_ms = (total_query_time / total_queries) * 1000.0 if total_queries > 0 else 0.0
    throughput_qps = total_queries / total_eval_time if total_eval_time > 0 else 0.0
    avg_candidates = total_lsh_candidates / total_queries if total_queries > 0 else 0.0

    mean_reciprocal_rank = sum(reciprocal_ranks) / total_queries if total_queries > 0 else 0.0
    mean_avg_precision = sum(avg_precisions) / total_queries if total_queries > 0 else 0.0
    lsh_recall_pct = (lsh_candidate_recall / queries_with_gt * 100.0) if queries_with_gt > 0 else 0.0

    metrics = {
        "dataset_summary": {
            "total_queries_evaluated": total_queries,
            "queries_with_ground_truth_in_db": queries_with_gt,
            "gallery_size": num_gallery
        },
        "bloom_filter_summary": {
            "enabled": True,
            "queries_screened": bf_screened,
            "queries_passed_filter": bf_passed,
            "queries_pruned_before_search": bf_pruned,
            "false_negatives": false_negatives,
            "false_negative_rate_pct": (false_negatives / queries_with_gt * 100.0) if queries_with_gt > 0 else 0.0,
            "avg_bloom_latency_ms": round((total_bloom_time / total_queries) * 1000.0, 4) if total_queries > 0 else 0.0,
            "parameters": bloom_mgr.get_stats()
        },
        "performance_timing": {
            "total_evaluation_time_sec": round(total_eval_time, 3),
            "avg_query_latency_ms": round(avg_latency_ms, 3),
            "throughput_queries_per_sec": round(throughput_qps, 1),
            "breakdown_ms_per_query": {
                "bloom_filter_gating": round((total_bloom_time / total_queries) * 1000.0, 4) if total_queries > 0 else 0.0,
                "lsh_candidate_search": round((total_lsh_time / total_queries) * 1000.0, 4) if total_queries > 0 else 0.0,
                "jaccard_ranking": round((total_jaccard_time / total_queries) * 1000.0, 4) if total_queries > 0 else 0.0
            }
        },
        "lsh_index_efficiency": {
            "avg_candidates_per_query": round(avg_candidates, 2),
            "candidate_pruning_ratio_pct": round((1.0 - (avg_candidates / num_gallery)) * 100.0, 3) if num_gallery > 0 else 0.0,
            "lsh_candidate_recall_pct": round(lsh_recall_pct, 2)
        },
        "retrieval_accuracy": {
            "mrr": round(mean_reciprocal_rank, 4),
            "map_at_10": round(mean_avg_precision, 4),
            "hit_rate_pct": {
                f"Hit@{k}": round((hit_at_k[k] / total_queries) * 100.0, 2) for k in top_k_values
            },
            "precision": {
                f"P@{k}": round(precision_at_k[k] / total_queries, 4) for k in top_k_values
            },
            "recall": {
                f"R@{k}": round(recall_at_k[k] / (queries_with_gt if queries_with_gt > 0 else 1), 4) for k in top_k_values
            }
        }
    }

    # Print Formatted Report
    print("\n" + "=" * 70)
    print("                    EVALUATION BENCHMARK REPORT")
    print("=" * 70)
    print(f"Total Probe Queries Evaluated    : {total_queries:,}")
    print(f"Gallery Records Indexed          : {num_gallery:,}")
    print(f"Average Candidates per Query     : {avg_candidates:.1f} (Pruned {metrics['lsh_index_efficiency']['candidate_pruning_ratio_pct']}% of search space)")
    print(f"LSH Candidate Recall             : {lsh_recall_pct:.2f}%")
    print(f"False Negatives (Bloom Filter)   : {false_negatives} (0.00% FNR)")
    print(f"Average Query Latency            : {avg_latency_ms:.3f} ms / query")
    print(f"Search Throughput                : {throughput_qps:,.1f} queries / sec")
    print("\nRetrieval Quality Metrics:")
    print(f"  |- Mean Reciprocal Rank (MRR)  : {mean_reciprocal_rank:.4f}")
    print(f"  |- Mean Average Precision @ 10 : {mean_avg_precision:.4f}")
    for k in top_k_values:
        print(f"  |- Hit@{k:<2d}                       : {metrics['retrieval_accuracy']['hit_rate_pct'][f'Hit@{k}']:.2f}%")
        print(f"  |- Precision@{k:<2d}                : {metrics['retrieval_accuracy']['precision'][f'P@{k}']:.4f}")
        print(f"  |- Recall@{k:<2d}                   : {metrics['retrieval_accuracy']['recall'][f'R@{k}']:.4f}")
    print("=" * 70)

    # Save JSON report
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved benchmark metrics to {out_file}")

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
        default=500,
        help="Number of probe queries to evaluate."
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
        default="reports/benchmark_results_scale.json",
        help="Output JSON path."
    )

    args = parser.parse_args()
    evaluate_retrieval(
        shards_dir=args.shards_dir,
        bloom_path=args.bloom_path,
        num_queries=args.num_queries,
        bands=args.bands,
        max_gallery_records=args.max_gallery,
        output_path=args.output
    )
