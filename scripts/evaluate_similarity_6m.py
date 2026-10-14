"""
Evaluation & Accuracy Benchmark for Large-Scale Duplicate Detection (6M Scale).

Changes from previous version:
  - Probe signatures are computed AT RUNTIME from raw minutiae (no pre-computed
    probe signatures loaded from HDF5). This includes signature computation time
    in the measured latency, reflecting real deployment cost.
  - Train/test split is randomised per-identity (not hardcoded orientation-based).
  - LSH-band pre-filter replaces the Bloom filter entirely. A query is fast-
    rejected if NONE of its band hashes appear in any gallery bucket (zero FN).
  - Similarity scoring uses ONLY fraction of matching signature positions
    (MinHash theorem estimator). Set-Jaccard is no longer used.

Metrics computed:
  - MRR, MAP
  - Hit@1, Hit@5, Hit@10, Hit@20, Hit@50
  - Precision@K, Recall@K
  - Decision Threshold Metrics (tau)
  - Zero-Recall breakdown by quality level
  - False Negative counts (pre-filter vs LSH candidate stage)
  - Latency percentiles (signature computation, pre-filter, LSH, scoring)
  - Throughput (QPS end-to-end)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import h5py
import numpy as np

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from src.lsh_scale.lsh_index_6m import ScaledLSHIndex
from src.lsh_scale.hdf5_signature_extractor import ShardedSignatureDataset
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine


# -----------------------------------------------------------------------
# Runtime probe loader  (raw minutiae → compute signature at query time)
# -----------------------------------------------------------------------

def load_probe_minutiae_balanced(
    dataset_dir: Path,
    shards_dir: Path,
    num_queries: int = 2500,
    seed: int = 42,
) -> Tuple[
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]],  # minutiae tuples
    List[int],   # identity IDs
    List[int],   # quality levels
]:
    """
    Loads raw minutiae for probe observations so signatures can be computed
    at query time (not pre-loaded).
    Selects balanced samples across all quality levels using metadata first, then
    loads minutiae only for the selected subset (O(num_queries) disk reads instead of O(probes)).
    """
    rng = np.random.default_rng(seed)

    sig_files = sorted(Path(shards_dir).glob("signatures_*.h5"))
    ds_files  = sorted(Path(dataset_dir).glob("dataset_*.h5"))

    # Step 1: Collect candidates by quality from metadata (fast 1D in-memory reads)
    candidates_by_qual: Dict[int, List[Tuple[int, int, int]]] = {}  # qual -> list of (shard_idx, gi, ident)

    print(f"  Scanning metadata from {len(sig_files)} signature shards...")
    for s_idx, sig_path in enumerate(sig_files):
        with h5py.File(sig_path, "r") as sf:
            is_train = sf["metadata/is_train"][:]
            ids      = sf["metadata/synthetic_identity"][:]
            quals    = sf["metadata/quality"][:]

            probe_idx = np.where(~is_train)[0]
            for gi in probe_idx:
                q = int(quals[gi])
                ident = int(ids[gi])
                candidates_by_qual.setdefault(q, []).append((s_idx, int(gi), ident))

    # Step 2: Randomly sample balanced queries per quality
    qualities = sorted(candidates_by_qual.keys())
    samples_per_q = num_queries // len(qualities) if qualities else 0
    selected_per_shard: Dict[int, List[Tuple[int, int, int]]] = {}  # shard_idx -> list of (gi, ident, qual)

    for q in qualities:
        cand_list = candidates_by_qual[q]
        n = min(samples_per_q, len(cand_list))
        chosen_indices = rng.choice(len(cand_list), size=n, replace=False)
        for ci in chosen_indices:
            s_idx, gi, ident = cand_list[ci]
            selected_per_shard.setdefault(s_idx, []).append((gi, ident, q))

    # Step 3: Read minutiae ONLY for selected items
    sel_minutiae = []
    sel_ids = []
    sel_quals = []

    print(f"  Loading raw minutiae for {sum(len(v) for v in selected_per_shard.values()):,} selected probes...")
    for s_idx, items in selected_per_shard.items():
        with h5py.File(ds_files[s_idx], "r") as df:
            d_min = df["data/minutiae"]
            d_num = df["data/num_minutiae"]
            for gi, ident, q in items:
                c = int(d_num[gi])
                pts    = d_min[gi, :c, :2].copy()
                angles = d_min[gi, :c, 2].copy()
                types  = d_min[gi, :c, 3].astype(np.int32)
                sel_minutiae.append((pts, angles, types))
                sel_ids.append(ident)
                sel_quals.append(q)

    # Shuffle the final query set
    perm = rng.permutation(len(sel_ids))
    return [sel_minutiae[i] for i in perm], [sel_ids[i] for i in perm], [sel_quals[i] for i in perm]


# -----------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------

def evaluate_retrieval(
    dataset_dir: str,
    shards_dir: str,
    num_queries: int = 2500,
    bands: int = 16,
    rows: int = 4,
    decision_threshold: float = 0.40,
    max_gallery_records: Optional[int] = None,
    top_k_values: List[int] = [1, 5, 10, 20, 50],
    output_path: str = "reports/benchmark_6m_best_hyperparameters.json",
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 78)
    print("     DUPLICATE DETECTION BENCHMARK AT FULL SCALE (6M SAMPLES)")
    print("     Probe signatures computed at runtime | LSH-band pre-filter")
    print("=" * 78)

    engine = VectorizedMinHashEngine(num_hashes=64, seed=seed)

    # 1. Load Gallery (pre-computed signatures are fine for the gallery)
    gallery_desc = f"max {max_gallery_records:,}" if max_gallery_records else "all shards"
    print(f"\nLoading Gallery Signatures from {shards_dir} ({gallery_desc})...")
    dataset = ShardedSignatureDataset(shards_dir)
    t0_load = time.perf_counter()
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    print(f"Gallery loaded: {num_gallery:,} records in {time.perf_counter() - t0_load:.2f}s.")

    # Ground-truth map
    print("Building ground-truth identity mapping...")
    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        gt_map.setdefault(int(ident), []).append(idx)
    num_unique_identities = len(gt_map)
    print(f"Ground truth: {num_unique_identities:,} unique identities.")

    # 2. Build LSH Index
    print(f"\nBuilding LSH Index (bands={bands}, rows={rows}, t={((1.0/bands)**(1.0/rows)):.3f})...")
    t0_idx = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands, rows=rows)
    lsh_index.index_signatures(gallery_sigs)
    indexing_time = time.perf_counter() - t0_idx
    print(f"Indexed {num_gallery:,} records in {indexing_time:.2f}s "
          f"({num_gallery/indexing_time:,.0f} rec/s).")
    print("Band pre-filter sets built (replaces Bloom filter).")

    # 3. Load probe minutiae for RUNTIME signature computation
    print(f"\nLoading probe minutiae (runtime computation, {num_queries} balanced queries)...")
    probe_minutiae, probe_ids, probe_quals = load_probe_minutiae_balanced(
        dataset_dir=Path(dataset_dir),
        shards_dir=Path(shards_dir),
        num_queries=num_queries,
        seed=seed,
    )
    total_queries = len(probe_minutiae)
    print(f"Loaded {total_queries:,} probe observations.")

    # 4. Accumulators
    max_k = max(top_k_values)

    sig_latencies     = []
    prefilter_latencies = []
    lsh_latencies     = []
    scoring_latencies = []
    query_latencies   = []

    hit_at_k       = {k: 0   for k in top_k_values}
    precision_at_k = {k: 0.0 for k in top_k_values}
    recall_at_k    = {k: 0.0 for k in top_k_values}

    queries_with_gt     = 0
    lsh_candidate_recall = 0
    total_lsh_candidates = 0

    pf_rejected         = 0
    pf_false_negatives  = 0
    lsh_false_negatives = 0
    total_false_negatives = 0

    reciprocal_ranks = []
    avg_precisions   = []

    thresh_tp_total        = 0
    thresh_fp_total        = 0
    thresh_queries_with_hit = 0
    thresh_recalls     = []
    thresh_precisions  = []

    unique_qualities = sorted(set(probe_quals))
    qual_stats = {
        int(q): {
            "query_count": 0, "zero_recall_count": 0,
            "total_recall_at_10": 0.0, "total_precision_at_10": 0.0,
            "hit_at_10": 0, "total_candidates": 0,
            "mrr_sum": 0.0, "map_sum": 0.0,
        }
        for q in unique_qualities
    }

    print(f"\nRunning evaluation ({total_queries:,} queries)...")
    t_eval_start = time.perf_counter()

    for i, ((pts, angles, types), q_ident, q_qual) in enumerate(
        zip(probe_minutiae, probe_ids, probe_quals)
    ):
        t_q0 = time.perf_counter()

        # --- Runtime signature computation ---
        q_sig = engine.process_observation(pts, angles, types)
        t_sig = time.perf_counter()
        sig_latencies.append((t_sig - t_q0) * 1000.0)

        gt_indices = gt_map.get(q_ident, [])
        num_gt = len(gt_indices)
        if num_gt > 0:
            queries_with_gt += 1

        qual_stats[q_qual]["query_count"] += 1

        # --- LSH band pre-filter (replaces Bloom filter) ---
        passed_pf = lsh_index.prefilter(q_sig)
        t_pf = time.perf_counter()
        prefilter_latencies.append((t_pf - t_sig) * 1000.0)

        if not passed_pf:
            pf_rejected += 1
            if num_gt > 0:
                pf_false_negatives += 1
                total_false_negatives += 1
            query_latencies.append((t_pf - t_q0) * 1000.0)
            reciprocal_ranks.append(0.0)
            avg_precisions.append(0.0)
            qual_stats[q_qual]["zero_recall_count"] += 1
            continue

        # --- LSH Candidate Retrieval ---
        candidates = lsh_index.find_candidates(q_sig)
        t_lsh = time.perf_counter()
        lsh_latencies.append((t_lsh - t_pf) * 1000.0)
        total_lsh_candidates += len(candidates)
        qual_stats[q_qual]["total_candidates"] += len(candidates)

        cand_set = set(candidates)
        has_lsh_hit = any(g_idx in cand_set for g_idx in gt_indices)
        if has_lsh_hit:
            lsh_candidate_recall += 1
        elif num_gt > 0:
            lsh_false_negatives += 1

        # --- Scoring: fraction of matching signature positions ---
        scores = ScaledLSHIndex.score_candidates(
            candidates, q_sig, gallery_sigs, k=max_k
        )
        t_score = time.perf_counter()
        scoring_latencies.append((t_score - t_lsh) * 1000.0)
        query_latencies.append((t_score - t_q0) * 1000.0)

        # MRR
        hit_ranks = [
            (rank + 1, s) for rank, (ci, s) in enumerate(scores)
            if gallery_ids[ci] == q_ident
        ]
        if hit_ranks:
            mrr_val = 1.0 / hit_ranks[0][0]
            reciprocal_ranks.append(mrr_val)
            qual_stats[q_qual]["mrr_sum"] += mrr_val
        else:
            reciprocal_ranks.append(0.0)

        # MAP
        if hit_ranks and num_gt > 0:
            hits_so_far, ap_sum = 0, 0.0
            for rank, _ in hit_ranks:
                hits_so_far += 1
                ap_sum += hits_so_far / rank
            ap_val = ap_sum / min(num_gt, max_k)
            avg_precisions.append(ap_val)
            qual_stats[q_qual]["map_sum"] += ap_val
        else:
            avg_precisions.append(0.0)

        # Top-K metrics
        hits_at_10_count = 0
        for k in top_k_values:
            k_results = scores[:k]
            hits_k = sum(1 for ci, _ in k_results if gallery_ids[ci] == q_ident)
            if hits_k > 0:
                hit_at_k[k] += 1
            precision_at_k[k] += hits_k / k if k > 0 else 0.0
            if num_gt > 0:
                recall_at_k[k] += hits_k / num_gt
            if k == 10:
                hits_at_10_count = hits_k
                if hits_k > 0:
                    qual_stats[q_qual]["hit_at_10"] += 1
                qual_stats[q_qual]["total_precision_at_10"] += hits_k / 10.0
                if num_gt > 0:
                    qual_stats[q_qual]["total_recall_at_10"] += hits_k / num_gt

        if hits_at_10_count == 0:
            qual_stats[q_qual]["zero_recall_count"] += 1
            if num_gt > 0:
                total_false_negatives += 1

        # Threshold metrics
        thresh_matches = [(ci, s) for ci, s in scores if s >= decision_threshold]
        tp_t = sum(1 for ci, _ in thresh_matches if gallery_ids[ci] == q_ident)
        fp_t = len(thresh_matches) - tp_t
        thresh_tp_total += tp_t
        thresh_fp_total += fp_t
        if tp_t > 0:
            thresh_queries_with_hit += 1
        thresh_recalls.append(tp_t / num_gt if num_gt > 0 else 0.0)
        thresh_precisions.append(tp_t / len(thresh_matches) if thresh_matches else 1.0)

    t_eval_end = time.perf_counter()
    total_eval_time = t_eval_end - t_eval_start

    # -----------------------------------------------------------------------
    # Compile statistics
    # -----------------------------------------------------------------------
    def arr_stats(arr):
        a = np.array(arr)
        if len(a) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return (float(np.mean(a)), float(np.median(a)),
                float(np.percentile(a, 90)), float(np.percentile(a, 99)),
                float(np.min(a)), float(np.max(a)))

    q_lat = np.array(query_latencies)
    avg_lat, p50_lat, p90_lat, p99_lat, min_lat, max_lat = arr_stats(query_latencies)
    throughput_qps = total_queries / total_eval_time if total_eval_time > 0 else 0.0
    avg_candidates = total_lsh_candidates / total_queries if total_queries > 0 else 0.0
    mrr  = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    mmap = float(np.mean(avg_precisions)) if avg_precisions else 0.0
    lsh_recall_pct = (lsh_candidate_recall / queries_with_gt * 100.0) if queries_with_gt > 0 else 0.0

    quality_breakdown = {}
    for q in unique_qualities:
        st = qual_stats[q]
        qc = st["query_count"]
        quality_breakdown[f"q_{q}"] = {
            "quality_level":              q,
            "queries_evaluated":          qc,
            "zero_recall_failures":       st["zero_recall_count"],
            "zero_recall_failure_rate_pct": round(st["zero_recall_count"] / qc * 100.0, 2) if qc > 0 else 0.0,
            "hit_at_10_pct":              round(st["hit_at_10"] / qc * 100.0, 2) if qc > 0 else 0.0,
            "precision_at_10":            round(st["total_precision_at_10"] / qc, 4) if qc > 0 else 0.0,
            "recall_at_10":               round(st["total_recall_at_10"] / qc, 4) if qc > 0 else 0.0,
            "mrr":                        round(st["mrr_sum"] / qc, 4) if qc > 0 else 0.0,
            "map_at_10":                  round(st["map_sum"] / qc, 4) if qc > 0 else 0.0,
            "avg_candidates":             round(st["total_candidates"] / qc, 2) if qc > 0 else 0.0,
        }

    metrics = {
        "dataset_summary": {
            "total_queries_evaluated":       total_queries,
            "queries_with_ground_truth_in_db": queries_with_gt,
            "gallery_size":                  num_gallery,
            "unique_identities_in_gallery":  num_unique_identities,
            "hyperparameters": {
                "bands":                     bands,
                "rows":                      rows,
                "theoretical_lsh_threshold": round((1.0 / bands) ** (1.0 / rows), 4),
                "decision_threshold_tau":    decision_threshold,
            },
            "split":                  "randomised_per_identity",
            "probe_signatures":       "computed_at_runtime",
            "similarity_metric":      "fraction_matching_positions (MinHash estimator)",
            "prefilter":              "lsh_band_bloom_filter (probabilistic bit array)",
        },
        "bloom_filter_summary": {
            "type":                        "LSH Band Bloom Filter (Probabilistic Bit Array)",
            "queries_screened":            total_queries,
            "queries_rejected_by_bloom":   pf_rejected,
            "bloom_filter_false_negatives": pf_false_negatives,
            "bloom_filter_fn_rate_pct":    round(pf_false_negatives / queries_with_gt * 100.0, 4) if queries_with_gt > 0 else 0.0,
            "avg_bloom_filter_latency_ms": round(float(np.mean(prefilter_latencies)), 4) if prefilter_latencies else 0.0,
        },
        "prefilter_summary": {
            "queries_screened":            total_queries,
            "queries_rejected_by_prefilter": pf_rejected,
            "prefilter_false_negatives":   pf_false_negatives,
            "prefilter_fn_rate_pct":       round(pf_false_negatives / queries_with_gt * 100.0, 4) if queries_with_gt > 0 else 0.0,
            "avg_prefilter_latency_ms":    round(float(np.mean(prefilter_latencies)), 4) if prefilter_latencies else 0.0,
        },
        "lsh_index_efficiency": {
            "gallery_size":                    num_gallery,
            "indexing_time_sec":               round(indexing_time, 2),
            "indexing_throughput_records_per_sec": round(num_gallery / indexing_time, 1) if indexing_time > 0 else 0.0,
            "avg_candidates_per_query":        round(avg_candidates, 2),
            "candidate_pruning_ratio_pct":     round((1.0 - avg_candidates / num_gallery) * 100.0, 4) if num_gallery > 0 else 0.0,
            "lsh_candidate_recall_pct":        round(lsh_recall_pct, 2),
            "lsh_false_negatives":             lsh_false_negatives,
        },
        "retrieval_accuracy": {
            "mrr":                round(mrr, 4),
            "map_at_k":           round(mmap, 4),
            "overall_hit_rate_pct": round(hit_at_k[10] / total_queries * 100.0, 2),
            "overall_precision":  round(precision_at_k[10] / total_queries, 4),
            "overall_recall":     round(recall_at_k[10] / (queries_with_gt if queries_with_gt > 0 else 1), 4),
            "total_false_negatives": total_false_negatives,
            "hit_at_k":           {f"Hit@{k}": round(hit_at_k[k] / total_queries * 100.0, 2) for k in top_k_values},
            "precision_at_k":     {f"P@{k}":   round(precision_at_k[k] / total_queries, 4) for k in top_k_values},
            "recall_at_k":        {f"R@{k}":   round(recall_at_k[k] / (queries_with_gt if queries_with_gt > 0 else 1), 4) for k in top_k_values},
        },
        "decision_threshold_metrics": {
            "threshold_tau":           decision_threshold,
            "true_positives":          thresh_tp_total,
            "false_positives":         thresh_fp_total,
            "hit_rate_at_threshold_pct": round(thresh_queries_with_hit / total_queries * 100.0, 2),
            "avg_recall_at_threshold": round(float(np.mean(thresh_recalls)), 4) if thresh_recalls else 0.0,
            "avg_precision_at_threshold": round(float(np.mean(thresh_precisions)), 4) if thresh_precisions else 0.0,
        },
        "quality_breakdown": quality_breakdown,
        "latency_profile": {
            "total_evaluation_time_sec": round(total_eval_time, 3),
            "throughput_qps":            round(throughput_qps, 1),
            "avg_latency_ms":            round(avg_lat, 3),
            "p50_latency_ms":            round(p50_lat, 3),
            "p90_latency_ms":            round(p90_lat, 3),
            "p99_latency_ms":            round(p99_lat, 3),
            "min_latency_ms":            round(min_lat, 3),
            "max_latency_ms":            round(max_lat, 3),
            "breakdown_ms_per_query": {
                "signature_computation":  round(float(np.mean(sig_latencies)), 4) if sig_latencies else 0.0,
                "lsh_band_prefilter":     round(float(np.mean(prefilter_latencies)), 4) if prefilter_latencies else 0.0,
                "lsh_candidate_search":   round(float(np.mean(lsh_latencies)), 4) if lsh_latencies else 0.0,
                "minhash_scoring":        round(float(np.mean(scoring_latencies)), 4) if scoring_latencies else 0.0,
            },
        },
    }

    # Print report
    print("\n" + "=" * 78)
    print("                 BENCHMARK REPORT — RUNTIME PROBE SIGNATURES")
    print("=" * 78)
    print(f"Total Probe Queries          : {total_queries:,}")
    print(f"Gallery Records              : {num_gallery:,}  ({num_unique_identities:,} unique identities)")
    print(f"Hyperparameters              : bands={bands}, rows={rows}, tau={decision_threshold}")
    print(f"Split                        : randomised per-identity (not orientation-fixed)")
    print(f"Index Build Time             : {indexing_time:.2f}s")
    print(f"Pre-filter Rejections        : {pf_rejected} (FN: {pf_false_negatives})")
    print(f"Avg Candidates per Query     : {avg_candidates:.2f}  (pruned {metrics['lsh_index_efficiency']['candidate_pruning_ratio_pct']}%)")
    print(f"LSH Candidate Recall         : {lsh_recall_pct:.2f}%")
    print(f"Total False Negatives        : {total_false_negatives} (pre-filter: {pf_false_negatives}, LSH: {lsh_false_negatives})")
    print(f"Avg Query Latency            : {avg_lat:.3f} ms  (P99: {p99_lat:.3f} ms)")
    print(f"Throughput                   : {throughput_qps:,.1f} QPS")
    print(f"  Breakdown: sig={float(np.mean(sig_latencies)):.3f}ms | pf={float(np.mean(prefilter_latencies)):.3f}ms | lsh={float(np.mean(lsh_latencies)):.3f}ms | score={float(np.mean(scoring_latencies)):.3f}ms")
    print(f"\nMRR: {mrr:.4f}  MAP: {mmap:.4f}  Hit@10: {hit_at_k[10]/total_queries*100:.2f}%")
    print(f"\nQuality Breakdown:")
    for q_name, qd in quality_breakdown.items():
        print(f"  {q_name}: Hit@10={qd['hit_at_10_pct']:.1f}%  P@10={qd['precision_at_10']:.4f}  R@10={qd['recall_at_10']:.4f}  MRR={qd['mrr']:.4f}  ZeroRecall={qd['zero_recall_failures']}")
    print("=" * 78)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved to {out_file}")
    return metrics


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-scale duplicate detection benchmark.")
    parser.add_argument("--dataset-dir",  default=str(workspace_root / "dataset"),                         type=str)
    parser.add_argument("--shards-dir",   default=str(workspace_root / "dataset" / "signatures_shards"),   type=str)
    parser.add_argument("--num-queries",  default=2500,  type=int)
    parser.add_argument("--bands",        default=16,    type=int)
    parser.add_argument("--rows",         default=4,     type=int)
    parser.add_argument("--threshold",    default=0.40,  type=float)
    parser.add_argument("--max-gallery",  default=None,  type=int)
    parser.add_argument("--seed",         default=42,    type=int)
    parser.add_argument("--output",       default="reports/benchmark_6m_best_hyperparameters.json", type=str)
    args = parser.parse_args()

    evaluate_retrieval(
        dataset_dir=args.dataset_dir,
        shards_dir=args.shards_dir,
        num_queries=args.num_queries,
        bands=args.bands,
        rows=args.rows,
        decision_threshold=args.threshold,
        max_gallery_records=args.max_gallery,
        output_path=args.output,
        seed=args.seed,
    )
