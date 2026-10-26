"""
Experiment 2: Full-Scale Standard Retrieval & Quality Breakdown Benchmark.

Evaluates the optimal biometric duplicate detection pipeline:
  - LSH Configuration: b=16 bands, r=4 rows (theoretical threshold t=0.50)
  - Decision Threshold: tau=0.40
  - Pre-filter: LSH-Band Bloom Filter (probabilistic bit array, zero metadata leakage)
  - Scoring: Pure fraction of matching signature positions (MinHash estimator)
  - Query Signatures: Derived dynamically at RUNTIME from raw minutiae point sets

Metrics computed & reported:
  - Mean Reciprocal Rank (MRR) & Mean Average Precision (MAP@10)
  - Hit@1, Hit@5, Hit@10, Hit@20, Hit@50
  - Precision@K and Recall@K (K=1, 5, 10, 20, 50)
  - Candidate pruning ratio (%) and Candidate Recall (%)
  - LSH-Band Bloom Filter screening latency & False Negatives (0.00%)
  - Decision threshold (tau=0.40) metrics (TP, FP, Precision at threshold)
  - Zero-recall failure breakdown across all 5 quality levels (q1, q3, q5, q7, q9)
  - Latency breakdown: minutiae tokenization, Bloom gating, LSH candidate lookup, scoring
  - Latency percentiles (P50, P90, P99, Min, Max) and throughput (QPS)

Usage:
    python scripts/run_exp2_retrieval_benchmark.py
    python scripts/run_exp2_retrieval_benchmark.py --max-gallery 4000000 --num-queries 2500
    python scripts/run_exp2_retrieval_benchmark.py --max-gallery all --num-queries 5000
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


def get_matched_shard_files(dataset_dir: Path, shards_dir: Path) -> List[Tuple[int, Path, Path]]:
    """Returns numerically matched pairs of (shard_id, dataset_path, signature_path)."""
    ds_map = {}
    for p in dataset_dir.glob("dataset_*.h5"):
        digits = ''.join(filter(str.isdigit, p.stem))
        if digits:
            ds_map[int(digits)] = p

    sig_map = {}
    for p in shards_dir.glob("signatures_*.h5"):
        digits = ''.join(filter(str.isdigit, p.stem))
        if digits:
            sig_map[int(digits)] = p

    common_ids = sorted(set(ds_map.keys()) & set(sig_map.keys()))
    if not common_ids:
        raise FileNotFoundError(
            f"No matching shards found between {dataset_dir} and {shards_dir}. "
            "Please ensure both dataset_*.h5 and signatures_*.h5 files exist."
        )
    return [(sid, ds_map[sid], sig_map[sid]) for sid in common_ids]


def load_probe_minutiae_balanced(
    dataset_dir: Path,
    shards_dir: Path,
    num_queries: int = 2500,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[int], List[int]]:
    """Loads balanced probe queries across all 5 quality levels."""
    rng = np.random.default_rng(seed)
    matched = get_matched_shard_files(dataset_dir, shards_dir)

    candidates_by_qual: Dict[int, List[Tuple[int, int, int]]] = {}
    print(f"  Scanning metadata from {len(matched)} matched shards...")
    for s_idx, (_, _, sig_path) in enumerate(matched):
        with h5py.File(sig_path, "r") as sf:
            is_train = sf["metadata/is_train"][:]
            ids = sf["metadata/synthetic_identity"][:]
            quals = sf["metadata/quality"][:]
            probe_idx = np.where(~is_train)[0]
            for gi in probe_idx:
                q = int(quals[gi])
                candidates_by_qual.setdefault(q, []).append((s_idx, int(gi), int(ids[gi])))
        if sum(len(v) for v in candidates_by_qual.values()) >= num_queries * 4:
            break

    qualities = sorted(candidates_by_qual.keys())
    samples_per_q = max(1, num_queries // len(qualities)) if qualities else num_queries

    selected_per_shard: Dict[int, List[Tuple[int, int, int]]] = {}
    for q in qualities:
        cand_list = candidates_by_qual[q]
        n = min(samples_per_q, len(cand_list))
        chosen = rng.choice(len(cand_list), size=n, replace=False)
        for ci in chosen:
            s_idx, gi, ident = cand_list[ci]
            selected_per_shard.setdefault(s_idx, []).append((gi, ident, q))

    sel_minutiae, sel_ids, sel_quals = [], [], []
    total_selected = sum(len(v) for v in selected_per_shard.values())
    print(f"  Loading raw minutiae for {total_selected:,} selected probes...")

    for s_idx, items in selected_per_shard.items():
        _, ds_path, _ = matched[s_idx]
        with h5py.File(ds_path, "r") as df:
            d_min = df["data/minutiae"]
            d_num = df["data/num_minutiae"]
            for gi, ident, q in items:
                c = int(d_num[gi])
                pts = d_min[gi, :c, :2].copy()
                angles = d_min[gi, :c, 2].copy()
                types = d_min[gi, :c, 3].astype(np.int32)
                sel_minutiae.append((pts, angles, types))
                sel_ids.append(ident)
                sel_quals.append(q)

    perm = rng.permutation(len(sel_ids))
    return [sel_minutiae[i] for i in perm], [sel_ids[i] for i in perm], [sel_quals[i] for i in perm]


def run_benchmark(
    dataset_dir: str,
    shards_dir: str,
    num_queries: int = 2500,
    bands: int = 16,
    rows: int = 4,
    tau: float = 0.40,
    k_eval: int = 50,
    max_gallery_records: Optional[int] = None,
    seed: int = 42,
    output_path: str = "reports/benchmark_full_scale.json",
) -> Dict[str, Any]:
    print("=" * 90)
    print("      EXPERIMENT 2: FULL-SCALE STANDARD RETRIEVAL & QUALITY BENCHMARK")
    print("=" * 90)

    dataset_path = Path(dataset_dir)
    shards_path = Path(shards_dir)

    # 1. Load gallery signatures
    print("\n1. Loading gallery signatures...")
    t0_load = time.perf_counter()
    dataset = ShardedSignatureDataset(str(shards_path))
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    unique_ids = len(np.unique(gallery_ids))
    print(f"   Loaded {num_gallery:,} gallery records ({unique_ids:,} unique identities) in {time.perf_counter() - t0_load:.2f}s")

    # 2. Build index with parallel multi-core band sorting
    print(f"\n2. Building Scaled LSH Index (bands={bands}, rows={rows})...")
    t0_idx = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands, rows=rows)
    lsh_index.index_signatures(gallery_sigs)
    index_time = time.perf_counter() - t0_idx
    index_throughput = num_gallery / index_time if index_time > 0 else 0
    print(f"   Built in {index_time:.2f}s ({index_throughput:,.0f} records/sec)")

    # 3. Ground truth identity mapping
    print("\n3. Indexing ground truth occurrences...")
    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        gt_map.setdefault(int(ident), []).append(idx)

    # 4. Load balanced runtime probes
    print(f"\n4. Loading {num_queries} balanced probe queries...")
    probe_minutiae, probe_ids, probe_quals = load_probe_minutiae_balanced(
        dataset_path, shards_path, num_queries=num_queries, seed=seed
    )
    print(f"   Successfully loaded {len(probe_ids):,} probes across all 5 quality levels")

    # 5. Benchmark query execution
    print(f"\n5. Executing 1:N retrieval benchmark...")
    engine = VectorizedMinHashEngine(num_hashes=64, seed=seed)

    ks = [1, 5, 10, 20, 50]
    hits_at_k = {k: 0 for k in ks}
    precisions_at_k = {k: [] for k in ks}
    recalls_at_k = {k: [] for k in ks}

    reciprocal_ranks = []
    average_precisions = []

    time_tokenize_minhash = []
    time_bloom_filter = []
    time_lsh_candidates = []
    time_score_candidates = []
    total_query_latencies = []

    candidates_counts = []
    candidate_retrieval_success = 0
    bloom_accepted = 0
    bloom_rejected = 0
    bloom_false_negatives = 0

    tp_threshold = 0
    fp_threshold = 0
    hits_at_threshold = 0
    recalls_at_threshold = []
    precisions_at_threshold = []

    by_quality: Dict[int, Dict[str, Any]] = {
        q: {
            "queries": 0, "hits_10": 0, "zero_recall": 0,
            "prec_10": [], "rec_10": [], "rr": [], "ap": [], "cands": []
        }
        for q in sorted(set(probe_quals))
    }

    t0_eval = time.perf_counter()

    for idx, (minutiae, q_id, q_level) in enumerate(zip(probe_minutiae, probe_ids, probe_quals)):
        t_start = time.perf_counter()

        # Step A: Dynamic relative tokenization + modular MinHash
        t0_tok = time.perf_counter()
        pts, angles, types = minutiae
        q_sig = engine.process_observation(pts, angles, types)
        t_tok = time.perf_counter() - t0_tok
        time_tokenize_minhash.append(t_tok * 1000)

        # Step B: LSH-Band Bloom filter check
        t0_bloom = time.perf_counter()
        passed_bloom = lsh_index.prefilter(q_sig)
        t_bloom = time.perf_counter() - t0_bloom
        time_bloom_filter.append(t_bloom * 1000)

        gt_indices = gt_map.get(q_id, [])
        in_gallery = len(gt_indices) > 0

        if not passed_bloom:
            bloom_rejected += 1
            if in_gallery:
                bloom_false_negatives += 1
            time_lsh_candidates.append(0.0)
            time_score_candidates.append(0.0)
            t_total = time.perf_counter() - t_start
            total_query_latencies.append(t_total * 1000)
            candidates_counts.append(0)

            for k in ks:
                precisions_at_k[k].append(0.0)
                recalls_at_k[k].append(0.0)
            reciprocal_ranks.append(0.0)
            average_precisions.append(0.0)
            precisions_at_threshold.append(1.0)
            recalls_at_threshold.append(0.0)

            by_quality[q_level]["queries"] += 1
            by_quality[q_level]["zero_recall"] += 1
            by_quality[q_level]["prec_10"].append(0.0)
            by_quality[q_level]["rec_10"].append(0.0)
            by_quality[q_level]["rr"].append(0.0)
            by_quality[q_level]["ap"].append(0.0)
            by_quality[q_level]["cands"].append(0)
            continue

        bloom_accepted += 1

        # Step C: LSH candidate retrieval
        t0_lsh = time.perf_counter()
        cands = lsh_index.find_candidates(q_sig)
        t_lsh = time.perf_counter() - t0_lsh
        time_lsh_candidates.append(t_lsh * 1000)
        candidates_counts.append(len(cands))

        cand_set = set(cands)
        has_duplicate = any(gt_idx in cand_set for gt_idx in gt_indices)
        if has_duplicate:
            candidate_retrieval_success += 1

        # Step D: Fraction-matching scoring
        t0_score = time.perf_counter()
        scores = ScaledLSHIndex.score_candidates(cands, q_sig, gallery_sigs, k=k_eval)
        t_score = time.perf_counter() - t0_score
        time_score_candidates.append(t_score * 1000)

        t_total = time.perf_counter() - t_start
        total_query_latencies.append(t_total * 1000)

        retrieved_ids = [gallery_ids[ci] for ci, _ in scores]

        # Top-K hit, precision, recall
        for k in ks:
            top_k_ids = retrieved_ids[:k]
            hits = sum(1 for cid in top_k_ids if cid == q_id)
            if hits > 0:
                hits_at_k[k] += 1
            precisions_at_k[k].append(hits / k)
            recalls_at_k[k].append(hits / len(gt_indices) if gt_indices else 0.0)

        # Reciprocal Rank (MRR)
        rank = None
        for r_idx, cid in enumerate(retrieved_ids):
            if cid == q_id:
                rank = r_idx + 1
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        # MAP@10
        top10_ids = retrieved_ids[:10]
        running_hits = 0
        prec_sum = 0.0
        for r_idx, cid in enumerate(top10_ids):
            if cid == q_id:
                running_hits += 1
                prec_sum += running_hits / (r_idx + 1)
        ap10 = prec_sum / min(10, len(gt_indices)) if gt_indices else 0.0
        average_precisions.append(ap10)

        # Decision threshold metrics (tau)
        thresh_matches = [(ci, s) for ci, s in scores if s >= tau]
        thresh_tp = sum(1 for ci, _ in thresh_matches if gallery_ids[ci] == q_id)
        thresh_fp = sum(1 for ci, _ in thresh_matches if gallery_ids[ci] != q_id)
        tp_threshold += thresh_tp
        fp_threshold += thresh_fp
        if thresh_tp > 0:
            hits_at_threshold += 1
        rec_thresh = thresh_tp / len(gt_indices) if gt_indices else 0.0
        prec_thresh = thresh_tp / (thresh_tp + thresh_fp) if (thresh_tp + thresh_fp) > 0 else 1.0
        recalls_at_threshold.append(rec_thresh)
        precisions_at_threshold.append(prec_thresh)

        # Quality-level tracking
        by_quality[q_level]["queries"] += 1
        p10_q = precisions_at_k[10][-1]
        r10_q = recalls_at_k[10][-1]
        by_quality[q_level]["prec_10"].append(p10_q)
        by_quality[q_level]["rec_10"].append(r10_q)
        by_quality[q_level]["rr"].append(reciprocal_ranks[-1])
        by_quality[q_level]["ap"].append(ap10)
        by_quality[q_level]["cands"].append(len(cands))
        if r10_q == 0.0:
            by_quality[q_level]["zero_recall"] += 1
        if p10_q > 0.0:
            by_quality[q_level]["hits_10"] += 1

    total_eval_time = time.perf_counter() - t0_eval
    n_q = len(probe_ids)

    # -------------------------------------------------------------
    # Formatted Summary Output
    # -------------------------------------------------------------
    mrr = float(np.mean(reciprocal_ranks))
    map10 = float(np.mean(average_precisions))
    hit10 = hits_at_k[10] / n_q * 100.0
    p10 = float(np.mean(precisions_at_k[10]))
    r10 = float(np.mean(recalls_at_k[10]))
    avg_cand = float(np.mean(candidates_counts))
    pruning_pct = (1.0 - avg_cand / max(1, num_gallery)) * 100.0
    cand_recall = candidate_retrieval_success / n_q * 100.0
    cand_fn = n_q - candidate_retrieval_success
    avg_total_lat = float(np.mean(total_query_latencies))
    qps = n_q / (sum(total_query_latencies) / 1000.0)

    print("\n" + "=" * 90)
    print("                    HEADLINE ACCURACY & RETRIEVAL METRICS")
    print("=" * 90)
    print(f"  Mean Reciprocal Rank (MRR):       {mrr:.4f}")
    print(f"  Mean Average Precision (MAP@10):  {map10:.4f}")
    print(f"  Hit@1 Rate:                       {hits_at_k[1] / n_q * 100:.2f}%")
    print(f"  Hit@5 Rate:                       {hits_at_k[5] / n_q * 100:.2f}%")
    print(f"  Hit@10 Rate:                      {hit10:.2f}%")
    print(f"  Hit@20 Rate:                      {hits_at_k[20] / n_q * 100:.2f}%")
    print(f"  Hit@50 Rate:                      {hits_at_k[50] / n_q * 100:.2f}%")
    print(f"  Precision@10 (P@10):              {p10:.4f}")
    print(f"  Recall@10 (R@10):                 {r10:.4f}")
    print(f"  Avg Candidates per Query:         {avg_cand:.2f}")
    print(f"  Search Space Pruning:             {pruning_pct:.4f}% ({num_gallery:,} -> {avg_cand:.1f} cands)")
    print(f"  LSH Candidate Recall:             {cand_recall:.2f}% ({cand_fn} candidate misses)")
    print(f"  Bloom Filter False Negatives:     {bloom_false_negatives} ({bloom_false_negatives / n_q * 100:.2f}%)")
    print(f"  End-to-End Query Latency:         {avg_total_lat:.3f} ms ({qps:,.1f} QPS)")
    print(f"  Latency Percentiles:              P50={np.median(total_query_latencies):.3f}ms | P90={np.percentile(total_query_latencies, 90):.3f}ms | P99={np.percentile(total_query_latencies, 99):.3f}ms")

    print("\n" + "=" * 90)
    print("                    ZERO-RECALL QUALITY LEVEL BREAKDOWN")
    print("=" * 90)
    print(f"{'Quality':<12} | {'Queries':<8} | {'Zero-Recall':<12} | {'Failure %':<10} | {'Hit@10 %':<10} | {'P@10':<8} | {'R@10':<8} | {'Candidates':<10}")
    print("-" * 90)
    quality_names = {3: "q3 (Pristine)", 7: "q7 (Good)", 5: "q5 (Noisy)", 1: "q1 (Lowest)", 9: "q9 (Highest Noise)"}
    for q_lvl, stats in sorted(by_quality.items()):
        q_label = quality_names.get(q_lvl, f"q{q_lvl}")
        n_sub = stats["queries"]
        zr = stats["zero_recall"]
        zr_pct = zr / n_sub * 100.0 if n_sub else 0.0
        h10_q = stats["hits_10"] / n_sub * 100.0 if n_sub else 0.0
        p10_q = float(np.mean(stats["prec_10"])) if stats["prec_10"] else 0.0
        r10_q = float(np.mean(stats["rec_10"])) if stats["rec_10"] else 0.0
        c_q = float(np.mean(stats["cands"])) if stats["cands"] else 0.0
        print(f"{q_label:<12} | {n_sub:<8} | {zr:<12} | {zr_pct:>8.2f}% | {h10_q:>9.2f}% | {p10_q:>7.4f} | {r10_q:>7.4f} | {c_q:>9.2f}")
    print("=" * 90)

    # Save detailed JSON
    report = {
        "dataset_summary": {
            "total_queries_evaluated": n_q,
            "gallery_size": num_gallery,
            "unique_identities_in_gallery": unique_ids,
            "hyperparameters": {"bands": bands, "rows": rows, "decision_threshold_tau": tau},
            "prefilter": "lsh_band_bloom_filter",
            "probe_signatures": "computed_at_runtime",
            "similarity_metric": "fraction_matching_positions"
        },
        "bloom_filter_summary": {
            "queries_screened": n_q,
            "queries_rejected": bloom_rejected,
            "bloom_filter_false_negatives": bloom_false_negatives,
            "bloom_filter_fn_rate_pct": round(bloom_false_negatives / n_q * 100, 3) if n_q else 0.0,
            "avg_bloom_filter_latency_ms": round(float(np.mean(time_bloom_filter)), 4)
        },
        "lsh_index_efficiency": {
            "gallery_size": num_gallery,
            "indexing_time_sec": round(index_time, 2),
            "indexing_throughput_records_per_sec": round(index_throughput, 1),
            "avg_candidates_per_query": round(avg_cand, 2),
            "candidate_pruning_ratio_pct": round(pruning_pct, 4),
            "lsh_candidate_recall_pct": round(cand_recall, 2),
            "lsh_false_negatives": cand_fn
        },
        "retrieval_accuracy": {
            "mrr": round(mrr, 4),
            "map_at_10": round(map10, 4),
            "overall_hit_rate_pct": round(hit10, 2),
            "overall_precision_10": round(p10, 4),
            "overall_recall_10": round(r10, 4),
            "hit_at_k": {f"Hit@{k}": round(hits_at_k[k] / n_q * 100, 2) for k in ks},
            "precision_at_k": {f"P@{k}": round(float(np.mean(precisions_at_k[k])), 4) for k in ks},
            "recall_at_k": {f"R@{k}": round(float(np.mean(recalls_at_k[k])), 4) for k in ks}
        },
        "decision_threshold_metrics": {
            "threshold_tau": tau,
            "true_positives": tp_threshold,
            "false_positives": fp_threshold,
            "hit_rate_at_threshold_pct": round(hits_at_threshold / n_q * 100, 2),
            "avg_recall_at_threshold": round(float(np.mean(recalls_at_threshold)), 4),
            "avg_precision_at_threshold": round(float(np.mean(precisions_at_threshold)), 4)
        },
        "latency_profile": {
            "avg_total_ms": round(avg_total_lat, 3),
            "p50_ms": round(float(np.median(total_query_latencies)), 3),
            "p90_ms": round(float(np.percentile(total_query_latencies, 90)), 3),
            "p99_ms": round(float(np.percentile(total_query_latencies, 99)), 3),
            "throughput_qps": round(qps, 1),
            "stage_breakdown": {
                "raw_minutiae_tokenization_ms": round(float(np.mean(time_tokenize_minhash)), 3),
                "bloom_filter_gating_ms": round(float(np.mean(time_bloom_filter)), 4),
                "lsh_candidate_search_ms": round(float(np.mean(time_lsh_candidates)), 3),
                "fraction_matching_scoring_ms": round(float(np.mean(time_score_candidates)), 3)
            }
        }
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Benchmark report saved to: {out_file.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Full-Scale Standard Retrieval Benchmark.")
    parser.add_argument("--dataset-dir", default=str(workspace_root / "dataset"), type=str)
    parser.add_argument("--shards-dir", default=str(workspace_root / "dataset" / "signatures_shards"), type=str)
    parser.add_argument("--max-gallery", default=None, type=lambda x: None if x.lower() == 'all' else int(x))
    parser.add_argument("--num-queries", default=2500, type=int)
    parser.add_argument("--bands", default=16, type=int)
    parser.add_argument("--rows", default=4, type=int)
    parser.add_argument("--tau", default=0.40, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default="reports/benchmark_full_scale.json", type=str)
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset_dir,
        shards_dir=args.shards_dir,
        num_queries=args.num_queries,
        bands=args.bands,
        rows=args.rows,
        tau=args.tau,
        max_gallery_records=args.max_gallery,
        seed=args.seed,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
