"""
4-Way Architecture Comparison Benchmark at Scale (6M Data Points).

Changes from previous version:
  - Probe signatures computed AT RUNTIME from raw minutiae.
  - Bloom filter replaced with LSH-band pre-filter in all modes.
  - Scoring uses fraction-of-matching-positions (MinHash estimator) only.
  - Negative queries use randomly generated signatures (not identity IDs).

Modes tested:
  1. Only MinHash (No LSH, No Pre-filter) — brute-force linear scan.
  2. LSH + Band Pre-filter — full pipeline: pre-filter + LSH candidate + scoring.
  3. Band Pre-filter Only (No LSH) — pre-filter then brute-force on positives.
  4. LSH (No Pre-filter) — LSH candidate retrieval without pre-filter gating.

Workloads:
  - Positive: Identities present in gallery (true duplicates exist).
  - Negative: Unseen identities (random signatures never indexed).
  - Mixed:    50% positive / 50% negative.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import h5py
import numpy as np

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from src.lsh_scale.lsh_index_6m import ScaledLSHIndex
from src.lsh_scale.hdf5_signature_extractor import ShardedSignatureDataset
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine


# -----------------------------------------------------------------------
# Runtime probe loader (raw minutiae → compute signature at query time)
# -----------------------------------------------------------------------

def load_positive_probe_minutiae(
    dataset_dir: Path,
    shards_dir: Path,
    num_queries: int,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[int]]:
    """Returns (minutiae_list, identity_ids) for positive probe queries."""
    rng = np.random.default_rng(seed)

    sig_files = sorted(Path(shards_dir).glob("signatures_*.h5"))
    ds_files  = sorted(Path(dataset_dir).glob("dataset_*.h5"))

    # Read probe indices from first shard(s) until we have enough candidates
    candidates: List[Tuple[int, int, int]] = []  # (shard_idx, gi, ident)
    for s_idx, (sf, _) in enumerate(zip(sig_files, ds_files)):
        with h5py.File(sf, "r") as s_f:
            is_train = s_f["metadata/is_train"][:]
            ids      = s_f["metadata/synthetic_identity"][:]
            probe_idx = np.where(~is_train)[0]
            for gi in probe_idx:
                candidates.append((s_idx, int(gi), int(ids[gi])))
        if len(candidates) >= num_queries * 2:
            break

    n = min(num_queries, len(candidates))
    chosen_cands = [candidates[i] for i in rng.choice(len(candidates), size=n, replace=False)]

    # Group by shard to minimize file open/close overhead
    by_shard: Dict[int, List[Tuple[int, int]]] = {}
    for s_idx, gi, ident in chosen_cands:
        by_shard.setdefault(s_idx, []).append((gi, ident))

    all_min: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_ids: List[int] = []
    for s_idx, items in by_shard.items():
        with h5py.File(ds_files[s_idx], "r") as d_f:
            d_min = d_f["data/minutiae"]
            d_num = d_f["data/num_minutiae"]
            for gi, ident in items:
                c      = int(d_num[gi])
                pts    = d_min[gi, :c, :2].copy()
                angles = d_min[gi, :c, 2].copy()
                types  = d_min[gi, :c, 3].astype(np.int32)
                all_min.append((pts, angles, types))
                all_ids.append(ident)

    return all_min, all_ids


# -----------------------------------------------------------------------
# Run a single workload across 4 modes
# -----------------------------------------------------------------------

def run_workload(
    workload_name: str,
    pos_minutiae: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pos_ids: List[int],
    neg_sigs: np.ndarray,        # Pre-generated random signatures for negatives
    workload_type: str,          # "positive" | "negative" | "mixed"
    engine: VectorizedMinHashEngine,
    lsh_index: ScaledLSHIndex,
    gallery_sigs: np.ndarray,
    gallery_ids: np.ndarray,
    gt_map: Dict[int, List[int]],
    num_gallery: int,
    k: int = 10,
) -> Dict[str, Any]:
    """Builds the query list and runs all 4 modes for a workload."""

    # Build query list
    # Each query: (minutiae_or_None, identity, in_db)
    # For negative queries, minutiae=None and a pre-generated random sig is used.
    if workload_type == "positive":
        queries = [(m, pid, True) for m, pid in zip(pos_minutiae, pos_ids)]
    elif workload_type == "negative":
        queries = [(None, -1 - i, False) for i in range(len(pos_minutiae))]
    else:  # mixed
        half = len(pos_minutiae) // 2
        queries = (
            [(m, pid, True) for m, pid in zip(pos_minutiae[:half], pos_ids[:half])] +
            [(None, -1 - i, False) for i in range(half)]
        )

    n_q = len(queries)
    eval_count = sum(1 for _, _, in_db in queries if in_db)

    def score_brute(q_sig):
        sims = np.mean(gallery_sigs == q_sig, axis=1)
        top_idx = np.argpartition(sims, -k)[-k:]
        return top_idx[np.argsort(-sims[top_idx])]

    def metrics_from_top(top_indices_or_scores, q_id, is_scored_list=False):
        if is_scored_list:
            hits = sum(1 for ci, _ in top_indices_or_scores if gallery_ids[ci] == q_id)
        else:
            hits = sum(1 for idx in top_indices_or_scores if gallery_ids[idx] == q_id)
        gt = gt_map.get(q_id, [])
        hit = hits > 0
        prec = hits / k
        rec  = hits / len(gt) if gt else 0.0
        return hit, prec, rec, (0 if hit else 1)

    # -- Helper: get query sig at runtime (compute for positive, random for negative)
    neg_sig_iter = iter(neg_sigs)
    def get_sig(minutiae, in_db):
        if in_db:
            pts, angles, types = minutiae
            return engine.process_observation(pts, angles, types)
        else:
            return next(neg_sig_iter)

    # Reset neg iterator each mode
    def reset_neg():
        nonlocal neg_sig_iter
        neg_sig_iter = iter(neg_sigs)

    results = {}

    # ================================================================
    # Mode 1: Only MinHash (Brute Force)
    # ================================================================
    reset_neg()
    lats1 = []; hits1 = prec1 = rec1 = fn1 = 0
    for minutiae, q_id, in_db in queries:
        t0 = time.perf_counter()
        q_sig = get_sig(minutiae, in_db)
        top_idx = score_brute(q_sig)
        t1 = time.perf_counter()
        lats1.append((t1 - t0) * 1000)
        if in_db:
            h, p, r, f = metrics_from_top(top_idx, q_id)
            hits1 += h; prec1 += p; rec1 += r; fn1 += f
    a1 = np.array(lats1)
    results["1_only_minhash"] = {
        "name": "Only MinHash (No LSH, No Pre-filter)",
        "avg_latency_ms":   round(float(np.mean(a1)), 3),
        "p50_latency_ms":   round(float(np.median(a1)), 3),
        "p99_latency_ms":   round(float(np.percentile(a1, 99)), 3),
        "total_sec":        round(float(np.sum(a1)) / 1000, 3),
        "throughput_qps":   round(n_q / (float(np.sum(a1)) / 1000), 1),
        "avg_candidates":   num_gallery,
        "hit_at_10_pct":    round(hits1 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10":  round(prec1 / eval_count, 4) if eval_count else None,
        "recall_at_10":     round(rec1  / eval_count, 4) if eval_count else None,
        "false_negatives":  fn1,
    }

    # ================================================================
    # Mode 2: LSH + Band Pre-filter (Full Pipeline)
    # ================================================================
    reset_neg()
    lats2 = []; hits2 = prec2 = rec2 = fn2 = cands2 = 0
    for minutiae, q_id, in_db in queries:
        t0 = time.perf_counter()
        q_sig = get_sig(minutiae, in_db)
        if not lsh_index.prefilter(q_sig):
            lats2.append((time.perf_counter() - t0) * 1000)
            if in_db:
                fn2 += 1
            continue
        cands = lsh_index.find_candidates(q_sig)
        cands2 += len(cands)
        scores = ScaledLSHIndex.score_candidates(cands, q_sig, gallery_sigs, k=k)
        lats2.append((time.perf_counter() - t0) * 1000)
        if in_db:
            h, p, r, f = metrics_from_top(scores, q_id, is_scored_list=True)
            hits2 += h; prec2 += p; rec2 += r; fn2 += f
    a2 = np.array(lats2)
    results["2_lsh_and_prefilter"] = {
        "name": "LSH + Band Pre-filter (Full Pipeline)",
        "avg_latency_ms":   round(float(np.mean(a2)), 3),
        "p50_latency_ms":   round(float(np.median(a2)), 3),
        "p99_latency_ms":   round(float(np.percentile(a2, 99)), 3),
        "total_sec":        round(float(np.sum(a2)) / 1000, 3),
        "throughput_qps":   round(n_q / (float(np.sum(a2)) / 1000), 1),
        "avg_candidates":   round(cands2 / n_q, 1),
        "hit_at_10_pct":    round(hits2 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10":  round(prec2 / eval_count, 4) if eval_count else None,
        "recall_at_10":     round(rec2  / eval_count, 4) if eval_count else None,
        "false_negatives":  fn2,
        "speedup_vs_minhash": round(float(np.mean(a1)) / float(np.mean(a2)), 1) if float(np.mean(a2)) > 0 else 0,
    }

    # ================================================================
    # Mode 3: Band Pre-filter Only (No LSH) — brute-force on accepted
    # ================================================================
    reset_neg()
    lats3 = []; hits3 = prec3 = rec3 = fn3 = cands3 = 0
    for minutiae, q_id, in_db in queries:
        t0 = time.perf_counter()
        q_sig = get_sig(minutiae, in_db)
        if not lsh_index.prefilter(q_sig):
            lats3.append((time.perf_counter() - t0) * 1000)
            if in_db:
                fn3 += 1
            continue
        cands3 += num_gallery
        top_idx = score_brute(q_sig)
        lats3.append((time.perf_counter() - t0) * 1000)
        if in_db:
            h, p, r, f = metrics_from_top(top_idx, q_id)
            hits3 += h; prec3 += p; rec3 += r; fn3 += f
    a3 = np.array(lats3)
    results["3_prefilter_no_lsh"] = {
        "name": "Band Pre-filter Only (No LSH, brute-force after accept)",
        "avg_latency_ms":   round(float(np.mean(a3)), 3),
        "p50_latency_ms":   round(float(np.median(a3)), 3),
        "p99_latency_ms":   round(float(np.percentile(a3, 99)), 3),
        "total_sec":        round(float(np.sum(a3)) / 1000, 3),
        "throughput_qps":   round(n_q / (float(np.sum(a3)) / 1000), 1),
        "avg_candidates":   round(cands3 / n_q, 1),
        "hit_at_10_pct":    round(hits3 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10":  round(prec3 / eval_count, 4) if eval_count else None,
        "recall_at_10":     round(rec3  / eval_count, 4) if eval_count else None,
        "false_negatives":  fn3,
        "speedup_vs_minhash": round(float(np.mean(a1)) / float(np.mean(a3)), 1) if float(np.mean(a3)) > 0 else 0,
    }

    # ================================================================
    # Mode 4: LSH Only (No Pre-filter)
    # ================================================================
    reset_neg()
    lats4 = []; hits4 = prec4 = rec4 = fn4 = cands4 = 0
    for minutiae, q_id, in_db in queries:
        t0 = time.perf_counter()
        q_sig = get_sig(minutiae, in_db)
        cands = lsh_index.find_candidates(q_sig)
        cands4 += len(cands)
        scores = ScaledLSHIndex.score_candidates(cands, q_sig, gallery_sigs, k=k)
        lats4.append((time.perf_counter() - t0) * 1000)
        if in_db:
            h, p, r, f = metrics_from_top(scores, q_id, is_scored_list=True)
            hits4 += h; prec4 += p; rec4 += r; fn4 += f
    a4 = np.array(lats4)
    results["4_lsh_no_prefilter"] = {
        "name": "LSH Only (No Pre-filter)",
        "avg_latency_ms":   round(float(np.mean(a4)), 3),
        "p50_latency_ms":   round(float(np.median(a4)), 3),
        "p99_latency_ms":   round(float(np.percentile(a4, 99)), 3),
        "total_sec":        round(float(np.sum(a4)) / 1000, 3),
        "throughput_qps":   round(n_q / (float(np.sum(a4)) / 1000), 1),
        "avg_candidates":   round(cands4 / n_q, 1),
        "hit_at_10_pct":    round(hits4 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10":  round(prec4 / eval_count, 4) if eval_count else None,
        "recall_at_10":     round(rec4  / eval_count, 4) if eval_count else None,
        "false_negatives":  fn4,
        "speedup_vs_minhash": round(float(np.mean(a1)) / float(np.mean(a4)), 1) if float(np.mean(a4)) > 0 else 0,
    }

    # Print summary row
    print(f"\n  {workload_name}")
    for key, r in results.items():
        fn_str = f"FN={r['false_negatives']}" if r['false_negatives'] is not None else ""
        h_str  = f"Hit@10={r['hit_at_10_pct']}%" if r['hit_at_10_pct'] is not None else "Hit@10=N/A"
        spd_str = f"  [{r.get('speedup_vs_minhash','—')}x]" if 'speedup_vs_minhash' in r else ""
        print(f"    [{key}] {r['name']:<48} {r['avg_latency_ms']:>8.3f} ms  {h_str:<14} {fn_str}{spd_str}")

    return results


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def run_latency_comparison(
    dataset_dir: str,
    shards_dir: str,
    num_queries: int = 100,
    bands: int = 16,
    rows: int = 4,
    max_gallery_records: Optional[int] = None,
    output_path: str = "reports/latency_4way_results_scale.json",
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 84)
    print("    4-WAY COMPARISON: RUNTIME PROBE SIGNATURES | LSH-BAND PRE-FILTER")
    print("=" * 84)

    engine = VectorizedMinHashEngine(num_hashes=64, seed=seed)

    # Gallery
    print(f"\nLoading gallery...")
    dataset = ShardedSignatureDataset(shards_dir)
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    print(f"  Gallery: {num_gallery:,} records")

    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        gt_map.setdefault(int(ident), []).append(idx)
    print(f"  Unique identities: {len(gt_map):,}")

    # Build LSH index (band_key_sets built automatically)
    print(f"\nBuilding LSH Index (bands={bands}, rows={rows})...")
    t0 = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands, rows=rows)
    lsh_index.index_signatures(gallery_sigs)
    print(f"  Built in {time.perf_counter()-t0:.2f}s")

    # Load positive probe minutiae
    print(f"\nLoading {num_queries} positive probe minutiae (runtime sig computation)...")
    pos_minutiae, pos_ids = load_positive_probe_minutiae(
        Path(dataset_dir), Path(shards_dir), num_queries, seed=seed
    )
    print(f"  Loaded {len(pos_minutiae)} positive probes")

    # Generate negative query signatures (random, never indexed)
    rng = np.random.default_rng(seed + 99)
    neg_sigs = rng.integers(0, np.iinfo(np.uint32).max, size=(num_queries * 2, 64), dtype=np.uint32)

    # Run workloads
    all_results: Dict[str, Any] = {
        "metadata": {
            "gallery_size":         num_gallery,
            "queries_per_workload": len(pos_minutiae),
            "bands": bands, "rows": rows,
            "theoretical_threshold": round((1.0 / bands) ** (1.0 / rows), 4),
            "probe_signatures":  "computed_at_runtime",
            "prefilter":         "lsh_band_keyset",
            "similarity_metric": "fraction_matching_positions",
            "split":             "randomised_per_identity",
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "workloads": {},
    }

    workload_specs = [
        ("Positive Queries (In Database)",           "positive"),
        ("Negative Queries (Not In Database)",        "negative"),
        ("Mixed Workload (50% Positive / 50% Neg)",  "mixed"),
    ]
    for name, wtype in workload_specs:
        res = run_workload(
            workload_name=name,
            pos_minutiae=pos_minutiae,
            pos_ids=pos_ids,
            neg_sigs=neg_sigs,
            workload_type=wtype,
            engine=engine,
            lsh_index=lsh_index,
            gallery_sigs=gallery_sigs,
            gallery_ids=gallery_ids,
            gt_map=gt_map,
            num_gallery=num_gallery,
        )
        all_results["workloads"][name] = res

    # Summary table
    print("\n" + "=" * 92)
    print("                    LATENCY SUMMARY TABLE")
    print("=" * 92)
    pos = all_results["workloads"]["Positive Queries (In Database)"]
    neg = all_results["workloads"]["Negative Queries (Not In Database)"]
    mix = all_results["workloads"]["Mixed Workload (50% Positive / 50% Neg)"]
    print(f"{'Configuration':<45} | {'Positive':>10} | {'Negative':>10} | {'Mixed':>10}")
    print("-" * 92)
    for key in ["1_only_minhash", "2_lsh_and_prefilter", "3_prefilter_no_lsh", "4_lsh_no_prefilter"]:
        name = pos[key]["name"]
        print(
            f"{name:<45} | {pos[key]['avg_latency_ms']:>9.3f}ms "
            f"| {neg[key]['avg_latency_ms']:>9.3f}ms "
            f"| {mix[key]['avg_latency_ms']:>9.3f}ms"
        )
    print("=" * 92)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved to {out_file}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",  default=str(workspace_root / "dataset"),                       type=str)
    parser.add_argument("--shards-dir",   default=str(workspace_root / "dataset" / "signatures_shards"), type=str)
    parser.add_argument("--num-queries",  default=100,  type=int)
    parser.add_argument("--bands",        default=16,   type=int)
    parser.add_argument("--rows",         default=4,    type=int)
    parser.add_argument("--max-gallery",  default=None, type=int)
    parser.add_argument("--seed",         default=42,   type=int)
    parser.add_argument("--output",       default="reports/latency_4way_results_scale.json", type=str)
    args = parser.parse_args()

    run_latency_comparison(
        dataset_dir=args.dataset_dir,
        shards_dir=args.shards_dir,
        num_queries=args.num_queries,
        bands=args.bands,
        rows=args.rows,
        max_gallery_records=args.max_gallery,
        output_path=args.output,
        seed=args.seed,
    )
