#!/usr/bin/env python3
"""
Experiment 3: 4-Way Architectural Latency & Throughput Benchmark.

Evaluates 4 architectures across Positive, Negative, and Mixed query workloads
at arbitrary scale (e.g. 6M to 100M+):
  1. Only MinHash (Brute Force Linear Scan)
  2. LSH + Band Bloom Filter (Full Hybrid Pipeline)
  3. Band Bloom Filter Only (No LSH Candidate Filtering)
  4. LSH Only (No Bloom Filter Gating)

Workloads evaluated:
  - Positive: Probe samples of genuine identities in the gallery.
  - Negative: Random un-indexed signatures simulating unseen impostors/non-duplicates.
  - Mixed:    50% genuine probes, 50% unseen impostor queries.

Key Features:
  - Probes computed AT RUNTIME from raw minutiae via VectorizedMinHashEngine.
  - Probabilistic LSH-Band Bloom Filter with Kirsch-Mitzenmacher double-hashing.
  - Pure fraction of matching signature positions (MinHash estimator).
  - High-performance, zero-duplicate RAM pre-allocation.
  - Comprehensive metrics: Mean, P50, P95, P99 latencies, QPS, speedups, and Hit@K.
  - Automatic JSON export to reports/latency_4way_results.json.
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


def load_positive_probe_minutiae(
    dataset_dir: Path,
    shards_dir: Path,
    num_queries: int = 100,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[int]]:
    """Loads positive probe minutiae for runtime MinHash signature generation."""
    rng = np.random.default_rng(seed)
    matched = get_matched_shard_files(dataset_dir, shards_dir)

    candidates: List[Tuple[int, int, int]] = []
    for s_idx, (_, _, sig_path) in enumerate(matched):
        with h5py.File(sig_path, "r") as sf:
            is_train = sf["metadata/is_train"][:]
            ids = sf["metadata/synthetic_identity"][:]
            probe_idx = np.where(~is_train)[0]
            for gi in probe_idx:
                candidates.append((s_idx, int(gi), int(ids[gi])))
        if len(candidates) >= num_queries * 4:
            break

    if not candidates:
        raise RuntimeError("No probe records (is_train == False) found in signature shards.")

    n = min(num_queries, len(candidates))
    selected = [candidates[i] for i in rng.choice(len(candidates), size=n, replace=False)]

    by_shard: Dict[int, List[Tuple[int, int]]] = {}
    for s_idx, gi, ident in selected:
        by_shard.setdefault(s_idx, []).append((gi, ident))

    all_minutiae: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_ids: List[int] = []

    for s_idx, items in by_shard.items():
        _, ds_path, _ = matched[s_idx]
        with h5py.File(ds_path, "r") as df:
            d_min = df["data/minutiae"]
            d_num = df["data/num_minutiae"]
            for gi, ident in items:
                c = int(d_num[gi])
                pts = d_min[gi, :c, :2].copy()
                angles = d_min[gi, :c, 2].copy()
                types = d_min[gi, :c, 3].astype(np.int32)
                all_minutiae.append((pts, angles, types))
                all_ids.append(ident)

    return all_minutiae, all_ids


def run_workload(
    workload_name: str,
    pos_minutiae: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pos_ids: List[int],
    neg_sigs: np.ndarray,
    workload_type: str,
    engine: VectorizedMinHashEngine,
    lsh_index: ScaledLSHIndex,
    gallery_sigs: np.ndarray,
    gallery_ids: np.ndarray,
    gt_map: Dict[int, List[int]],
    num_gallery: int,
    k: int = 10,
) -> Dict[str, Any]:
    """Evaluates 4 architectures on the given workload."""
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

    def score_brute(q_sig: np.ndarray) -> np.ndarray:
        sims = np.mean(gallery_sigs == q_sig, axis=1)
        if len(sims) <= k:
            return np.argsort(-sims)
        top_part = np.argpartition(sims, -k)[-k:]
        return top_part[np.argsort(-sims[top_part])]

    def metrics_from_top(top_indices_or_scores: Any, q_id: int, is_scored_list: bool = False):
        if is_scored_list:
            hits = sum(1 for ci, _ in top_indices_or_scores if gallery_ids[ci] == q_id)
        else:
            hits = sum(1 for idx in top_indices_or_scores if gallery_ids[idx] == q_id)
        gt = gt_map.get(q_id, [])
        hit = hits > 0
        prec = hits / k
        rec = hits / len(gt) if gt else 0.0
        return hit, prec, rec, (0 if hit else 1)

    neg_sig_iter = iter(neg_sigs)

    def get_sig(minutiae: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], in_db: bool) -> np.ndarray:
        if in_db and minutiae is not None:
            pts, angles, types = minutiae
            return engine.process_observation(pts, angles, types)
        else:
            return next(neg_sig_iter)

    def reset_neg():
        nonlocal neg_sig_iter
        neg_sig_iter = iter(neg_sigs)

    results: Dict[str, Any] = {}

    # ================================================================
    # Mode 1: Only MinHash (Brute Force Linear Scan)
    # ================================================================
    reset_neg()
    lats1: List[float] = []
    hits1 = prec1 = rec1 = fn1 = 0
    for minutiae, q_id, in_db in queries:
        t0 = time.perf_counter()
        q_sig = get_sig(minutiae, in_db)
        top_idx = score_brute(q_sig)
        lats1.append((time.perf_counter() - t0) * 1000)
        if in_db:
            h, p, r, f = metrics_from_top(top_idx, q_id)
            hits1 += h; prec1 += p; rec1 += r; fn1 += f
    a1 = np.array(lats1)
    results["1_only_minhash"] = {
        "name": "Only MinHash (Brute Force Scan)",
        "avg_latency_ms": round(float(np.mean(a1)), 3),
        "p50_latency_ms": round(float(np.median(a1)), 3),
        "p95_latency_ms": round(float(np.percentile(a1, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(a1, 99)), 3),
        "total_sec": round(float(np.sum(a1)) / 1000, 3),
        "throughput_qps": round(n_q / (float(np.sum(a1)) / 1000 + 1e-9), 1),
        "avg_candidates": num_gallery,
        "hit_at_10_pct": round(hits1 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10": round(prec1 / eval_count, 4) if eval_count else None,
        "recall_at_10": round(rec1 / eval_count, 4) if eval_count else None,
        "false_negatives": fn1,
        "bloom_pruned_pct": 0.0,
    }

    # ================================================================
    # Mode 2: LSH + Band Bloom Filter (Full Pipeline)
    # ================================================================
    reset_neg()
    lats2: List[float] = []
    hits2 = prec2 = rec2 = fn2 = cands2 = pruned2 = 0
    for minutiae, q_id, in_db in queries:
        t0 = time.perf_counter()
        q_sig = get_sig(minutiae, in_db)
        if not lsh_index.prefilter(q_sig):
            lats2.append((time.perf_counter() - t0) * 1000)
            pruned2 += 1
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
    mean_a1 = float(np.mean(a1))
    mean_a2 = float(np.mean(a2))
    results["2_lsh_and_prefilter"] = {
        "name": "LSH + Band Bloom Filter (Full Pipeline)",
        "avg_latency_ms": round(mean_a2, 3),
        "p50_latency_ms": round(float(np.median(a2)), 3),
        "p95_latency_ms": round(float(np.percentile(a2, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(a2, 99)), 3),
        "total_sec": round(float(np.sum(a2)) / 1000, 3),
        "throughput_qps": round(n_q / (float(np.sum(a2)) / 1000 + 1e-9), 1),
        "avg_candidates": round(cands2 / n_q, 1),
        "hit_at_10_pct": round(hits2 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10": round(prec2 / eval_count, 4) if eval_count else None,
        "recall_at_10": round(rec2 / eval_count, 4) if eval_count else None,
        "false_negatives": fn2,
        "bloom_pruned_pct": round(pruned2 / n_q * 100, 1),
        "speedup_vs_minhash": round(mean_a1 / mean_a2, 1) if mean_a2 > 0 else 0,
    }

    # ================================================================
    # Mode 3: Band Bloom Filter Only (No LSH Candidate Filtering)
    # ================================================================
    reset_neg()
    lats3: List[float] = []
    hits3 = prec3 = rec3 = fn3 = cands3 = pruned3 = 0
    for minutiae, q_id, in_db in queries:
        t0 = time.perf_counter()
        q_sig = get_sig(minutiae, in_db)
        if not lsh_index.prefilter(q_sig):
            lats3.append((time.perf_counter() - t0) * 1000)
            pruned3 += 1
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
    mean_a3 = float(np.mean(a3))
    results["3_prefilter_no_lsh"] = {
        "name": "Band Bloom Filter Only (Scan on Accept)",
        "avg_latency_ms": round(mean_a3, 3),
        "p50_latency_ms": round(float(np.median(a3)), 3),
        "p95_latency_ms": round(float(np.percentile(a3, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(a3, 99)), 3),
        "total_sec": round(float(np.sum(a3)) / 1000, 3),
        "throughput_qps": round(n_q / (float(np.sum(a3)) / 1000 + 1e-9), 1),
        "avg_candidates": round(cands3 / n_q, 1),
        "hit_at_10_pct": round(hits3 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10": round(prec3 / eval_count, 4) if eval_count else None,
        "recall_at_10": round(rec3 / eval_count, 4) if eval_count else None,
        "false_negatives": fn3,
        "bloom_pruned_pct": round(pruned3 / n_q * 100, 1),
        "speedup_vs_minhash": round(mean_a1 / mean_a3, 1) if mean_a3 > 0 else 0,
    }

    # ================================================================
    # Mode 4: LSH Only (No Bloom Filter Gating)
    # ================================================================
    reset_neg()
    lats4: List[float] = []
    hits4 = prec4 = rec4 = fn4 = cands4 = 0
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
    mean_a4 = float(np.mean(a4))
    results["4_lsh_no_prefilter"] = {
        "name": "LSH Only (No Bloom Gating)",
        "avg_latency_ms": round(mean_a4, 3),
        "p50_latency_ms": round(float(np.median(a4)), 3),
        "p95_latency_ms": round(float(np.percentile(a4, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(a4, 99)), 3),
        "total_sec": round(float(np.sum(a4)) / 1000, 3),
        "throughput_qps": round(n_q / (float(np.sum(a4)) / 1000 + 1e-9), 1),
        "avg_candidates": round(cands4 / n_q, 1),
        "hit_at_10_pct": round(hits4 / eval_count * 100, 2) if eval_count else None,
        "precision_at_10": round(prec4 / eval_count, 4) if eval_count else None,
        "recall_at_10": round(rec4 / eval_count, 4) if eval_count else None,
        "false_negatives": fn4,
        "bloom_pruned_pct": 0.0,
        "speedup_vs_minhash": round(mean_a1 / mean_a4, 1) if mean_a4 > 0 else 0,
    }

    # Print summary block
    print(f"\n  === Workload: {workload_name} ({n_q} queries) ===")
    print(f"    {'Architecture':<42} | {'Mean Latency':>12} | {'P50':>9} | {'P99':>9} | {'QPS':>7} | {'Hit@10':>8} | {'Speedup':>7}")
    print("    " + "-" * 105)
    for key in ["1_only_minhash", "2_lsh_and_prefilter", "3_prefilter_no_lsh", "4_lsh_no_prefilter"]:
        r = results[key]
        h_str = f"{r['hit_at_10_pct']}%" if r['hit_at_10_pct'] is not None else "N/A"
        spd = f"{r.get('speedup_vs_minhash', 1.0):.1f}x"
        print(f"    {r['name']:<42} | {r['avg_latency_ms']:>10.3f} ms | {r['p50_latency_ms']:>7.3f}ms | {r['p99_latency_ms']:>7.3f}ms | {r['throughput_qps']:>7.1f} | {h_str:>8} | {spd:>7}")

    return results


def run_latency_comparison(
    dataset_dir: str,
    shards_dir: str,
    num_queries: int = 100,
    bands: int = 16,
    rows: int = 4,
    bloom_p: float = 0.001,
    max_gallery_records: Optional[int] = None,
    output_path: str = "reports/latency_4way_results.json",
    seed: int = 42,
) -> Dict[str, Any]:
    print("=" * 95)
    print("   EXPERIMENT 3: 4-WAY ARCHITECTURAL LATENCY & THROUGHPUT BENCHMARK")
    print(f"   Bands: {bands} | Rows: {rows} | Bloom Target FP: {bloom_p} | Seed: {seed}")
    print("=" * 95)

    engine = VectorizedMinHashEngine(num_hashes=64, seed=seed)

    print("\n[Phase 1/4] Loading Gallery Signatures (Pre-allocated zero-copy)...")
    dataset = ShardedSignatureDataset(shards_dir)
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery_records)
    num_gallery = len(gallery_sigs)
    print(f"  Loaded gallery: {num_gallery:,} records (Memory: {gallery_sigs.nbytes / (1024**2):.1f} MB)")

    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        gt_map.setdefault(int(ident), []).append(idx)
    print(f"  Unique identities: {len(gt_map):,}")

    print(f"\n[Phase 2/4] Building LSH Index & Probabilistic Band Bloom Filter (Parallel Sort)...")
    t0 = time.perf_counter()
    lsh_index = ScaledLSHIndex(bands=bands, rows=rows, bloom_fp_rate=bloom_p)
    lsh_index.index_signatures(gallery_sigs)
    build_time = time.perf_counter() - t0
    print(f"  Index and Bloom Filter built in {build_time:.2f}s")

    print(f"\n[Phase 3/4] Extracting {num_queries} Positive Probe Minutiae for Runtime MinHash...")
    pos_minutiae, pos_ids = load_positive_probe_minutiae(
        Path(dataset_dir), Path(shards_dir), num_queries=num_queries, seed=seed
    )
    print(f"  Extracted {len(pos_minutiae)} genuine probe samples across shards")

    # Generate random negative signatures (un-indexed impostors)
    rng = np.random.default_rng(seed + 999)
    neg_sigs = rng.integers(0, np.iinfo(np.uint32).max, size=(num_queries * 4, 64), dtype=np.uint32)

    all_results: Dict[str, Any] = {
        "metadata": {
            "experiment": "exp3_latency_4way",
            "gallery_size": num_gallery,
            "queries_per_workload": len(pos_minutiae),
            "bands": bands,
            "rows": rows,
            "bloom_p": bloom_p,
            "theoretical_threshold": round((1.0 / bands) ** (1.0 / rows), 4),
            "probe_computation": "runtime_minutiae_minhash",
            "similarity_metric": "fraction_matching_positions",
            "index_build_sec": round(build_time, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "workloads": {},
    }

    print("\n[Phase 4/4] Executing Workloads across 4 System Configurations...")
    workload_specs = [
        ("Positive Workload (100% In Database)", "positive"),
        ("Negative Workload (100% Not In Database)", "negative"),
        ("Mixed Workload (50% Positive / 50% Negative)", "mixed"),
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

    # Master summary table
    print("\n" + "=" * 105)
    print("                                4-WAY ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 105)
    pos = all_results["workloads"]["Positive Workload (100% In Database)"]
    neg = all_results["workloads"]["Negative Workload (100% Not In Database)"]
    mix = all_results["workloads"]["Mixed Workload (50% Positive / 50% Negative)"]

    print(f"{'Architecture':<42} | {'Positive (ms)':>14} | {'Negative (ms)':>14} | {'Mixed (ms)':>14} | {'Max Speedup':>12}")
    print("-" * 105)
    for key in ["1_only_minhash", "2_lsh_and_prefilter", "3_prefilter_no_lsh", "4_lsh_no_prefilter"]:
        name = pos[key]["name"]
        p_ms = f"{pos[key]['avg_latency_ms']:.3f} ms"
        n_ms = f"{neg[key]['avg_latency_ms']:.3f} ms"
        m_ms = f"{mix[key]['avg_latency_ms']:.3f} ms"
        spds = [pos[key].get("speedup_vs_minhash", 1.0), neg[key].get("speedup_vs_minhash", 1.0), mix[key].get("speedup_vs_minhash", 1.0)]
        max_spd = f"{max(spds):.1f}x" if max(spds) > 1.0 else "1.0x"
        print(f"{name:<42} | {p_ms:>14} | {n_ms:>14} | {m_ms:>14} | {max_spd:>12}")
    print("=" * 105)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n[Artifact] Experiment 3 results successfully saved to: {out_file.resolve()}\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: 4-Way Architectural Latency & Throughput Benchmark")
    parser.add_argument("--dataset-dir", default=str(workspace_root / "dataset"), type=str, help="Directory containing dataset_*.h5")
    parser.add_argument("--shards-dir", default=str(workspace_root / "dataset" / "signatures_shards"), type=str, help="Directory containing signatures_*.h5")
    parser.add_argument("--num-queries", default=100, type=int, help="Number of queries per workload")
    parser.add_argument("--bands", default=16, type=int, help="LSH band count")
    parser.add_argument("--rows", default=4, type=int, help="LSH row count")
    parser.add_argument("--bloom-p", default=0.001, type=float, help="Bloom filter target false positive rate")
    parser.add_argument("--max-gallery", default=None, type=int, help="Max gallery records to index (None = all)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--output", default="reports/latency_4way_results.json", type=str, help="Output JSON path")
    args = parser.parse_args()

    run_latency_comparison(
        dataset_dir=args.dataset_dir,
        shards_dir=args.shards_dir,
        num_queries=args.num_queries,
        bands=args.bands,
        rows=args.rows,
        bloom_p=args.bloom_p,
        max_gallery_records=args.max_gallery,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
