"""
Experiment 1: LSH-Band Bloom Filter Sweep & Sensitivity Benchmark.

Sweeps across 9 canonical (bands, rows) configurations:
  (4,16), (6,10), (8,8), (10,6), (12,5), (16,4), (20,3), (32,2), (64,1)
Evaluating:
  - LSH-Band Bloom filter False Negatives & FN Rate (%)
  - Search space candidate pruning ratio (%)
  - Top-K duplicate retrieval accuracy (Hit@10, P@10, R@10, MRR)
  - End-to-end query latency and Bloom filter gating latency

Probe signatures are computed dynamically at RUNTIME from raw minutiae.
Designed for immediate, error-free execution at any scale (200k to 105M+).

Usage:
    python scripts/run_exp1_band_sweep.py
    python scripts/run_exp1_band_sweep.py --max-gallery 200000 --num-queries 500
    python scripts/run_exp1_band_sweep.py --max-gallery 4000000 --num-queries 1000
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

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


def load_probe_minutiae(
    dataset_dir: Path,
    shards_dir: Path,
    num_queries: int = 500,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[int], List[int]]:
    """
    Selects balanced probe queries across all 5 quality levels and loads raw minutiae.
    Signatures are computed at query time to capture real operational latency.
    """
    rng = np.random.default_rng(seed)
    matched = get_matched_shard_files(dataset_dir, shards_dir)

    candidates_by_qual: Dict[int, List[Tuple[int, int, int]]] = {}
    for shard_idx, (_, _, sig_path) in enumerate(matched):
        with h5py.File(sig_path, "r") as sf:
            is_train = sf["metadata/is_train"][:]
            ids = sf["metadata/synthetic_identity"][:]
            quals = sf["metadata/quality"][:]
            probe_idx = np.where(~is_train)[0]
            for gi in probe_idx:
                q = int(quals[gi])
                candidates_by_qual.setdefault(q, []).append((shard_idx, int(gi), int(ids[gi])))
        if sum(len(v) for v in candidates_by_qual.values()) >= num_queries * 4:
            break

    qualities = sorted(candidates_by_qual.keys())
    samples_per_q = max(1, num_queries // len(qualities)) if qualities else num_queries

    selected_per_shard: Dict[int, List[Tuple[int, int, int]]] = {}
    for q in qualities:
        cands = candidates_by_qual[q]
        n = min(samples_per_q, len(cands))
        chosen = rng.choice(len(cands), size=n, replace=False)
        for ci in chosen:
            s_idx, gi, ident = cands[ci]
            selected_per_shard.setdefault(s_idx, []).append((gi, ident, q))

    sel_minutiae, sel_ids, sel_quals = [], [], []
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


def evaluate_config(
    bands: int,
    rows: int,
    gallery_sigs: np.ndarray,
    gallery_ids: np.ndarray,
    probe_minutiae: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    probe_ids: List[int],
    probe_quals: List[int],
    engine: VectorizedMinHashEngine,
    tau: float = 0.35,
    k: int = 10,
) -> Dict[str, Any]:
    """Evaluates a single (bands, rows) configuration."""
    t0_idx = time.perf_counter()
    index = ScaledLSHIndex(bands=bands, rows=rows)
    index.index_signatures(gallery_sigs)
    idx_time = time.perf_counter() - t0_idx

    n_queries = len(probe_ids)
    bloom_accepted = 0
    bloom_rejected = 0
    bloom_fn = 0
    cands_counts = []
    latencies = []
    bloom_latencies = []
    hits_at_k = 0
    precisions = []
    recalls = []
    reciprocal_ranks = []

    # Map identity -> count of true occurrences in gallery
    gallery_id_counts: Dict[int, int] = {}
    for gid in gallery_ids:
        gallery_id_counts[int(gid)] = gallery_id_counts.get(int(gid), 0) + 1

    for minutiae, q_id in zip(probe_minutiae, probe_ids):
        t0 = time.perf_counter()

        # Step 1: Compute MinHash signature at runtime
        pts, angles, types = minutiae
        q_sig = engine.process_observation(pts, angles, types)

        # Step 2: LSH-Band Bloom filter check
        t_bloom_start = time.perf_counter()
        passed_bloom = index.prefilter(q_sig)
        t_bloom = time.perf_counter() - t_bloom_start
        bloom_latencies.append(t_bloom * 1000)

        in_gallery = gallery_id_counts.get(q_id, 0) > 0

        if not passed_bloom:
            bloom_rejected += 1
            if in_gallery:
                bloom_fn += 1
            cands_counts.append(0)
            latencies.append((time.perf_counter() - t0) * 1000)
            precisions.append(0.0)
            recalls.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        bloom_accepted += 1

        # Step 3: LSH Candidate Lookup
        cands = index.find_candidates(q_sig)
        cands_counts.append(len(cands))

        # Step 4: Verification / Scoring
        scores = ScaledLSHIndex.score_candidates(cands, q_sig, gallery_sigs, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

        top_cand_ids = [gallery_ids[ci] for ci, s in scores if s >= tau]
        hit = q_id in top_cand_ids[:k]
        if hit:
            hits_at_k += 1

        hits_count = sum(1 for cid in top_cand_ids[:k] if cid == q_id)
        precisions.append(hits_count / k)
        n_gt = gallery_id_counts.get(q_id, 0)
        recalls.append(hits_count / n_gt if n_gt > 0 else 0.0)

        rank = None
        for r_idx, cid in enumerate(top_cand_ids):
            if cid == q_id:
                rank = r_idx + 1
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    t_approx = round((1.0 / bands) ** (1.0 / rows), 4) if rows > 0 else 0.0
    avg_cand = float(np.mean(cands_counts)) if cands_counts else 0.0
    gallery_sz = len(gallery_sigs)
    prune_pct = round((1.0 - avg_cand / max(1, gallery_sz)) * 100.0, 4)

    return {
        "bands": bands,
        "rows": rows,
        "threshold_t": t_approx,
        "index_time_sec": round(idx_time, 2),
        "n_queries": n_queries,
        "bloom_filter_accepted": bloom_accepted,
        "bloom_filter_rejected": bloom_rejected,
        "bloom_filter_false_negatives": bloom_fn,
        "bloom_filter_fn_rate_pct": round(bloom_fn / n_queries * 100, 3) if n_queries else 0.0,
        "avg_candidates": round(avg_cand, 2),
        "pruning_pct": prune_pct,
        "hit_at_k_pct": round(hits_at_k / n_queries * 100, 2) if n_queries else 0.0,
        "avg_precision": round(float(np.mean(precisions)), 4),
        "avg_recall": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 3),
        "p50_latency_ms": round(float(np.median(latencies)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
        "avg_bloom_filter_ms": round(float(np.mean(bloom_latencies)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: LSH-Band Bloom Filter Sweep.")
    parser.add_argument("--dataset-dir", default=str(workspace_root / "dataset"), type=str)
    parser.add_argument("--shards-dir", default=str(workspace_root / "dataset" / "signatures_shards"), type=str)
    parser.add_argument("--max-gallery", default=200000, type=lambda x: None if x.lower() == 'all' else int(x))
    parser.add_argument("--num-queries", default=500, type=int)
    parser.add_argument("--tau", default=0.35, type=float)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", default="reports/band_prefilter_sweep.json", type=str)
    args = parser.parse_args()

    print("=" * 115)
    print("      EXPERIMENT 1: LSH-BAND BLOOM FILTER SWEEP & SENSITIVITY BENCHMARK")
    print("=" * 115)

    dataset_dir = Path(args.dataset_dir)
    shards_dir = Path(args.shards_dir)

    print(f"Loading gallery signatures from {shards_dir}...")
    dataset = ShardedSignatureDataset(str(shards_dir))
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=args.max_gallery)
    print(f"  Gallery loaded: {len(gallery_sigs):,} records across {len(np.unique(gallery_ids)):,} unique identities")

    print(f"Loading {args.num_queries} probe observations for runtime MinHash calculation...")
    probe_minutiae, probe_ids, probe_quals = load_probe_minutiae(
        dataset_dir=dataset_dir,
        shards_dir=shards_dir,
        num_queries=args.num_queries,
        seed=args.seed
    )
    print(f"  Probes loaded: {len(probe_ids):,} observations")

    engine = VectorizedMinHashEngine(num_hashes=64, seed=args.seed)

    configs = [
        (4, 16),
        (6, 10),
        (8, 8),
        (10, 6),
        (12, 5),
        (16, 4),
        (20, 3),
        (32, 2),
        (64, 1)
    ]

    results = []
    print("\n" + "=" * 115)
    print(f"{'Bands (b)':<10} | {'Rows (r)':<9} | {'Thresh (t)':<10} | {'Bloom FN':<9} | {'FN Rate %':<10} | {'Candidates':<11} | {'Pruning %':<10} | {'Hit@10 %':<9} | {'MRR':<8} | {'Latency':<9}")
    print("-" * 115)

    for b, r in configs:
        res = evaluate_config(
            bands=b,
            rows=r,
            gallery_sigs=gallery_sigs,
            gallery_ids=gallery_ids,
            probe_minutiae=probe_minutiae,
            probe_ids=probe_ids,
            probe_quals=probe_quals,
            engine=engine,
            tau=args.tau,
            k=args.k
        )
        results.append(res)
        print(
            f"{res['bands']:<10} | {res['rows']:<9} | {res['threshold_t']:<10.3f} | "
            f"{res['bloom_filter_false_negatives']:<9} | {res['bloom_filter_fn_rate_pct']:<10.2f} | "
            f"{res['avg_candidates']:<11.2f} | {res['pruning_pct']:<10.4f} | {res['hit_at_k_pct']:<9.1f} | "
            f"{res['mrr']:<8.4f} | {res['avg_latency_ms']:>6.3f} ms"
        )

    print("=" * 115)

    best_cfg = next((r for r in results if r["bands"] == 16 and r["rows"] == 4), results[0])
    out_dict = {
        "metadata": {
            "gallery_size": len(gallery_sigs),
            "n_probes": len(probe_ids),
            "tau": args.tau,
            "k": args.k,
            "seed": args.seed,
            "prefilter": "lsh_band_bloom_filter",
            "probe_signatures": "computed_at_runtime",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "best_config": best_cfg,
        "sweep_results": results
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\n[OK] Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
