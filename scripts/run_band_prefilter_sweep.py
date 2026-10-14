"""
Band Pre-Filter Sweep Experiment.

Sweeps over different (bands, rows) configurations to characterise:
  - Pre-filter false negative rate  (did we miss a true match?)
  - Pre-filter false positive rate  (accepted queries that have no match)
  - Fraction of gallery queries accepted by the pre-filter
  - Full-pipeline Hit@10, P@10, R@10, MRR
  - Latency: pre-filter time vs full query time

All probe signatures are computed at RUNTIME from raw minutiae (no pre-computed
probe signatures are used). Gallery signatures are pre-loaded once.

Usage:
    python scripts/run_band_prefilter_sweep.py
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


# -------------------------------------------------------------------------
# Runtime probe loader — reads raw minutiae from dataset_XXX.h5 files
# -------------------------------------------------------------------------

def load_probe_minutiae(
    dataset_dir: Path,
    shards_dir: Path,
    num_shards: int = 2,
    seed: int = 42,
    samples_per_quality: int = 100,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[int], List[int]]:
    """
    Returns raw minutiae for probe samples so signatures can be computed at runtime.
    Selects balanced samples across all quality levels using metadata first, then
    loads minutiae only for the selected subset (O(samples) disk reads instead of O(probes)).
    """
    rng = np.random.default_rng(seed)

    sig_files = sorted(shards_dir.glob("signatures_*.h5"))[:num_shards]
    ds_files  = sorted(dataset_dir.glob("dataset_*.h5"))[:num_shards]

    # Step 1: Collect candidates by quality from metadata (fast 1D in-memory reads)
    candidates_by_qual: Dict[int, List[Tuple[int, int, int]]] = {}  # qual -> list of (shard_idx, gi, ident)

    for shard_idx, sig_path in enumerate(sig_files):
        with h5py.File(sig_path, "r") as sf:
            is_train = sf["metadata/is_train"][:]
            ids      = sf["metadata/synthetic_identity"][:]
            quals    = sf["metadata/quality"][:]

            probe_idx = np.where(~is_train)[0]
            for gi in probe_idx:
                q = int(quals[gi])
                ident = int(ids[gi])
                candidates_by_qual.setdefault(q, []).append((shard_idx, int(gi), ident))

    # Step 2: Randomly sample balanced queries per quality
    selected_per_shard: Dict[int, List[Tuple[int, int, int]]] = {}  # shard_idx -> list of (gi, ident, qual)
    for q in sorted(candidates_by_qual.keys()):
        cand_list = candidates_by_qual[q]
        n = min(samples_per_quality, len(cand_list))
        chosen_indices = rng.choice(len(cand_list), size=n, replace=False)
        for ci in chosen_indices:
            s_idx, gi, ident = cand_list[ci]
            selected_per_shard.setdefault(s_idx, []).append((gi, ident, q))

    # Step 3: Read minutiae ONLY for selected items
    sel_minutiae = []
    sel_ids = []
    sel_quals = []

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


# -------------------------------------------------------------------------
# Single (bands, rows) evaluation
# -------------------------------------------------------------------------

def evaluate_config(
    bands: int,
    rows: int,
    gallery_sigs: np.ndarray,
    gallery_ids:  np.ndarray,
    probe_minutiae: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    probe_ids:   List[int],
    probe_quals: List[int],
    engine: VectorizedMinHashEngine,
    tau: float = 0.35,
    k: int = 10,
) -> Dict[str, Any]:
    """Builds index for (bands, rows) and evaluates the full pre-filter + retrieval pipeline."""

    # Build index
    t_idx0 = time.perf_counter()
    lsh = ScaledLSHIndex(bands=bands, rows=rows)
    lsh.index_signatures(gallery_sigs)
    t_idx1 = time.perf_counter()

    # Ground truth: identity → gallery indices
    gt_map: Dict[int, List[int]] = {}
    for idx, ident in enumerate(gallery_ids):
        ident = int(ident)
        gt_map.setdefault(ident, []).append(idx)

    n_queries   = len(probe_minutiae)
    prefilter_accepted = 0
    prefilter_rejected = 0
    prefilter_false_negatives = 0   # rejected but had a true match
    hits_at_k   = 0
    total_recall = 0.0
    total_prec   = 0.0
    total_mrr    = 0.0
    candidates_list = []
    latencies_full  = []
    latencies_pf    = []

    for (pts, angles, types), qid, qq in zip(probe_minutiae, probe_ids, probe_quals):

        # --- Runtime signature computation ---
        t0 = time.perf_counter()
        q_sig = engine.process_observation(pts, angles, types)
        t_sig = time.perf_counter()

        # --- Pre-filter ---
        accepted = lsh.prefilter(q_sig)
        t_pf = time.perf_counter()
        latencies_pf.append((t_pf - t_sig) * 1000)

        gt_indices = set(gt_map.get(qid, []))

        if not accepted:
            prefilter_rejected += 1
            if gt_indices:
                prefilter_false_negatives += 1
            latencies_full.append((t_pf - t0) * 1000)
            continue

        prefilter_accepted += 1

        # --- Full retrieval ---
        candidates = lsh.find_candidates(q_sig)
        candidates_list.append(len(candidates))

        results = ScaledLSHIndex.score_candidates(
            candidates, q_sig, gallery_sigs, k=k
        )
        t1 = time.perf_counter()
        latencies_full.append((t1 - t0) * 1000)

        # Metrics
        retrieved_gt = [idx for idx, score in results if idx in gt_indices and score >= tau]
        hit = any(idx in gt_indices for idx, _ in results)
        if hit:
            hits_at_k += 1

        recall = len(retrieved_gt) / len(gt_indices) if gt_indices else 0.0
        prec   = len(retrieved_gt) / k
        total_recall += recall
        total_prec   += prec

        for rank, (idx, _) in enumerate(results, 1):
            if idx in gt_indices:
                total_mrr += 1.0 / rank
                break

    lats = np.array(latencies_full)
    pf_lats = np.array(latencies_pf)

    return {
        "bands":                      bands,
        "rows":                       rows,
        "threshold_t":                round((1.0 / bands) ** (1.0 / rows), 4),
        "index_time_sec":             round(t_idx1 - t_idx0, 2),
        "n_queries":                  n_queries,
        "bloom_filter_accepted":      prefilter_accepted,
        "bloom_filter_rejected":      prefilter_rejected,
        "bloom_filter_false_negatives": prefilter_false_negatives,
        "bloom_filter_fn_rate_pct":   round(prefilter_false_negatives / n_queries * 100, 3),
        "prefilter_accepted":         prefilter_accepted,
        "prefilter_rejected":         prefilter_rejected,
        "prefilter_fn":               prefilter_false_negatives,
        "prefilter_fn_rate":          round(prefilter_false_negatives / n_queries * 100, 3),
        "avg_candidates":             round(float(np.mean(candidates_list)), 2) if candidates_list else 0,
        "pruning_pct":                round((1 - np.mean(candidates_list) / len(gallery_sigs)) * 100, 4) if candidates_list else 100.0,
        "hit_at_k_pct":               round(hits_at_k / n_queries * 100, 3),
        "avg_precision":              round(total_prec / n_queries, 4),
        "avg_recall":                 round(total_recall / n_queries, 4),
        "mrr":                        round(total_mrr / n_queries, 4),
        "avg_latency_ms":             round(float(np.mean(lats)), 3),
        "p50_latency_ms":             round(float(np.percentile(lats, 50)), 3),
        "p99_latency_ms":             round(float(np.percentile(lats, 99)), 3),
        "avg_bloom_filter_ms":        round(float(np.mean(pf_lats)), 4),
        "avg_prefilter_ms":           round(float(np.mean(pf_lats)), 4),
    }


# -------------------------------------------------------------------------
# Main sweep
# -------------------------------------------------------------------------

CONFIGS = [
    (4,  16),
    (6,  10),
    (8,  8),
    (10, 6),
    (12, 5),
    (16, 4),
    (20, 3),
    (32, 2),
    (64, 1),
]


def main():
    parser = argparse.ArgumentParser(description="LSH Band Pre-Filter Sweep")
    parser.add_argument("--dataset-dir",  default="dataset",                        type=str)
    parser.add_argument("--shards-dir",   default="dataset/signatures_shards",      type=str)
    parser.add_argument("--num-shards",   default=2,    type=int,
                        help="Raw dataset shards to use for probes (2 = 20k identities)")
    parser.add_argument("--max-gallery-records", default=None, type=int,
                        help="Max gallery records to load (default: num_shards * 100k)")
    parser.add_argument("--samples-per-quality", default=100, type=int)
    parser.add_argument("--tau",          default=0.35, type=float)
    parser.add_argument("--k",            default=10,   type=int)
    parser.add_argument("--seed",         default=42,   type=int)
    parser.add_argument("--output",       default="reports/band_prefilter_sweep.json", type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    shards_dir  = Path(args.shards_dir)
    engine      = VectorizedMinHashEngine(num_hashes=64, seed=args.seed)

    # Load gallery (from pre-computed signature shards — gallery is fixed)
    max_gallery = args.max_gallery_records
    if max_gallery is None and args.num_shards and args.num_shards < 40:
        max_gallery = args.num_shards * 100_000

    print(f"Loading gallery signatures (max_records={max_gallery})...")
    dataset = ShardedSignatureDataset(str(shards_dir))
    gallery_sigs, gallery_ids = dataset.load_gallery(max_records=max_gallery)
    print(f"  Gallery: {len(gallery_sigs):,} records, {len(np.unique(gallery_ids)):,} unique identities")

    # Load probe minutiae at runtime
    print(f"\nLoading probe minutiae from {args.num_shards} raw shards (runtime computation)...")
    probe_minutiae, probe_ids, probe_quals = load_probe_minutiae(
        dataset_dir   = dataset_dir,
        shards_dir    = shards_dir,
        num_shards    = args.num_shards,
        seed          = args.seed,
        samples_per_quality = args.samples_per_quality,
    )
    print(f"  Probes: {len(probe_minutiae):,} (balanced across {len(set(probe_quals))} quality levels)")

    # Sweep
    results = []
    print(f"\n{'Band':>6} {'Rows':>5} {'t':>6} | {'FN%':>6} | {'FN_n':>5} | {'Hit@K':>7} | {'MRR':>6} | {'Cand':>8} | {'Lat_ms':>8} | {'PF_ms':>7}")
    print("-" * 90)

    for bands, rows in CONFIGS:
        if bands * rows != 64:
            # Nearest valid rows
            rows = 64 // bands
        r = evaluate_config(
            bands=bands, rows=rows,
            gallery_sigs=gallery_sigs, gallery_ids=gallery_ids,
            probe_minutiae=probe_minutiae, probe_ids=probe_ids, probe_quals=probe_quals,
            engine=engine, tau=args.tau, k=args.k,
        )
        results.append(r)
        print(
            f"{bands:>6} {rows:>5} {r['threshold_t']:>6.3f} | "
            f"{r['prefilter_fn_rate']:>6.3f} | "
            f"{r['prefilter_fn']:>5} | "
            f"{r['hit_at_k_pct']:>7.2f} | "
            f"{r['mrr']:>6.4f} | "
            f"{r['avg_candidates']:>8.1f} | "
            f"{r['avg_latency_ms']:>8.3f} | "
            f"{r['avg_prefilter_ms']:>7.4f}"
        )

    # Best config: zero FN with fewest candidates
    zero_fn = [r for r in results if r["prefilter_fn"] == 0]
    best = min(zero_fn, key=lambda x: x["avg_candidates"]) if zero_fn else min(results, key=lambda x: x["prefilter_fn"])
    print(f"\nBest config (zero FN, fewest candidates): bands={best['bands']}, rows={best['rows']}, t={best['threshold_t']}")

    out = {
        "metadata": {
            "gallery_size": len(gallery_sigs),
            "n_probes":     len(probe_minutiae),
            "tau":          args.tau,
            "k":            args.k,
            "seed":         args.seed,
            "split":        "randomised_per_identity",
            "probe_signatures": "computed_at_runtime",
        },
        "best_config": best,
        "sweep_results": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
