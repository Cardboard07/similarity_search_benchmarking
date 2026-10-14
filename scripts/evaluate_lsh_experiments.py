"""
Comprehensive Evaluation of LSH Parameter Tuning, Recall Mechanics, and Precision-Candidate Curves.

Evaluates 20,000 unique synthetic fingerprint identities (200,000 gallery samples)
across all 5 quality levels and 3 orientations:
  1. Recall formulation: mathematical definition, per-query vs macro-average,
     zero-recall failure cases, and False Negative comparison (With vs Without LSH).
  2. LSH Slider Tuning: systematic sweep over (bands, rows) configurations,
     measuring candidate count, pruning ratio, candidate precision, Top-K precision,
     recall, zero-candidate failures, and latency.
  3. Precision vs Candidates Curve: mapping the empirical precision-candidate tradeoff.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from src.lsh_scale.lsh_index_6m import ScaledLSHIndex
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine


def load_20k_dataset(
    shards_dir: Path, dataset_dir: Path, samples_per_quality: int = 100, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Loads 20,000 identities (shards 0 and 1).
    Gallery signatures are loaded from signature shards.
    Probe raw minutiae are loaded from dataset shards for runtime signature computation.
    Selects balanced probe queries across metadata first, then loads minutiae only for the subset.
    Returns:
        gallery_sigs, gallery_ids, query_minutiae, query_ids, query_quals
    """
    print("[1/5] Loading 20,000 identities (shards 0 and 1)...")
    t0 = time.time()
    g_sigs, g_ids = [], []
    rng = np.random.default_rng(seed)

    candidates_by_qual: Dict[int, List[Tuple[int, int, int]]] = {}

    for shard in [0, 1]:
        sf = shards_dir / f"signatures_{shard:03d}.h5"
        with h5py.File(sf, "r") as s_f:
            is_train = s_f["metadata/is_train"][:]
            sigs = s_f["data/signatures"][:]
            ids = s_f["metadata/synthetic_identity"][:]
            quals = s_f["metadata/quality"][:]

            g_sigs.append(sigs[is_train])
            g_ids.append(ids[is_train])

            probe_idx = np.where(~is_train)[0]
            for gi in probe_idx:
                q = int(quals[gi])
                ident = int(ids[gi])
                candidates_by_qual.setdefault(q, []).append((shard, int(gi), ident))

    gallery_sigs = np.vstack(g_sigs)
    gallery_ids = np.concatenate(g_ids)

    # Balanced query selection across [3, 7, 5, 1, 9]
    selected_per_shard: Dict[int, List[Tuple[int, int, int]]] = {}
    for q in [3, 7, 5, 1, 9]:
        cand_list = candidates_by_qual.get(q, [])
        n = min(samples_per_quality, len(cand_list))
        chosen = rng.choice(len(cand_list), size=n, replace=False)
        for ci in chosen:
            s_idx, gi, ident = cand_list[ci]
            selected_per_shard.setdefault(s_idx, []).append((gi, ident, q))

    # Read minutiae ONLY for selected queries
    query_minutiae, query_ids, query_quals = [], [], []
    for s_idx, items in selected_per_shard.items():
        df = dataset_dir / f"dataset_{s_idx:03d}.h5"
        with h5py.File(df, "r") as d_f:
            d_min = d_f["data/minutiae"]
            d_num = d_f["data/num_minutiae"]
            for gi, ident, q in items:
                c = int(d_num[gi])
                pts = d_min[gi, :c, :2].copy()
                angles = d_min[gi, :c, 2].copy()
                types = d_min[gi, :c, 3].astype(np.int32)
                query_minutiae.append((pts, angles, types))
                query_ids.append(ident)
                query_quals.append(q)

    perm = rng.permutation(len(query_ids))
    sel_min = [query_minutiae[i] for i in perm]
    sel_ids = np.array([query_ids[i] for i in perm])
    sel_quals = np.array([query_quals[i] for i in perm])

    elapsed = time.time() - t0
    print(f"      Gallery loaded: {len(gallery_sigs):,} records across {len(np.unique(gallery_ids)):,} identities.")
    print(f"      Probe minutiae loaded: {len(sel_min):,} balanced queries. (Elapsed: {elapsed:.2f}s)")
    return gallery_sigs, gallery_ids, sel_min, sel_ids, sel_quals


def run_experiment_1_recall_analysis(
    gallery_sigs: np.ndarray,
    gallery_ids: np.ndarray,
    query_minutiae: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    query_ids: np.ndarray,
    query_quals: np.ndarray,
    engine: VectorizedMinHashEngine,
) -> Dict[str, Any]:
    """
    Experiment 1: Comprehensive Recall & Zero-Recall Analysis.
    Compares Without LSH (Brute Force) vs LSH at multiple configurations.
    Probe signatures are computed AT RUNTIME from raw minutiae.
    """
    print("\n[2/5] Running Experiment 1: Recall Formulation & Zero-Recall Breakdown...")
    n_queries = len(query_minutiae)
    n_gallery = len(gallery_sigs)

    # 1. Without LSH (Brute-Force Linear Scan)
    print("      Evaluating Without LSH (Brute-Force Scan of 200k gallery)...")
    t0_bf = time.perf_counter()
    bf_recall_10_list = []
    bf_recall_1_list = []
    bf_hit_10_list = []
    bf_hit_1_list = []
    bf_by_qual = {q: {"recalls_10": [], "hits_10": [], "zero_cnt": 0} for q in [3, 7, 5, 1, 9]}

    for i in range(n_queries):
        q_id = query_ids[i]
        pts, angles, types = query_minutiae[i]
        q_q = query_quals[i]

        q_sig = engine.process_observation(pts, angles, types)
        sims = np.mean(gallery_sigs == q_sig, axis=1)
        top10_idx = np.argpartition(sims, -10)[-10:]
        top10_idx = top10_idx[np.argsort(-sims[top10_idx])]

        hits10 = np.sum(gallery_ids[top10_idx] == q_id)
        hits1 = 1 if gallery_ids[top10_idx[0]] == q_id else 0

        rec10 = hits10 / 10.0
        rec1 = hits1 / 10.0

        bf_recall_10_list.append(rec10)
        bf_recall_1_list.append(rec1)
        bf_hit_10_list.append(1 if hits10 > 0 else 0)
        bf_hit_1_list.append(hits1)

        bf_by_qual[q_q]["recalls_10"].append(rec10)
        bf_by_qual[q_q]["hits_10"].append(1 if hits10 > 0 else 0)
        if hits10 == 0:
            bf_by_qual[q_q]["zero_cnt"] += 1

    t_bf_elapsed = time.perf_counter() - t0_bf
    bf_zero_rec10 = sum(1 for r in bf_recall_10_list if r == 0.0)

    # Helper for evaluating an LSH configuration
    def eval_lsh_cfg(b: int, r: int, label: str):
        print(f"      Evaluating {label} (bands={b}, rows={r})...")
        lsh = ScaledLSHIndex(bands=b, rows=r)
        lsh.index_signatures(gallery_sigs)
        t0 = time.perf_counter()
        rec10_list, hit10_list, cand_rec_list, cand_hit_list = [], [], [], []
        cand_counts = []
        zero_cand_cnt = 0
        zero_rec10_cnt = 0
        bf_rejected_cnt = 0
        bf_fn_cnt = 0
        by_qual = {q: {"recalls_10": [], "hits_10": [], "cand_hits": [], "zero_cnt": 0, "zero_cand_cnt": 0} for q in [3, 7, 5, 1, 9]}
        
        for i in range(n_queries):
            q_id = query_ids[i]
            pts, angles, types = query_minutiae[i]
            q_q = query_quals[i]

            q_sig = engine.process_observation(pts, angles, types)

            # Test LSH Band Bloom Filter
            if not lsh.prefilter(q_sig):
                bf_rejected_cnt += 1
                bf_fn_cnt += 1
                cand_counts.append(0)
                zero_cand_cnt += 1
                zero_rec10_cnt += 1
                rec10_list.append(0.0)
                hit10_list.append(0)
                cand_rec_list.append(0.0)
                cand_hit_list.append(0)
                by_qual[q_q]["recalls_10"].append(0.0)
                by_qual[q_q]["hits_10"].append(0)
                by_qual[q_q]["cand_hits"].append(0)
                by_qual[q_q]["zero_cnt"] += 1
                by_qual[q_q]["zero_cand_cnt"] += 1
                continue

            cands = lsh.find_candidates(q_sig)
            n_c = len(cands)
            cand_counts.append(n_c)
            
            if n_c == 0:
                zero_cand_cnt += 1
                zero_rec10_cnt += 1
                rec10_list.append(0.0)
                hit10_list.append(0)
                cand_rec_list.append(0.0)
                cand_hit_list.append(0)
                by_qual[q_q]["recalls_10"].append(0.0)
                by_qual[q_q]["hits_10"].append(0)
                by_qual[q_q]["cand_hits"].append(0)
                by_qual[q_q]["zero_cnt"] += 1
                by_qual[q_q]["zero_cand_cnt"] += 1
                continue
                
            c_arr = np.array(cands, dtype=np.int64)
            c_ids = gallery_ids[c_arr]
            n_true = np.sum(c_ids == q_id)
            cand_hit = 1 if n_true > 0 else 0
            cand_rec = n_true / 10.0
            cand_hit_list.append(cand_hit)
            cand_rec_list.append(cand_rec)
            by_qual[q_q]["cand_hits"].append(cand_hit)
            if n_true == 0:
                zero_cand_cnt += 1
                by_qual[q_q]["zero_cand_cnt"] += 1
                
            c_sigs = gallery_sigs[c_arr]
            sims = np.mean(c_sigs == q_sig, axis=1)
            if n_c > 10:
                top_sub = np.argpartition(sims, -10)[-10:]
                top_sub = top_sub[np.argsort(-sims[top_sub])]
            else:
                top_sub = np.argsort(-sims)
                
            hits10 = np.sum(c_ids[top_sub] == q_id)
            rec10 = hits10 / 10.0
            hit10 = 1 if hits10 > 0 else 0
            
            rec10_list.append(rec10)
            hit10_list.append(hit10)
            by_qual[q_q]["recalls_10"].append(rec10)
            by_qual[q_q]["hits_10"].append(hit10)
            if hits10 == 0:
                zero_rec10_cnt += 1
                by_qual[q_q]["zero_cnt"] += 1
                
        elapsed = time.perf_counter() - t0
        return {
            "method": f"LSH Index (bands={b}, rows={r} - {label})",
            "total_queries": n_queries,
            "bloom_filter_rejected": int(bf_rejected_cnt),
            "bloom_filter_false_negatives": int(bf_fn_cnt),
            "bloom_filter_fn_rate_pct": float(bf_fn_cnt / n_queries * 100.0),
            "avg_candidates": float(np.mean(cand_counts)),
            "candidate_hit_rate_pct": float(np.mean(cand_hit_list) * 100.0),
            "candidate_mean_recall": float(np.mean(cand_rec_list)),
            "mean_recall_at_10": float(np.mean(rec10_list)),
            "hit_rate_at_10_pct": float(np.mean(hit10_list) * 100.0),
            "candidate_false_negatives": int(zero_cand_cnt),
            "candidate_false_negative_pct": float(zero_cand_cnt / n_queries * 100.0),
            "zero_recall_at_10_count": int(zero_rec10_cnt),
            "zero_recall_at_10_pct": float(zero_rec10_cnt / n_queries * 100.0),
            "avg_latency_ms": float(elapsed / n_queries * 1000.0),
            "by_quality": {
                int(q): {
                    "candidate_hit_pct": float(np.mean(by_qual[q]["cand_hits"]) * 100.0),
                    "hit_at_10_pct": float(np.mean(by_qual[q]["hits_10"]) * 100.0),
                    "mean_recall_at_10": float(np.mean(by_qual[q]["recalls_10"])),
                    "zero_recall_count": int(by_qual[q]["zero_cnt"]),
                    "zero_recall_pct": float(by_qual[q]["zero_cnt"] / len(by_qual[q]["hits_10"]) * 100.0)
                } for q in [3, 7, 5, 1, 9]
            }
        }

    # Evaluate multiple key configurations
    res_b16 = eval_lsh_cfg(16, 4, "High Sensitivity")
    res_b12 = eval_lsh_cfg(12, 5, "Balanced Tuned")
    res_b8 = eval_lsh_cfg(8, 8, "Standard Baseline")

    exp1_results = {
        "mathematical_formulation": {
            "per_query_definition": "Recall@K(q) = |Top-K(q) ∩ GroundTruth(q)| / |GroundTruth(q)|",
            "ground_truth_per_identity": 10,
            "averaging_method": "Macro-averaged across all |Q| queries: Mean_Recall = (1/|Q|) * sum_q Recall(q)",
            "equivalence_note": "Because every query identity has exactly 10 ground-truth samples in the gallery, Macro-Averaged Recall is mathematically identical to Micro-Averaged Recall."
        },
        "without_lsh": {
            "method": "Brute-Force Linear Scan (200,000 gallery records)",
            "total_queries": n_queries,
            "mean_recall_at_1": float(np.mean(bf_recall_1_list)),
            "mean_recall_at_10": float(np.mean(bf_recall_10_list)),
            "hit_rate_at_1_pct": float(np.mean(bf_hit_1_list) * 100.0),
            "hit_rate_at_10_pct": float(np.mean(bf_hit_10_list) * 100.0),
            "zero_recall_at_10_count": int(bf_zero_rec10),
            "zero_recall_at_10_pct": float(bf_zero_rec10 / n_queries * 100.0),
            "false_negatives_at_candidate_stage": 0,
            "false_negative_rate_candidate_pct": 0.0,
            "avg_latency_ms": float(t_bf_elapsed / n_queries * 1000.0),
            "by_quality": {
                int(q): {
                    "hit_at_10_pct": float(np.mean(bf_by_qual[q]["hits_10"]) * 100.0),
                    "mean_recall_at_10": float(np.mean(bf_by_qual[q]["recalls_10"])),
                    "zero_recall_count": int(bf_by_qual[q]["zero_cnt"]),
                    "zero_recall_pct": float(bf_by_qual[q]["zero_cnt"] / len(bf_by_qual[q]["hits_10"]) * 100.0)
                } for q in [3, 7, 5, 1, 9]
            }
        },
        "with_lsh_tuned_b16": res_b16,
        "with_lsh_balanced_b12": res_b12,
        "with_lsh_baseline_b8": res_b8
    }
    return exp1_results


def run_experiment_2_slider_tuning_and_curves(
    gallery_sigs: np.ndarray,
    gallery_ids: np.ndarray,
    query_minutiae: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    query_ids: np.ndarray,
    engine: VectorizedMinHashEngine,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Experiment 2: Tuning LSH Sliders (bands, rows) and Precision-Candidate Curves.
    Probe signatures are computed AT RUNTIME from raw minutiae.
    """
    print("\n[3/5] Running Experiment 2: LSH Slider Tuning & Precision-Candidate Mapping...")
    n_queries = len(query_minutiae)
    n_gallery = len(gallery_sigs)

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

    tuning_table = []

    for b, r in configs:
        print(f"      Tuning config: bands={b:2d}, rows={r:2d} (Threshold t ~ {(1/b)**(1/r):.3f})...")
        t0_build = time.perf_counter()
        lsh = ScaledLSHIndex(bands=b, rows=r)
        lsh.index_signatures(gallery_sigs)
        t_build = time.perf_counter() - t0_build

        cand_counts = []
        cand_hits = []
        cand_precisions = []
        top10_hits = []
        top10_recalls = []
        top10_precisions = []
        zero_cand_cnt = 0
        zero_top10_cnt = 0

        t0_q = time.perf_counter()
        for i in range(n_queries):
            q_id = query_ids[i]
            pts, angles, types = query_minutiae[i]

            q_sig = engine.process_observation(pts, angles, types)
            cands = lsh.find_candidates(q_sig)
            n_c = len(cands)
            cand_counts.append(n_c)

            if n_c == 0:
                zero_cand_cnt += 1
                zero_top10_cnt += 1
                cand_hits.append(0)
                cand_precisions.append(0.0)
                top10_hits.append(0)
                top10_recalls.append(0.0)
                top10_precisions.append(0.0)
                continue

            c_arr = np.array(cands, dtype=np.int64)
            c_ids = gallery_ids[c_arr]
            n_true = np.sum(c_ids == q_id)

            cand_hits.append(1 if n_true > 0 else 0)
            cand_precisions.append(n_true / n_c if n_c > 0 else 0.0)

            # Score candidates
            c_sigs = gallery_sigs[c_arr]
            sims = np.mean(c_sigs == q_sig, axis=1)
            k_eval = min(10, n_c)
            if n_c > 10:
                top_idx = np.argpartition(sims, -10)[-10:]
                top_idx = top_idx[np.argsort(-sims[top_idx])]
            else:
                top_idx = np.argsort(-sims)

            hits_in_top10 = np.sum(c_ids[top_idx] == q_id)
            top10_hits.append(1 if hits_in_top10 > 0 else 0)
            top10_recalls.append(hits_in_top10 / 10.0)
            top10_precisions.append(hits_in_top10 / 10.0)
            if hits_in_top10 == 0:
                zero_top10_cnt += 1

        t_q_total = time.perf_counter() - t0_q
        avg_cands = float(np.mean(cand_counts))
        pruning_ratio = float((1.0 - (avg_cands / n_gallery)) * 100.0)
        cand_hit_rate = float(np.mean(cand_hits) * 100.0)
        mean_cand_prec = float(np.mean(cand_precisions) * 100.0)
        mean_top10_hit = float(np.mean(top10_hits) * 100.0)
        mean_top10_rec = float(np.mean(top10_recalls))
        mean_top10_prec = float(np.mean(top10_precisions))
        lat_per_query_ms = float(t_q_total / n_queries * 1000.0)

        row = {
            "bands": b,
            "rows": r,
            "theoretical_threshold": round(float((1/b)**(1/r)), 4),
            "index_build_sec": round(t_build, 2),
            "avg_candidates": round(avg_cands, 1),
            "pruning_ratio_pct": round(pruning_ratio, 3),
            "candidate_hit_rate_pct": round(cand_hit_rate, 2),
            "candidate_precision_pct": round(mean_cand_prec, 4),
            "top10_hit_rate_pct": round(mean_top10_hit, 2),
            "top10_recall": round(mean_top10_rec, 4),
            "top10_precision": round(mean_top10_prec, 4),
            "zero_candidates_count": zero_cand_cnt,
            "zero_top10_count": zero_top10_cnt,
            "query_latency_ms": round(lat_per_query_ms, 3)
        }
        tuning_table.append(row)
        print(f"        -> AvgCand: {avg_cands:8.1f} | CandHit: {cand_hit_rate:5.1f}% | CandPrec: {mean_cand_prec:6.4f}% | Hit@10: {mean_top10_hit:5.1f}% | R@10: {mean_top10_rec:.4f} | Lat: {lat_per_query_ms:5.2f}ms")

    # Explore Candidate Budget Curve for Tuned High-Sensitivity Index (b=16, r=4)
    print("\n      Mapping Precision vs Candidate Budget Curve for b=16, r=4...")
    lsh_opt = ScaledLSHIndex(bands=16, rows=4)
    lsh_opt.index_signatures(gallery_sigs)

    candidate_budgets = [1, 2, 4, 8, 12, 16, 25, 50, 100, 200, 500]
    budget_curve = []

    # Compute runtime signatures and candidates for candidate budget curve
    all_query_sigs = [engine.process_observation(*query_minutiae[i]) for i in range(n_queries)]
    all_query_cands = [lsh_opt.find_candidates(all_query_sigs[i]) for i in range(n_queries)]

    for budget in candidate_budgets:
        prec_at_budget = []
        rec_at_budget = []
        hit_at_budget = []

        for i in range(n_queries):
            q_id = query_ids[i]
            q_sig = all_query_sigs[i]
            cands = all_query_cands[i]
            if len(cands) == 0:
                prec_at_budget.append(0.0)
                rec_at_budget.append(0.0)
                hit_at_budget.append(0)
                continue

            c_arr = np.array(cands, dtype=np.int64)
            c_sigs = gallery_sigs[c_arr]
            sims = np.mean(c_sigs == q_sig, axis=1)

            eval_k = min(budget, len(cands))
            if len(cands) > budget:
                top_sub = np.argpartition(sims, -budget)[-budget:]
                top_sub = top_sub[np.argsort(-sims[top_sub])]
            else:
                top_sub = np.argsort(-sims)

            sub_ids = gallery_ids[c_arr[top_sub]]
            hits = np.sum(sub_ids == q_id)
            prec_at_budget.append(hits / eval_k)
            rec_at_budget.append(hits / 10.0)
            hit_at_budget.append(1 if hits > 0 else 0)

        budget_row = {
            "candidate_budget": budget,
            "mean_precision_at_budget": round(float(np.mean(prec_at_budget)), 4),
            "mean_recall_at_budget": round(float(np.mean(rec_at_budget)), 4),
            "hit_rate_at_budget_pct": round(float(np.mean(hit_at_budget) * 100.0), 2)
        }
        budget_curve.append(budget_row)

    return {"lsh_tuning_grid": tuning_table}, {"candidate_budget_curve": budget_curve}


def generate_visualization_figures(
    exp1_data: Dict[str, Any],
    exp2_grid: List[Dict[str, Any]],
    budget_curve: List[Dict[str, Any]],
    output_dir: Path
):
    """
    Generates publication-quality visualization figures and updates artifact directory.
    """
    print("\n[4/5] Generating Visualization Figures...")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_fig_dir = Path(r"C:\Users\hrish\.gemini\antigravity-ide\brain\453cc283-9b7b-497a-bbbe-83074734020c\figures")
    artifact_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Precision vs Number of Candidates Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cand_counts = [row["avg_candidates"] for row in exp2_grid if row["avg_candidates"] > 0]
    cand_precs = [row["candidate_precision_pct"] for row in exp2_grid if row["avg_candidates"] > 0]
    band_labels = [f"b={row['bands']},r={row['rows']}" for row in exp2_grid if row["avg_candidates"] > 0]

    ax1.plot(cand_counts, cand_precs, marker="o", color="#1f77b4", linewidth=2.5, markersize=8, label="Candidate Precision (%)")
    for x, y, lbl in zip(cand_counts, cand_precs, band_labels):
        ax1.annotate(lbl, (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlabel("Average Number of Candidates Retrieved per Query (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Candidate Precision (%) = True Positives / Candidates", fontsize=11, fontweight="bold")
    ax1.set_title("Candidate Precision vs Candidate Pool Size\n(Dilution of Candidate Pool across LSH Configurations)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    budgets = [r["candidate_budget"] for r in budget_curve]
    precs = [r["mean_precision_at_budget"] * 100 for r in budget_curve]
    recs = [r["mean_recall_at_budget"] * 100 for r in budget_curve]

    ax2.plot(budgets, precs, marker="s", color="#d62728", linewidth=2.5, label="Precision@C (%)")
    ax2.plot(budgets, recs, marker="^", color="#2ca02c", linewidth=2.5, label="Recall@C (%)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Candidate Evaluation Budget C (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Value (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Precision@C & Recall@C vs Candidate Budget C (b=16, r=4)\n(Ranked Retrieval Pareto Tradeoff)", fontsize=12, fontweight="bold")
    ax2.legend(loc="center right", fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    fig1_path = output_dir / "lsh_precision_vs_candidates_curve.png"
    fig.savefig(fig1_path, dpi=300)
    fig.savefig(artifact_fig_dir / "lsh_precision_vs_candidates_curve.png", dpi=300)
    plt.close(fig)
    print(f"      Saved: {fig1_path}")

    # Figure 2: LSH Sliders Tuning Matrix (4 metrics vs Bands b)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    bands = [r["bands"] for r in exp2_grid]

    ax1.plot(bands, [r["avg_candidates"] for r in exp2_grid], marker="o", color="#1f77b4", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("LSH Bands b (Slider)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Avg Candidates (Log Scale)", fontsize=10, fontweight="bold")
    ax1.set_title("Candidates Retrieved vs Bands b", fontsize=11, fontweight="bold")
    ax1.grid(True, ls="--", alpha=0.5)

    ax2.plot(bands, [r["pruning_ratio_pct"] for r in exp2_grid], marker="s", color="#2ca02c", linewidth=2)
    ax2.set_xlabel("LSH Bands b (Slider)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Search Space Pruning (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Gallery Pruning Ratio (%) vs Bands b", fontsize=11, fontweight="bold")
    ax2.grid(True, ls="--", alpha=0.5)

    ax3.plot(bands, [r["candidate_hit_rate_pct"] for r in exp2_grid], marker="^", color="#ff7f0e", linewidth=2, label="Candidate Hit Rate (%)")
    ax3.plot(bands, [r["top10_hit_rate_pct"] for r in exp2_grid], marker="d", color="#d62728", linewidth=2, label="Hit@10 (%)")
    ax3.set_xlabel("LSH Bands b (Slider)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Retrieval Rate (%)", fontsize=10, fontweight="bold")
    ax3.set_title("Candidate Hit Rate & Hit@10 vs Bands b", fontsize=11, fontweight="bold")
    ax3.legend(loc="lower right")
    ax3.grid(True, ls="--", alpha=0.5)

    ax4.plot(bands, [r["query_latency_ms"] for r in exp2_grid], marker="v", color="#9467bd", linewidth=2)
    ax4.set_yscale("log")
    ax4.set_xlabel("LSH Bands b (Slider)", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Latency (ms/query, Log Scale)", fontsize=10, fontweight="bold")
    ax4.set_title("Query Latency vs Bands b", fontsize=11, fontweight="bold")
    ax4.grid(True, ls="--", alpha=0.5)

    fig.tight_layout()
    fig2_path = output_dir / "lsh_tuning_tradeoffs.png"
    fig.savefig(fig2_path, dpi=300)
    fig.savefig(artifact_fig_dir / "lsh_tuning_tradeoffs.png", dpi=300)
    plt.close(fig)
    print(f"      Saved: {fig2_path}")

    # Figure 3: Recall and Failure Breakdown by Quality Level
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    qual_names = ["q3 (Pristine)", "q7 (Good)", "q5 (Noisy)", "q1 (Bad)", "q9 (Very Bad)"]
    quals = [3, 7, 5, 1, 9]

    bf_by_q = exp1_data["without_lsh"]["by_quality"]
    lsh_by_q = exp1_data["with_lsh_tuned_b16"]["by_quality"]
    b8_by_q = exp1_data["with_lsh_baseline_b8"]["by_quality"]

    bf_hit_rates = [bf_by_q[q]["hit_at_10_pct"] for q in quals]
    lsh_hit_rates = [lsh_by_q[q]["hit_at_10_pct"] for q in quals]
    b8_hit_rates = [b8_by_q[q]["hit_at_10_pct"] for q in quals]

    x = np.arange(len(qual_names))
    width = 0.25

    ax1.bar(x - width, bf_hit_rates, width, label="Without LSH (Brute Force)", color="#1f77b4")
    ax1.bar(x, b8_hit_rates, width, label="Baseline LSH (b=8, r=8)", color="#ff7f0e")
    ax1.bar(x + width, lsh_hit_rates, width, label="Tuned LSH (b=16, r=4)", color="#2ca02c")
    ax1.set_ylabel("Hit@10 Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Hit@10 Success Rate by Quality Level\n(Robustness across degradation levels)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(qual_names, fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.grid(True, axis="y", ls="--", alpha=0.5)

    bf_zero_cnts = [bf_by_q[q]["zero_recall_count"] for q in quals]
    b8_zero_cnts = [b8_by_q[q]["zero_recall_count"] for q in quals]
    lsh_zero_cnts = [lsh_by_q[q]["zero_recall_count"] for q in quals]

    ax2.bar(x - width, bf_zero_cnts, width, label="Without LSH (Brute Force)", color="#1f77b4")
    ax2.bar(x, b8_zero_cnts, width, label="Baseline LSH (b=8, r=8)", color="#ff7f0e")
    ax2.bar(x + width, lsh_zero_cnts, width, label="Tuned LSH (b=16, r=4)", color="#2ca02c")
    ax2.set_ylabel("Zero-Recall Failure Count (out of 100 queries)", fontsize=11, fontweight="bold")
    ax2.set_title("Complete Retrieval Failures (Recall=0) by Quality Level\n(Where false negatives concentrate)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(qual_names, fontsize=10)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, axis="y", ls="--", alpha=0.5)

    fig.tight_layout()
    fig3_path = output_dir / "recall_by_quality_breakdown.png"
    fig.savefig(fig3_path, dpi=300)
    fig.savefig(artifact_fig_dir / "recall_by_quality_breakdown.png", dpi=300)
    plt.close(fig)
    print(f"      Saved: {fig3_path}")


def main():
    shards_dir = workspace_root / "dataset" / "signatures_shards"
    dataset_dir = workspace_root / "dataset"
    reports_dir = workspace_root / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      LSH TUNING, RECALL FORMULATION & PRECISION-CANDIDATE EXPERIMENTS")
    print("      Database: 20,000 Unique Fingerprint Identities (200,000 Gallery Records)")
    print("      Runtime probe signature generation enabled")
    print("=" * 80)

    engine = VectorizedMinHashEngine(num_hashes=64, seed=42)

    # 1. Load 20k dataset & select balanced queries
    gallery_sigs, gallery_ids, query_minutiae, query_ids, query_quals = load_20k_dataset(
        shards_dir=shards_dir, dataset_dir=dataset_dir, samples_per_quality=100, seed=42
    )
    print(f"[2/5] Selected {len(query_minutiae)} balanced test queries (100 from each quality: 3, 7, 5, 1, 9).")

    # 3. Run Experiment 1: Recall Formulation & Zero-Recall Analysis
    exp1_results = run_experiment_1_recall_analysis(
        gallery_sigs, gallery_ids, query_minutiae, query_ids, query_quals, engine
    )

    # 4. Run Experiment 2: Slider Tuning & Precision vs Candidates Curve
    exp2_grid, exp2_curve = run_experiment_2_slider_tuning_and_curves(
        gallery_sigs, gallery_ids, query_minutiae, query_ids, engine
    )

    # 5. Generate Figures
    generate_visualization_figures(
        exp1_results, exp2_grid["lsh_tuning_grid"], exp2_curve["candidate_budget_curve"], figures_dir
    )

    # 6. Save Complete Report
    full_report = {
        "metadata": {
            "dataset": "20,000 unique identities (shards 0 & 1)",
            "gallery_size": len(gallery_sigs),
            "probe_queries_evaluated": len(query_minutiae),
            "eval_distribution": "100 queries each for qualities [3, 7, 5, 1, 9]",
            "probe_signatures": "computed_at_runtime",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        },
        "experiment_1_recall_formulation_and_failures": exp1_results,
        "experiment_2_slider_tuning_grid": exp2_grid["lsh_tuning_grid"],
        "experiment_2_precision_candidate_curve": exp2_curve["candidate_budget_curve"]
    }

    report_path = reports_dir / "lsh_20k_experiments_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[5/5] Full benchmark report successfully saved to: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
