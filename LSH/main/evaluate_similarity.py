import os
import sys
import json
import time
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Ensure main directory is in sys.path when running from workspace root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lsh import find_cand, dicts
from min_hash import minhash_signature, hash_funcs
from similarity import jaccard, score_jaccard
from bloom_filter import build_fingerprint_bloom_filter, FingerprintBloomFilter


def resolve_path(p: str, default_rel: str) -> str:
    """Resolves path relative to CWD, script dir, or workspace root."""
    if os.path.exists(p):
        return p
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cand1 = os.path.join(script_dir, "..", p)
    if os.path.exists(cand1):
        return os.path.abspath(cand1)
    cand2 = os.path.join(script_dir, "..", default_rel)
    if os.path.exists(cand2):
        return os.path.abspath(cand2)
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    cand3 = os.path.join(workspace_root, default_rel)
    if os.path.exists(cand3):
        return os.path.abspath(cand3)
    return p


def extract_x2(filename_or_path: str) -> str:
    """
    Extracts the fingerprint identifier (x2) from filename structured as qx1_x2_x3.json
    Example: 'q1_100_1.json' -> '100'
    """
    basename = os.path.basename(filename_or_path).replace(".json", "")
    parts = basename.split("_")
    return parts[1] if len(parts) >= 2 else basename


def load_train_database(signatures_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """
    Loads train signatures and builds an index mapping x2 -> list of train file_names.
    """
    with open(signatures_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    db_x2_map: Dict[str, List[str]] = {}
    for item in data:
        fname = item["file_name"]
        x2 = extract_x2(fname)
        if x2 not in db_x2_map:
            db_x2_map[x2] = []
        db_x2_map[x2].append(fname)

    return data, db_x2_map


def evaluate(
    test_dir: str,
    signatures_path: str,
    top_k_values: List[int] = [1, 5, 10],
    max_docs: int = None,
    use_bloom_filter: bool = True,
    bloom_error_rate: float = 0.001,
    bloom_mode: str = "identity"
) -> Dict[str, Any]:
    """
    Runs benchmark on test files and computes timing, retrieval quality metrics,
    and Bloom Filter gating performance.
    """
    signatures_path = resolve_path(signatures_path, os.path.join("LSH", "data_structure", "signatures.json"))
    test_dir = resolve_path(test_dir, os.path.join("fingerprint_dataset", "parsed_data", "test"))

    train_data, db_x2_map = load_train_database(signatures_path)

    test_files = glob.glob(os.path.join(test_dir, "*.json"))
    test_files.sort()

    if max_docs is not None and max_docs > 0:
        test_files = test_files[:max_docs]

    total_queries = len(test_files)
    max_k = max(top_k_values)

    print(f"Starting evaluation on {total_queries} test documents...")

    # Initialize Bloom Filter
    bloom_filter_mgr = None
    if use_bloom_filter:
        print(f"Building Bloom Filter (target FPR = {bloom_error_rate})...")
        bloom_filter_mgr = build_fingerprint_bloom_filter(signatures_path, error_rate=bloom_error_rate)

    # Metrics accumulators
    total_bloom_time = 0.0
    total_minhash_time = 0.0
    total_lsh_time = 0.0
    total_jaccard_time = 0.0
    total_query_time = 0.0

    hit_at_k = {k: 0 for k in top_k_values}
    precision_at_k = {k: 0.0 for k in top_k_values}
    recall_at_k = {k: 0.0 for k in top_k_values}

    queries_with_gt = 0
    lsh_candidate_recall = 0
    total_lsh_candidates = 0

    # Bloom filter specific metrics
    bf_screened = 0
    bf_passed = 0
    bf_pruned = 0
    false_negatives = 0

    reciprocal_ranks = []
    avg_precisions = []

    start_eval_time = time.time()

    for file_path in tqdm(test_files, desc="Evaluating Queries"):
        q_start = time.time()

        with open(file_path, "r", encoding="utf-8") as f:
            doc_raw = json.load(f)

        q_name = doc_raw.get("file_name", os.path.basename(file_path))
        q_x2 = extract_x2(q_name)

        # Ground truth in train DB
        gt_files = db_x2_map.get(q_x2, [])
        num_gt = len(gt_files)
        if num_gt > 0:
            queries_with_gt += 1

        # 1. Bloom Filter Pre-Check
        doc_record = {"file_name": q_name}
        passed_bf = True
        if use_bloom_filter and bloom_filter_mgr:
            t_bf0 = time.time()
            passed_bf, _ = bloom_filter_mgr.check_fingerprint(doc_record, mode=bloom_mode)
            t_bf1 = time.time()
            total_bloom_time += (t_bf1 - t_bf0)
            bf_screened += 1

            if not passed_bf:
                bf_pruned += 1
                # Check for false negative (if GT existed in DB but Bloom filter said NO)
                if num_gt > 0:
                    false_negatives += 1
                total_query_time += (time.time() - q_start)
                reciprocal_ranks.append(0.0)
                avg_precisions.append(0.0)
                continue
            else:
                bf_passed += 1

        # 2. Compute MinHash signature
        t0 = time.time()
        doc = {
            "file_name": q_name,
            "signatures": minhash_signature(doc_raw["relative_tokens"], hash_funcs)
        }
        t1 = time.time()
        total_minhash_time += (t1 - t0)

        # 3. Retrieve LSH candidates
        candidates = find_cand(dicts, doc)
        t2 = time.time()
        total_lsh_time += (t2 - t1)

        # 4. Score candidates with Jaccard similarity
        results = score_jaccard(candidates, doc, k=max_k)
        t3 = time.time()
        total_jaccard_time += (t3 - t2)

        total_query_time += (t3 - q_start)
        total_lsh_candidates += len(candidates)

        # Check LSH recall (did LSH return at least 1 match?)
        lsh_has_match = any(extract_x2(c["file_name"]) == q_x2 for c in candidates)
        if lsh_has_match:
            lsh_candidate_recall += 1

        # Evaluate top-K rankings
        hit_ranks = []
        for idx, (cand_fname, sim_score) in enumerate(results):
            cand_x2 = extract_x2(cand_fname)
            if cand_x2 == q_x2:
                hit_ranks.append(idx + 1)

        # MRR calculation (1-indexed rank of first hit)
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

        # Compute K-specific metrics
        for k in top_k_values:
            k_results = results[:k]
            hits = [1 for fname, _ in k_results if extract_x2(fname) == q_x2]
            num_hits = len(hits)

            if num_hits > 0:
                hit_at_k[k] += 1

            precision_at_k[k] += num_hits / k if k > 0 else 0.0

            if num_gt > 0:
                recall_at_k[k] += num_hits / num_gt

    end_eval_time = time.time()
    total_eval_time = end_eval_time - start_eval_time

    # Aggregate metric calculations
    avg_latency_ms = (total_query_time / total_queries) * 1000 if total_queries > 0 else 0.0
    throughput_qps = total_queries / total_eval_time if total_eval_time > 0 else 0.0
    avg_candidates = total_lsh_candidates / total_queries if total_queries > 0 else 0.0

    mean_reciprocal_rank = sum(reciprocal_ranks) / total_queries if total_queries > 0 else 0.0
    mean_avg_precision = sum(avg_precisions) / total_queries if total_queries > 0 else 0.0

    lsh_recall_pct = (lsh_candidate_recall / queries_with_gt * 100.0) if queries_with_gt > 0 else 0.0

    metrics = {
        "dataset_summary": {
            "total_queries_evaluated": total_queries,
            "queries_with_ground_truth_in_db": queries_with_gt,
            "queries_without_ground_truth_in_db": total_queries - queries_with_gt,
            "train_database_size": len(train_data)
        },
        "bloom_filter_summary": {
            "enabled": use_bloom_filter,
            "mode": bloom_mode,
            "queries_screened": bf_screened,
            "queries_passed_filter": bf_passed,
            "queries_pruned_before_search": bf_pruned,
            "false_negatives": false_negatives,
            "false_negative_rate_pct": (false_negatives / queries_with_gt * 100.0) if queries_with_gt > 0 else 0.0,
            "avg_bloom_latency_ms": round((total_bloom_time / total_queries) * 1000, 4) if total_queries > 0 else 0.0,
            "parameters": {
                "identity_filter": bloom_filter_mgr.identity_filter.get_stats() if bloom_filter_mgr else None,
                "record_filter": bloom_filter_mgr.record_filter.get_stats() if bloom_filter_mgr and bloom_filter_mgr.record_filter else None,
                "band_filter": bloom_filter_mgr.band_filter.get_stats() if bloom_filter_mgr and bloom_filter_mgr.band_filter else None
            }
        },
        "performance_timing": {
            "total_evaluation_time_sec": round(total_eval_time, 3),
            "avg_query_latency_ms": round(avg_latency_ms, 2),
            "throughput_queries_per_sec": round(throughput_qps, 2),
            "breakdown_ms_per_query": {
                "bloom_filter_gating": round((total_bloom_time / total_queries) * 1000, 4) if total_queries > 0 else 0.0,
                "minhash_signature": round((total_minhash_time / total_queries) * 1000, 2) if total_queries > 0 else 0.0,
                "lsh_candidate_search": round((total_lsh_time / total_queries) * 1000, 2) if total_queries > 0 else 0.0,
                "jaccard_ranking": round((total_jaccard_time / total_queries) * 1000, 2) if total_queries > 0 else 0.0
            }
        },
        "lsh_index_efficiency": {
            "avg_candidates_per_query": round(avg_candidates, 2),
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

    return metrics


def print_report(metrics: Dict[str, Any]) -> None:
    """Prints a formatted evaluation report."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("        DUPLICATE DETECTION BENCHMARK & EVALUATION REPORT       ")
    print("=" * 70)

    ds = metrics["dataset_summary"]
    print("\n--- DATASET SUMMARY ---")
    print(f"Total Test Queries Evaluated     : {ds['total_queries_evaluated']:,}")
    print(f"Queries with Ground Truth in DB : {ds['queries_with_ground_truth_in_db']:,}")
    print(f"Queries without Target in DB    : {ds['queries_without_ground_truth_in_db']:,}")
    print(f"Train Database Size (Records)   : {ds['train_database_size']:,}")

    bfs = metrics.get("bloom_filter_summary", {})
    if bfs.get("enabled"):
        print("\n--- BLOOM FILTER PRE-SEARCH GATING ---")
        print(f"Bloom Filter Status             : ACTIVE (Mode: {bfs.get('mode')})")
        print(f"Queries Screened                : {bfs.get('queries_screened'):,}")
        print(f"Queries Passed (Committed)      : {bfs.get('queries_passed_filter'):,}")
        print(f"Queries Pruned (Zero-Cost Skip) : {bfs.get('queries_pruned_before_search'):,}")
        print(f"False Negatives                 : {bfs.get('false_negatives')} ({bfs.get('false_negative_rate_pct'):.2f}%)")
        print(f"Avg Bloom Filter Latency        : {bfs.get('avg_bloom_latency_ms', 0):.4f} ms / query")
        
        bf_params = bfs.get("parameters", {}).get("identity_filter")
        if bf_params:
            print("\n  [Identity Bloom Filter Parameters]")
            print(f"  |- Capacity (n)               : {bf_params['capacity']:,} identities")
            print(f"  |- Target False Positive Rate : {bf_params['target_error_rate']}")
            print(f"  |- Bit Array Size (m)         : {bf_params['bit_array_size_bits (m)']:,} bits ({bf_params['bit_array_size_kb']} KB)")
            print(f"  |- Number of Hash Functions(k): {bf_params['num_hash_functions (k)']}")
            print(f"  |- Bits Set (1s)              : {bf_params['bits_set_ones']:,} (Fill Ratio: {bf_params['bit_fill_ratio'] * 100:.2f}%)")
            print(f"  \\- Theoretical FPR            : {bf_params['theoretical_fpr']:.6f}")

    pt = metrics["performance_timing"]
    print("\n--- RUNTIME & PERFORMANCE ---")
    print(f"Total Evaluation Time           : {pt['total_evaluation_time_sec']:.2f} seconds")
    print(f"Average Query Latency           : {pt['avg_query_latency_ms']:.2f} ms / query")
    print(f"Search Throughput               : {pt['throughput_queries_per_sec']:.2f} queries / sec")
    bd = pt["breakdown_ms_per_query"]
    if "bloom_filter_gating" in bd:
        print(f"  |- Bloom Filter Gating Time   : {bd['bloom_filter_gating']:.4f} ms")
    print(f"  |- MinHash Signature Time     : {bd['minhash_signature']:.2f} ms")
    print(f"  |- LSH Candidate Search Time  : {bd['lsh_candidate_search']:.2f} ms")
    print(f"  \\- Jaccard Ranking Time       : {bd['jaccard_ranking']:.2f} ms")

    le = metrics["lsh_index_efficiency"]
    print("\n--- LSH INDEX EFFICIENCY ---")
    print(f"Avg Candidates Retained / Query : {le['avg_candidates_per_query']:.2f} docs (out of {ds['train_database_size']} total)")
    print(f"LSH Candidate Recall            : {le['lsh_candidate_recall_pct']:.2f}% (true matches caught by LSH)")

    ra = metrics["retrieval_accuracy"]
    print("\n--- RETRIEVAL ACCURACY METRICS ---")
    print(f"Mean Reciprocal Rank (MRR)      : {ra['mrr']:.4f}")
    print(f"Mean Average Precision (MAP@10)  : {ra['map_at_10']:.4f}")
    print("\nHit Rates (Top-K Accuracy):")
    for k_name, val in ra["hit_rate_pct"].items():
        print(f"  |- {k_name:8s}: {val:.2f}%")

    print("\nPrecision & Recall:")
    for k in [1, 5, 10]:
        p_val = ra["precision"].get(f"P@{k}", 0.0)
        r_val = ra["recall"].get(f"R@{k}", 0.0)
        print(f"  |- Top-{k:2d}: Precision@{k} = {p_val:.4f}  |  Recall@{k} = {r_val:.4f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate duplicate detection similarity on test dataset.")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.path.join("fingerprint_dataset", "parsed_data", "test"),
        help="Path to directory containing test json files."
    )
    parser.add_argument(
        "--signatures",
        type=str,
        default=os.path.join("LSH", "data_structure", "signatures.json"),
        help="Path to train dataset signatures json."
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional maximum number of test docs to evaluate (for fast testing)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Path to save output JSON metrics file."
    )
    parser.add_argument(
        "--bloom-mode",
        type=str,
        default="identity",
        choices=["identity", "band", "hybrid"],
        help="Bloom filter gating mode."
    )
    parser.add_argument(
        "--bloom-fpr",
        type=float,
        default=0.001,
        help="Target false positive rate for Bloom filter."
    )
    parser.add_argument(
        "--no-bloom",
        action="store_true",
        help="Disable Bloom filter pre-checking."
    )

    args = parser.parse_args()

    metrics = evaluate(
        test_dir=args.test_dir,
        signatures_path=args.signatures,
        top_k_values=[1, 5, 10],
        max_docs=args.max_docs,
        use_bloom_filter=not args.no_bloom,
        bloom_error_rate=args.bloom_fpr,
        bloom_mode=args.bloom_mode
    )

    print_report(metrics)

    output_path = resolve_path(args.output, os.path.join("LSH", args.output))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved benchmark results to: {output_path}")

