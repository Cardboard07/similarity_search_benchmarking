import os
import sys
import json
from pathlib import Path

# Ensure directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lsh import find_cand, dicts, write_f
from min_hash import minhash_signature, hash_funcs
from bloom_filter import build_fingerprint_bloom_filter, FingerprintBloomFilter

# Default query path
_workspace_dir = Path(__file__).resolve().parent.parent.parent
doc = str(_workspace_dir / "fingerprint_dataset" / "parsed_data" / "test" / "q1_1_2.json")

_bloom_filter_cache = None


def get_bloom_filter():
    """Lazily initializes and caches the Bloom filter over the database."""
    global _bloom_filter_cache
    if _bloom_filter_cache is None:
        _bloom_filter_cache = build_fingerprint_bloom_filter(write_f, error_rate=0.001)
    return _bloom_filter_cache


def jaccard(a, b):
    set_a, set_b = set(a), set(b)
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(intersection) / len(union) if union else 0.0


def score_jaccard(candidates, doc, k):
    similar = {}
    for c in candidates:
        similar[c['file_name']] = jaccard(c['signatures'], doc['signatures'])
    sorted_similar = sorted(similar.items(), key=lambda x: x[1], reverse=True)
    return sorted_similar[:k]


def run_sim(dicts=dicts, doc=doc, k=10, bloom_filter=None, mode="identity"):
    """
    Executes fingerprint retrieval with Bloom Filter pre-checking.
    
    Before committing to the LSH candidate search and Jaccard scoring,
    the Bloom Filter checks if the fingerprint exists in the database.
    If not in database, the search is aborted immediately, saving query time.
    """
    if bloom_filter is None:
        bloom_filter = get_bloom_filter()
    if isinstance(doc, (str, Path)):
        with open(doc, "r", encoding="utf-8") as file_obj:
            data_dict = json.load(file_obj)
    else:
        data_dict = doc
    
    doc_record = {
        'file_name': data_dict['file_name'],
        'signatures': minhash_signature(data_dict['relative_tokens'], hash_funcs)
    }
    
    print(f"\n--- Fingerprint Query: {doc_record['file_name']} ---")
    
    # 1. Bloom Filter Pre-Check
    is_present, status_msg = bloom_filter.check_fingerprint(doc_record, mode=mode)
    print(f"[Bloom Filter Check] Result: {'IN DATABASE' if is_present else 'NOT IN DATABASE'}")
    print(f"                     Detail: {status_msg}")
    
    if not is_present:
        print("[Bloom Filter Pruning] Fingerprint is DEFINITIVELY NOT in the database.")
        print("[Bloom Filter Pruning] Skipped LSH search & Jaccard scoring (zero search cost).")
        return []
    
    # 2. Commit to LSH search
    print("[Search Committed] Querying LSH candidate index...")
    candidates = find_cand(dicts, doc_record)
    print(f"[Search Committed] Found {len(candidates)} candidate match(es) from LSH bands.")
    
    # 3. Jaccard similarity scoring
    result = score_jaccard(candidates, doc_record, k=k)
    print(f"[Results] Top-{k} most similar fingerprints in database:")
    for (key, val) in result:
        print(f"  |- {key:20s} (Jaccard Similarity: {val:.4f})")
    
    return result


if __name__ == "__main__":
    # Test with standard query
    run_sim()

    # Test with a non-existent / negative query
    negative_doc = {
        'file_name': 'q1_99999_1.json',
        'relative_tokens': [[1, 2, 3, 4, 5]]
    }
    print("\n" + "=" * 50)
    print("Testing Bloom Filter rejection on unknown fingerprint query:")
    run_sim(doc=negative_doc)

