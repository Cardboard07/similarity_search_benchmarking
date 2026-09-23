"""
Unit tests verifying correctness and mathematical equivalence of the large-scale LSH pipeline.
"""

import math
import numpy as np
import pytest

from src.lsh_scale.vectorized_minhash import (
    VectorizedMinHashEngine,
    make_hash_family,
    token_to_int
)
from src.lsh_scale.bloom_filter_6m import ScaledIdentityBloomFilter
from src.lsh_scale.lsh_index_6m import ScaledLSHIndex
from LSH.scripts.parse import relative_tokens as baseline_relative_tokens
from LSH.main.min_hash import minhash_signature as baseline_minhash_signature


def test_token_extraction_equivalence():
    """
    Verifies that VectorizedMinHashEngine.extract_relative_tokens produces
    the EXACT same set of tokens as the baseline relative_tokens function.
    """
    np.random.seed(42)
    n = 45
    pts = np.random.uniform(50, 450, size=(n, 2))
    angles = np.random.uniform(0, 360, size=n)
    types = np.random.choice([1, 2], size=n)

    # Convert to baseline minutiae list of dicts
    baseline_minutiae = [
        {"x": float(pts[i, 0]), "y": float(pts[i, 1]), "angle": float(angles[i]), "type": int(types[i])}
        for i in range(n)
    ]

    engine = VectorizedMinHashEngine(k=2, dist_bin=10, dir_bin=20, orient_bin=20, include_type=True)
    vectorized_tokens = engine.extract_relative_tokens(pts, angles, types)
    baseline_tokens = baseline_relative_tokens(baseline_minutiae, k=2, dist_bin=10, dir_bin=20, orient_bin=20, include_type=True)

    assert vectorized_tokens == baseline_tokens
    assert len(vectorized_tokens) > 0


def test_minhash_signature_equivalence():
    """
    Verifies that VectorizedMinHashEngine.compute_signature produces
    the EXACT same signature as the baseline minhash_signature.
    """
    engine = VectorizedMinHashEngine(num_hashes=64, seed=42)
    
    # Generate arbitrary token set
    tokens = {
        (1, 2, 3, 1, 2),
        (5, 10, 15, 2, 1),
        (8, 4, 12, 1, 1),
        (20, 1, 7, 2, 2)
    }

    vec_sig = engine.compute_signature(tokens).tolist()
    base_sig = baseline_minhash_signature(tokens, engine.hash_funcs)

    assert vec_sig == base_sig


def test_bloom_filter_zero_false_negatives():
    """
    Verifies that ScaledIdentityBloomFilter achieves 0.00% False Negatives on inserted identities.
    """
    bf = ScaledIdentityBloomFilter(capacity=10000, error_rate=0.001)
    inserted_ids = [100000 + i for i in range(5000)]
    bf.add_all(inserted_ids)

    # Zero false negatives check: every single inserted identity must test True
    for ident in inserted_ids:
        assert bf.contains(ident) is True

    # Check non-existent queries have low FPR
    unseen_ids = [900000 + i for i in range(5000)]
    false_positives = sum(1 for uid in unseen_ids if bf.contains(uid))
    empirical_fpr = false_positives / len(unseen_ids)

    assert false_positives < 20  # Well below 1%
    assert bf.count == 5000


def test_bloom_filter_save_and_load(tmp_path):
    """
    Tests binary serialization and deserialization of ScaledIdentityBloomFilter.
    """
    bf = ScaledIdentityBloomFilter(capacity=1000, error_rate=0.001)
    test_ids = [1, 2, 3, 400000, 999999]
    bf.add_all(test_ids)

    save_fp = tmp_path / "test_bloom"
    bf.save(save_fp)

    loaded_bf = ScaledIdentityBloomFilter.load(save_fp)
    assert loaded_bf.capacity == bf.capacity
    assert loaded_bf.m == bf.m
    assert loaded_bf.k == bf.k
    assert loaded_bf.count == bf.count

    for tid in test_ids:
        assert loaded_bf.contains(tid) is True
    assert loaded_bf.contains(12345678) is False


def test_lsh_index_candidate_retrieval():
    """
    Verifies that ScaledLSHIndex correctly indexes signatures and retrieves true duplicates.
    """
    np.random.seed(42)
    index = ScaledLSHIndex(bands=8)

    # Create dummy database of 100 random signatures
    sigs = np.random.randint(0, 1000000, size=(100, 64), dtype=np.uint32)
    index.index_signatures(sigs)

    # Test exact duplicate: querying with sigs[10] must return index 10 in candidates
    query_sig = sigs[10].copy()
    candidates = index.find_candidates(query_sig)
    assert 10 in candidates

    # Top-1 scored candidate must be index 10 with similarity 1.0
    scores = index.score_candidates_vectorized(candidates, query_sig, sigs, k=5)
    assert len(scores) > 0
    top_idx, top_score = scores[0]
    assert top_idx == 10
    assert top_score == 1.0
