"""
Memory-Efficient LSH Inverted Index for Million-Scale Fingerprint Signatures.

Indexes integer sample IDs (uint32) across band buckets using sorted numpy
arrays for cache-friendly binary-search lookups.

Pre-filter:
    After indexing, a frozenset of all bucket keys is built per band.
    A query can be fast-rejected if NONE of its band hashes exist anywhere
    in the gallery — zero false negatives by construction (same guarantee
    as a Bloom filter, but derived entirely from the LSH structure itself,
    no separate data structure required).

Scoring:
    Similarity is computed as the fraction of MinHash positions that match
    exactly between query and candidate:  sim = mean(sig_a == sig_b)
    This is the direct MinHash theorem estimator of Jaccard similarity and
    does NOT use set-based Jaccard.
"""

from typing import List, Set, Tuple, Optional, Union
import numpy as np

from src.lsh_scale.bloom_filter_6m import LSHBandBloomFilter


class ScaledLSHIndex:
    """
    Locality-Sensitive Hashing index for 64-element MinHash signatures.
    """
    def __init__(self, bands: int = 16, rows: Optional[int] = None):
        self.bands: int = bands
        self.rows: int = rows if rows is not None else (64 // bands)
        if self.bands * self.rows > 64:
            raise ValueError(
                f"bands * rows ({self.bands * self.rows}) exceeds signature length 64."
            )

        # Sorted-array inverted index (built on index_signatures)
        self.sorted_keys: List[np.ndarray] = []
        self.sorted_indices: List[np.ndarray] = []

        # Pre-filter: LSH Band Bloom Filter
        self.band_bloom_filter: Optional[LSHBandBloomFilter] = None

        self.num_indexed: int = 0
        self._fnv_mult = int(0x517CC1B727220A95)

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_band_vectorized(band_slice: np.ndarray, mult: int) -> np.ndarray:
        """Hashes N rows of `rows` uint32 values into N uint64 keys."""
        n, r = band_slice.shape
        keys = np.zeros(n, dtype=np.uint64)
        for col in range(r):
            c_val = band_slice[:, col].astype(np.uint64)
            keys = (keys ^ c_val) * np.uint64(mult)
        return keys

    @staticmethod
    def _hash_band_single(band_slice: np.ndarray, mult: int) -> np.uint64:
        """Hashes a single row of `rows` uint32 values into one uint64 key."""
        key = 0
        mask = 0xFFFFFFFFFFFFFFFF
        for v in band_slice:
            key = ((key ^ int(v)) * mult) & mask
        return np.uint64(key)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_signatures(self, signatures: np.ndarray, start_idx: int = 0) -> None:
        """
        Indexes a full array of gallery signatures.

        Args:
            signatures: Shape (N, 64) uint32 array of MinHash signatures.
            start_idx:  Global index offset (for multi-batch indexing).
        """
        n_samples = len(signatures)
        rows = self.rows

        self.sorted_keys = []
        self.sorted_indices = []
        # Capacity estimate: unique keys across bands
        est_unique = max(100000, min(5000000, n_samples * self.bands // 4))
        self.band_bloom_filter = LSHBandBloomFilter(capacity=est_unique, error_rate=0.001)

        for b in range(self.bands):
            band_slice = signatures[:, b * rows: (b + 1) * rows]
            keys = self._hash_band_vectorized(band_slice, self._fnv_mult)

            order = np.argsort(keys)
            sorted_k = keys[order]
            sorted_i = (order + start_idx).astype(np.int32)

            self.sorted_keys.append(sorted_k)
            self.sorted_indices.append(sorted_i)
            # Add unique band keys to the Bloom Filter
            self.band_bloom_filter.add_band_keys(b, sorted_k)

        self.num_indexed += n_samples

    # ------------------------------------------------------------------
    # Pre-filter (LSH Band Bloom Filter)
    # ------------------------------------------------------------------

    def prefilter(self, query_sig: np.ndarray) -> bool:
        """
        Fast biometric pre-filter using the LSH Band Bloom Filter.

        Returns True  → at least one band key hits the Bloom filter; full
                        candidate lookup is warranted.
        Returns False → the query signature shares NO band bucket with any
                        gallery record in the Bloom filter; this is an early
                        non-match and the candidate search is skipped.
        """
        if self.band_bloom_filter is None:
            return True
        rows = self.rows
        for b in range(self.bands):
            b_slice = query_sig[b * rows: (b + 1) * rows]
            q_key = int(self._hash_band_single(b_slice, self._fnv_mult))
            if self.band_bloom_filter.contains(b, q_key):
                return True
        return False

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def find_candidates(self, query_sig: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Returns all gallery indices that share at least one band bucket
        with the query signature.
        """
        if not isinstance(query_sig, np.ndarray):
            query_sig = np.array(query_sig, dtype=np.uint32)

        candidates: List[int] = []
        seen: set = set()
        rows = self.rows

        for b in range(self.bands):
            b_slice = query_sig[b * rows: (b + 1) * rows]
            q_key = self._hash_band_single(b_slice, self._fnv_mult)
            keys_arr = self.sorted_keys[b]
            l = int(np.searchsorted(keys_arr, q_key, side='left'))
            r = int(np.searchsorted(keys_arr, q_key, side='right'))
            if l < r:
                for idx in self.sorted_indices[b][l:r]:
                    i = int(idx)
                    if i not in seen:
                        seen.add(i)
                        candidates.append(i)

        return candidates

    # ------------------------------------------------------------------
    # Scoring — fraction of matching positions (MinHash theorem estimator)
    # ------------------------------------------------------------------

    @staticmethod
    def score_candidates(
        candidate_indices: List[int],
        query_sig: np.ndarray,
        gallery_signatures: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Scores candidates using the direct MinHash estimator:
            sim(A, B) = mean(sig_A == sig_B)

        This is the unbiased estimator of Jaccard similarity given by the
        MinHash theorem:  E[sim(A, B)] = J(Tokens(A), Tokens(B)).

        Args:
            candidate_indices:  List of gallery row indices to score.
            query_sig:          Shape (64,) uint32 query signature.
            gallery_signatures: Shape (N, 64) uint32 gallery array.
            k:                  Number of top results to return.

        Returns:
            List of (gallery_index, similarity_score) sorted descending.
        """
        if not candidate_indices:
            return []

        cand_arr = np.array(candidate_indices, dtype=np.int64)
        cand_sigs = gallery_signatures[cand_arr]          # (C, 64)

        # Vectorized: fraction of hash positions that are identical
        matches = np.mean(cand_sigs == query_sig, axis=1)  # (C,)

        if len(cand_arr) > k:
            top_sub = np.argpartition(matches, -k)[-k:]
            top_sub = top_sub[np.argsort(-matches[top_sub])]
        else:
            top_sub = np.argsort(-matches)

        return [(int(cand_arr[i]), float(matches[i])) for i in top_sub]

    score_candidates_vectorized = score_candidates
