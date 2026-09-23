"""
Memory-Efficient LSH Inverted Index for Million-Scale Fingerprint Signatures.

Indexes integer sample IDs (uint32) across band buckets to maintain a compact
RAM footprint (< 450 MB for 4,000,000 gallery records).
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class ScaledLSHIndex:
    """
    Locality-Sensitive Hashing index for 64-element MinHash signatures.
    """
    def __init__(self, bands: int = 8, rows: Optional[int] = None):
        self.bands: int = bands
        self.rows: int = rows if rows is not None else (64 // bands)
        if self.bands * self.rows > 64:
            raise ValueError(f"bands * rows ({self.bands * self.rows}) exceeds signature length 64.")

        # Inverted index: list of dictionaries, one per band
        # band_tables[i]: maps band_tuple -> list of int32 record indices
        self.band_tables: List[Dict[Tuple[int, ...], List[int]]] = [{} for _ in range(self.bands)]
        self.num_indexed: int = 0

    def index_signatures(self, signatures: np.ndarray, start_idx: int = 0) -> None:
        """
        Indexes a batch or full array of signatures.
        signatures: Shape (N, 64) array of uint32 signatures.
        start_idx: Global offset index for the batch.
        """
        n_samples = len(signatures)
        rows = self.rows

        for b in range(self.bands):
            table = self.band_tables[b]
            band_slice = signatures[:, b * rows : (b + 1) * rows]
            for i in range(n_samples):
                key = tuple(band_slice[i])
                idx = start_idx + i
                if key in table:
                    table[key].append(idx)
                else:
                    table[key] = [idx]

        self.num_indexed += n_samples

    def find_candidates(self, query_sig: Union[np.ndarray, List[int]]) -> List[int]:
        """
        Finds all candidate gallery indices that share at least one band bucket with query_sig.
        """
        if isinstance(query_sig, np.ndarray):
            sig = query_sig
        else:
            sig = np.array(query_sig, dtype=np.uint32)

        candidates = []
        seen = set()
        rows = self.rows

        for b in range(self.bands):
            table = self.band_tables[b]
            key = tuple(sig[b * rows : (b + 1) * rows])
            bucket = table.get(key)
            if bucket:
                for idx in bucket:
                    if idx not in seen:
                        seen.add(idx)
                        candidates.append(idx)

        return candidates

    @staticmethod
    def compute_jaccard_set(sig_a: Union[np.ndarray, List[int]], sig_b: Union[np.ndarray, List[int]]) -> float:
        """
        Exact set Jaccard matching LSH/main/similarity.py.
        """
        sa, sb = set(sig_a), set(sig_b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union > 0 else 0.0

    @staticmethod
    def score_candidates_vectorized(
        candidate_indices: List[int],
        query_sig: np.ndarray,
        gallery_signatures: np.ndarray,
        k: int = 10,
        use_set_jaccard: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Scores candidate records against query signature and returns top-k (idx, score).
        """
        if not candidate_indices:
            return []

        cand_arr = np.array(candidate_indices, dtype=np.int64)
        cand_sigs = gallery_signatures[cand_arr]  # Shape (C, 64)

        if use_set_jaccard:
            # Set-based Jaccard identical to baseline
            q_set = set(query_sig)
            scores = []
            for i, c_idx in enumerate(cand_arr):
                c_set = set(cand_sigs[i])
                inter = len(q_set & c_set)
                union = len(q_set | c_set)
                sim = inter / union if union > 0 else 0.0
                scores.append((int(c_idx), sim))
        else:
            # MinHash theorem direct fraction of equal hash values: P(h(A)==h(B)) = J(A,B)
            matches = np.mean(cand_sigs == query_sig, axis=1)
            scores = [(int(cand_arr[i]), float(matches[i])) for i in range(len(cand_arr))]

        # Sort descending by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
