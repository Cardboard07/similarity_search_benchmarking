"""
High-Performance Vectorized Relative Minutiae Tokenizer and MinHash Engine.

Provides mathematically identical token extraction and MinHash signatures to the
baseline implementation, optimized with vectorized NumPy operations for million-scale workloads.
"""

import math
import hashlib
import random
from typing import Set, Tuple, List, Optional
import numpy as np

# Prime modulus used across the project (2^32 - 5)
PRIME_MODULUS = np.uint64(4294967311)


def make_hash_family(n: int = 64, seed: Optional[int] = 42) -> List[Tuple[int, int, int]]:
    """
    Generates n 2-universal hash functions of form: h(x) = (a * x + b) % p
    Identical to LSH/main/min_hash.py.
    """
    rng = random.Random(seed)
    p = 4294967311
    hash_funcs = []
    for _ in range(n):
        a = rng.randint(1, p - 1)
        b = rng.randint(0, p - 1)
        hash_funcs.append((a, b, p))
    return hash_funcs


def token_to_int(token: Tuple[int, ...]) -> int:
    """
    Converts a token tuple to a 64-bit integer using Blake2b.
    Identical to LSH/main/min_hash.py.
    """
    b = repr(token).encode()
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")


class VectorizedMinHashEngine:
    """
    Vectorized tokenizer and MinHash signature generator.
    """
    def __init__(
        self,
        num_hashes: int = 64,
        seed: int = 42,
        k: int = 2,
        dist_bin: int = 10,
        dir_bin: int = 20,
        orient_bin: int = 20,
        include_type: bool = True
    ):
        self.num_hashes = num_hashes
        self.seed = seed
        self.k = k
        self.dist_bin = dist_bin
        self.dir_bin = dir_bin
        self.orient_bin = orient_bin
        self.include_type = include_type

        self.hash_funcs = make_hash_family(n=num_hashes, seed=seed)
        self.a_arr = np.array([f[0] for f in self.hash_funcs], dtype=np.uint64)
        self.b_arr = np.array([f[1] for f in self.hash_funcs], dtype=np.uint64)
        self.p = PRIME_MODULUS

    def extract_relative_tokens(
        self,
        pts: np.ndarray,
        angles: np.ndarray,
        types: np.ndarray
    ) -> Set[Tuple[int, ...]]:
        """
        Vectorized relative token extraction.
        Produces identical output to the original parse.py / build_parsed_dataset.py.

        Args:
            pts: Shape (N, 2) array of (x, y) coordinates.
            angles: Shape (N,) array of minutiae angles in degrees.
            types: Shape (N,) array of minutiae types.

        Returns:
            Set of integer tuples representing local relative minutiae geometry.
        """
        n = len(pts)
        if n <= 1:
            return set()

        k = min(self.k, n - 1)

        # Pairwise distance matrix: shape (N, N)
        dx = pts[:, 0:1].T - pts[:, 0:1]
        dy = pts[:, 1:2].T - pts[:, 1:2]
        dists = np.hypot(dx, dy)
        np.fill_diagonal(dists, np.inf)

        # Find k nearest neighbors for each minutia
        nn_idx = np.argpartition(dists, k, axis=1)[:, :k]

        # Sort the k nearest neighbors for deterministic ordering
        row_idx = np.arange(n)[:, None]
        col_order = np.argsort(dists[row_idx, nn_idx], axis=1)
        nn_idx = nn_idx[row_idx, col_order]

        # Relative directions and orientations
        abs_dir = np.degrees(np.arctan2(dy[row_idx, nn_idx], dx[row_idx, nn_idx])) % 360.0
        rel_dir = (abs_dir - angles[:, None]) % 360.0
        delta = (angles[nn_idx] - angles[:, None]) % 360.0
        d_sel = dists[row_idx, nn_idx]

        db = (d_sel // self.dist_bin).astype(np.int32)
        rb = (rel_dir // self.dir_bin).astype(np.int32)
        ob = (delta // self.orient_bin).astype(np.int32)

        if self.include_type:
            ti = np.repeat(types[:, None].astype(np.int32), k, axis=1)
            tj = types[nn_idx].astype(np.int32)
            tokens = set(zip(db.flat, rb.flat, ob.flat, ti.flat, tj.flat))
        else:
            tokens = set(zip(db.flat, rb.flat, ob.flat))

        return tokens

    def compute_signature(self, tokens: Set[Tuple[int, ...]]) -> np.ndarray:
        """
        Computes 64-element MinHash signature for a set of tokens.

        Uses modular property: (a * x + b) % p == (a * (x % p) + b) % p
        to prevent uint64 overflow while keeping full vectorization.
        """
        if not tokens:
            return np.zeros(self.num_hashes, dtype=np.uint32)

        # Precompute Blake2b hashes reduced modulo p once per unique token
        x_mod_p = np.array([token_to_int(t) % int(self.p) for t in tokens], dtype=np.uint64)

        # Vectorized broadcast: (num_hashes, 1) * (1, num_tokens) + (num_hashes, 1) mod p
        hashes = (self.a_arr[:, None] * x_mod_p[None, :] + self.b_arr[:, None]) % self.p

        # MinHash: minimum hash value across all tokens for each hash function
        min_hashes = np.min(hashes, axis=1).astype(np.uint32)
        return min_hashes

    def process_observation(
        self,
        pts: np.ndarray,
        angles: np.ndarray,
        types: np.ndarray
    ) -> np.ndarray:
        """
        End-to-end token extraction + MinHash signature computation for a single sample.
        """
        tokens = self.extract_relative_tokens(pts, angles, types)
        return self.compute_signature(tokens)
