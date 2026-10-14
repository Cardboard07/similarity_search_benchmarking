"""
Large-Scale Fingerprint MinHash, LSH, and Bloom Filter Package.
"""

from src.lsh_scale.vectorized_minhash import (
    VectorizedMinHashEngine,
    make_hash_family,
    token_to_int
)
from src.lsh_scale.bloom_filter_6m import (
    ScaledIdentityBloomFilter,
    LargeBitArray,
    LSHBandBloomFilter
)
from src.lsh_scale.lsh_index_6m import ScaledLSHIndex
from src.lsh_scale.hdf5_signature_extractor import (
    process_single_shard,
    ShardedSignatureDataset
)

__all__ = [
    "VectorizedMinHashEngine",
    "make_hash_family",
    "token_to_int",
    "ScaledIdentityBloomFilter",
    "LargeBitArray",
    "LSHBandBloomFilter",
    "ScaledLSHIndex",
    "process_single_shard",
    "ShardedSignatureDataset"
]
