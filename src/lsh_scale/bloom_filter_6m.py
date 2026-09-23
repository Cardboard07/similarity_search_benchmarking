"""
Million-Scale Bloom Filter for 400,000 Fingerprint Identities.

Provides sub-microsecond pre-search screening to reject queries of non-existent identities
with guaranteed zero false negatives and <= 0.1% false positive probability.
"""

import os
import math
import json
import hashlib
from pathlib import Path
from typing import Any, Iterable, List, Tuple, Union, Optional


class LargeBitArray:
    """
    Memory-efficient bit array backed by Python's bytearray.
    1 byte = 8 bits. Fast bit set, get, and population count.
    """
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("LargeBitArray size must be positive.")
        self.size: int = size
        self.num_bytes: int = (size + 7) // 8
        self._array: bytearray = bytearray(self.num_bytes)

    def set(self, index: int) -> None:
        """Sets the bit at index to 1."""
        self._array[index >> 3] |= (1 << (index & 7))

    def get(self, index: int) -> bool:
        """Returns True if the bit at index is 1, else False."""
        return (self._array[index >> 3] & (1 << (index & 7))) != 0

    def count_ones(self) -> int:
        """Returns the total number of set bits (1s) in the bit array."""
        return sum(bin(b).count("1") for b in self._array)

    def to_bytes(self) -> bytes:
        return bytes(self._array)

    @classmethod
    def from_bytes(cls, data: bytes, size: int) -> "LargeBitArray":
        arr = cls(size)
        if len(data) != arr.num_bytes:
            raise ValueError(f"Byte length mismatch: expected {arr.num_bytes}, got {len(data)}")
        arr._array = bytearray(data)
        return arr

    def __len__(self) -> int:
        return self.size


class ScaledIdentityBloomFilter:
    """
    Bloom filter configured for large-scale identity pre-screening.
    Uses Kirsch-Mitzenmacher optimization with Blake2b double-hashing:
        h_i(x) = (h1(x) + i * h2(x)) % m
    """
    def __init__(
        self,
        capacity: int = 400000,
        error_rate: float = 0.001,
        num_bits: Optional[int] = None,
        num_hashes: Optional[int] = None
    ):
        self.capacity: int = capacity
        self.target_error_rate: float = error_rate

        if num_bits is not None and num_hashes is not None:
            self.m: int = num_bits
            self.k: int = num_hashes
        else:
            # Optimal m = - (n * ln(p)) / (ln(2)^2)
            self.m = int(math.ceil(- (capacity * math.log(error_rate)) / (math.log(2) ** 2)))
            # Optimal k = (m / n) * ln(2)
            self.k = int(round((self.m / capacity) * math.log(2)))
            if self.k < 1:
                self.k = 1

        self.bit_array: LargeBitArray = LargeBitArray(self.m)
        self.count: int = 0

    def _get_hashes(self, item_bytes: bytes) -> List[int]:
        """
        Derives k independent hash indices in [0, m-1] using double hashing.
        """
        h1 = int.from_bytes(hashlib.blake2b(item_bytes, digest_size=8, salt=b"bloom_h1").digest(), "big")
        h2 = int.from_bytes(hashlib.blake2b(item_bytes, digest_size=8, salt=b"bloom_h2").digest(), "big")
        if h2 == 0:
            h2 = 1

        m = self.m
        return [(h1 + i * h2) % m for i in range(self.k)]

    def _to_bytes(self, item: Any) -> bytes:
        if isinstance(item, bytes):
            return item
        if isinstance(item, (int, float)):
            return str(item).encode("utf-8")
        if isinstance(item, str):
            return item.encode("utf-8")
        return str(item).encode("utf-8")

    def add(self, identity: Any) -> None:
        """Inserts an identity ID into the Bloom filter."""
        b = self._to_bytes(identity)
        for idx in self._get_hashes(b):
            self.bit_array.set(idx)
        self.count += 1

    def add_all(self, identities: Iterable[Any]) -> None:
        """Bulk inserts identities."""
        for ident in identities:
            self.add(ident)

    def contains(self, identity: Any) -> bool:
        """
        Returns False if the identity is DEFINITIVELY NOT in the gallery (0 false negatives).
        Returns True if the identity is PROBABLY in the gallery.
        """
        b = self._to_bytes(identity)
        for idx in self._get_hashes(b):
            if not self.bit_array.get(idx):
                return False
        return True

    def __contains__(self, identity: Any) -> bool:
        return self.contains(identity)

    def current_false_positive_rate(self) -> float:
        """Calculates theoretical FPR from fraction of 1s in the bit array."""
        set_bits = self.bit_array.count_ones()
        return (set_bits / self.m) ** self.k

    def get_stats(self) -> dict:
        set_bits = self.bit_array.count_ones()
        return {
            "capacity": self.capacity,
            "items_inserted": self.count,
            "target_error_rate": self.target_error_rate,
            "bit_array_size_bits (m)": self.m,
            "bit_array_size_bytes": self.bit_array.num_bytes,
            "bit_array_size_kb": round(self.bit_array.num_bytes / 1024, 2),
            "num_hash_functions (k)": self.k,
            "bits_set_ones": set_bits,
            "bit_fill_ratio": round(set_bits / self.m, 4),
            "theoretical_fpr": round(self.current_false_positive_rate(), 6),
            "bits_per_item": round(self.m / self.capacity, 2)
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves metadata (.json) and bit array bytes (.bin)."""
        fp = Path(filepath)
        meta_path = fp.with_suffix(".json")
        bin_path = fp.with_suffix(".bin")

        metadata = {
            "capacity": self.capacity,
            "target_error_rate": self.target_error_rate,
            "m": self.m,
            "k": self.k,
            "count": self.count
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(bin_path, "wb") as f:
            f.write(self.bit_array.to_bytes())

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ScaledIdentityBloomFilter":
        fp = Path(filepath)
        meta_path = fp.with_suffix(".json")
        bin_path = fp.with_suffix(".bin")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        bf = cls(
            capacity=meta["capacity"],
            error_rate=meta["target_error_rate"],
            num_bits=meta["m"],
            num_hashes=meta["k"]
        )
        bf.count = meta["count"]

        with open(bin_path, "rb") as f:
            data = f.read()

        bf.bit_array = LargeBitArray.from_bytes(data, meta["m"])
        return bf
