import os
import math
import json
import hashlib
from typing import Any, Iterable, List, Tuple, Union, Optional
from pathlib import Path


class BitArray:
    """
    A memory-efficient bit array implementation backed by Python's bytearray.
    Each byte stores 8 bits. Provides fast bit-level set, get, and population count operations.
    """
    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("BitArray size must be positive.")
        self.size: int = size
        self.num_bytes: int = (size + 7) // 8
        self._array: bytearray = bytearray(self.num_bytes)

    def set(self, index: int) -> None:
        """Sets the bit at the given index to 1."""
        if not (0 <= index < self.size):
            raise IndexError(f"Index {index} out of range for BitArray of size {self.size}")
        byte_idx = index >> 3
        bit_mask = 1 << (index & 7)
        self._array[byte_idx] |= bit_mask

    def get(self, index: int) -> bool:
        """Returns True if the bit at index is 1, else False."""
        if not (0 <= index < self.size):
            raise IndexError(f"Index {index} out of range for BitArray of size {self.size}")
        byte_idx = index >> 3
        bit_mask = 1 << (index & 7)
        return (self._array[byte_idx] & bit_mask) != 0

    def test(self, index: int) -> bool:
        """Alias for get(index)."""
        return self.get(index)

    def clear(self) -> None:
        """Resets all bits in the array to 0."""
        self._array = bytearray(self.num_bytes)

    def count_ones(self) -> int:
        """Returns the total number of set bits (1s) in the bit array."""
        return sum(bin(b).count("1") for b in self._array)

    def to_bytes(self) -> bytes:
        """Returns raw bytes representation."""
        return bytes(self._array)

    @classmethod
    def from_bytes(cls, data: bytes, size: int) -> "BitArray":
        """Reconstructs a BitArray from raw bytes and bit size."""
        bit_arr = cls(size)
        if len(data) != bit_arr.num_bytes:
            raise ValueError(f"Data length ({len(data)} bytes) does not match expected ({bit_arr.num_bytes} bytes).")
        bit_arr._array = bytearray(data)
        return bit_arr

    def __len__(self) -> int:
        return self.size


class BloomFilter:
    """
    A space-efficient probabilistic data structure used to test set membership.
    False positive matches are possible with probability p, but false negatives are never possible.

    Uses the Kirsch-Mitzenmacher optimization to derive k independent hash functions
    from two 64-bit cryptographic hash digests:
        h_i(x) = (h1(x) + i * h2(x)) % m
    """
    def __init__(
        self,
        capacity: Optional[int] = None,
        error_rate: float = 0.01,
        num_bits: Optional[int] = None,
        num_hashes: Optional[int] = None
    ):
        """
        Initialize BloomFilter.
        Either provide (capacity, error_rate) to compute optimal m and k,
        or provide explicit (num_bits, num_hashes).
        """
        if num_bits is not None and num_hashes is not None:
            self.m: int = num_bits
            self.k: int = num_hashes
            self.capacity: Optional[int] = capacity
            self.target_error_rate: Optional[float] = error_rate
        elif capacity is not None and capacity > 0:
            if not (0.0 < error_rate < 1.0):
                raise ValueError("error_rate must be strictly between 0 and 1.")
            self.capacity = capacity
            self.target_error_rate = error_rate
            # Optimal size: m = - (n * ln(p)) / (ln(2)^2)
            self.m = int(math.ceil(- (capacity * math.log(error_rate)) / (math.log(2) ** 2)))
            # Optimal hash count: k = (m / n) * ln(2)
            self.k = int(round((self.m / capacity) * math.log(2)))
            if self.k < 1:
                self.k = 1
        else:
            raise ValueError("Must specify either capacity > 0 or both num_bits and num_hashes.")

        self.bit_array: BitArray = BitArray(self.m)
        self.count: int = 0  # Approximate number of elements inserted

    def _get_hashes(self, item_bytes: bytes) -> List[int]:
        """
        Generates k hash indices in range [0, m-1] using double hashing (Kirsch-Mitzenmacher technique).
        Produces two 64-bit independent hashes using blake2b with two distinct salts.
        """
        h1 = int.from_bytes(hashlib.blake2b(item_bytes, digest_size=8, salt=b"bloom_h1").digest(), "big")
        h2 = int.from_bytes(hashlib.blake2b(item_bytes, digest_size=8, salt=b"bloom_h2").digest(), "big")

        if h2 == 0:
            h2 = 1

        indices = []
        for i in range(self.k):
            idx = (h1 + i * h2) % self.m
            indices.append(idx)
        return indices

    def _item_to_bytes(self, item: Any) -> bytes:
        """Converts arbitrary items to bytes for hashing."""
        if isinstance(item, bytes):
            return item
        elif isinstance(item, str):
            return item.encode("utf-8")
        elif isinstance(item, (int, float)):
            return str(item).encode("utf-8")
        elif isinstance(item, (list, tuple)):
            return repr(item).encode("utf-8")
        else:
            return str(item).encode("utf-8")

    def add(self, item: Any) -> None:
        """Inserts an element into the Bloom filter."""
        item_bytes = self._item_to_bytes(item)
        for idx in self._get_hashes(item_bytes):
            self.bit_array.set(idx)
        self.count += 1

    def add_all(self, items: Iterable[Any]) -> None:
        """Inserts multiple elements into the Bloom filter."""
        for item in items:
            self.add(item)

    def contains(self, item: Any) -> bool:
        """
        Tests if item is in the set.
        Returns:
            False: The item is DEFINITIVELY NOT in the set (zero false negatives).
            True: The item is PROBABLY in the set (with false positive rate <= p).
        """
        item_bytes = self._item_to_bytes(item)
        for idx in self._get_hashes(item_bytes):
            if not self.bit_array.get(idx):
                return False
        return True

    def __contains__(self, item: Any) -> bool:
        return self.contains(item)

    def current_false_positive_rate(self) -> float:
        """
        Computes the theoretical false positive probability based on the fraction
        of set bits in the bit array:
            FPR = (set_bits / m) ^ k
        """
        set_bits = self.bit_array.count_ones()
        fraction_ones = set_bits / self.m
        return fraction_ones ** self.k

    def get_stats(self) -> dict:
        """Returns diagnostic statistics for the Bloom filter."""
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
            "bits_per_item": round(self.m / self.capacity, 2) if self.capacity else None
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves Bloom filter metadata and bit array bytes to disk."""
        filepath = Path(filepath)
        metadata = {
            "m": self.m,
            "k": self.k,
            "capacity": self.capacity,
            "target_error_rate": self.target_error_rate,
            "count": self.count
        }
        meta_path = filepath.with_suffix(".json")
        bin_path = filepath.with_suffix(".bin")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(bin_path, "wb") as f:
            f.write(self.bit_array.to_bytes())

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BloomFilter":
        """Loads Bloom filter metadata and bit array bytes from disk."""
        filepath = Path(filepath)
        meta_path = filepath.with_suffix(".json")
        bin_path = filepath.with_suffix(".bin")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        bf = cls(
            capacity=metadata["capacity"],
            error_rate=metadata["target_error_rate"] or 0.01,
            num_bits=metadata["m"],
            num_hashes=metadata["k"]
        )
        bf.count = metadata["count"]

        with open(bin_path, "rb") as f:
            raw_bytes = f.read()

        bf.bit_array = BitArray.from_bytes(raw_bytes, metadata["m"])
        return bf


class FingerprintBloomFilter:
    """
    Application-level Bloom Filter manager for fingerprint duplicate detection and retrieval.
    
    Indexes fingerprints in the database to allow rapid pre-filtering:
    1. Identity Bloom Filter: Checks if fingerprint identity (e.g. subject ID x2) exists in DB.
    2. Band Bloom Filter: Checks if query MinHash LSH bands match any registered band buckets.
    3. Record Bloom Filter: Checks if the exact file or record signature exists in DB.
    """
    def __init__(
        self,
        identity_filter: BloomFilter,
        band_filter: Optional[BloomFilter] = None,
        record_filter: Optional[BloomFilter] = None
    ):
        self.identity_filter = identity_filter
        self.band_filter = band_filter
        self.record_filter = record_filter

    @staticmethod
    def extract_identity(filename_or_path: str) -> str:
        """Extracts subject/identity identifier from filename structured as q{quality}_{identity}_{impression}.json"""
        basename = os.path.basename(filename_or_path).replace(".json", "")
        parts = basename.split("_")
        return parts[1] if len(parts) >= 2 else basename

    def contains_identity(self, identity_or_filename: str) -> bool:
        """Returns True if the fingerprint identity is probably in the database, False if definitely not."""
        ident = self.extract_identity(identity_or_filename)
        return self.identity_filter.contains(ident)

    def contains_bands(self, signatures: List[int], bands: int = 8) -> bool:
        """
        Checks if ANY of the query's LSH band sub-signatures are present in the database.
        If this returns False, LSH candidate search is guaranteed to return 0 matches.
        """
        if self.band_filter is None:
            return True
        rows = len(signatures) // bands
        for i in range(bands):
            band_tuple = tuple(signatures[rows * i : rows * (i + 1)])
            band_key = f"b{i}_{band_tuple}"
            if self.band_filter.contains(band_key):
                return True
        return False

    def contains_record(self, file_name: str) -> bool:
        """Checks if exact record filename is already in the database."""
        if self.record_filter is None:
            return False
        return self.record_filter.contains(file_name)

    def check_fingerprint(
        self,
        doc: dict,
        mode: str = "identity",
        bands: int = 8
    ) -> Tuple[bool, str]:
        """
        Unified pre-retrieval check.
        Returns:
            (is_present, reason)
        """
        if mode == "identity":
            fname = doc.get("file_name", "")
            if not self.contains_identity(fname):
                return False, f"Identity '{self.extract_identity(fname)}' definitively not found in database."
            return True, f"Identity '{self.extract_identity(fname)}' probably in database."

        elif mode == "band":
            sigs = doc.get("signatures", [])
            if not sigs:
                return False, "No signatures found in doc to check bands."
            if not self.contains_bands(sigs, bands=bands):
                return False, "No LSH bands matched any database buckets (zero candidates guaranteed)."
            return True, "One or more LSH bands matched database buckets."

        elif mode == "hybrid":
            fname = doc.get("file_name", "")
            if not self.contains_identity(fname):
                return False, f"Identity '{self.extract_identity(fname)}' definitively not in database."
            sigs = doc.get("signatures", [])
            if sigs and self.band_filter and not self.contains_bands(sigs, bands=bands):
                return False, "Identity passed, but no LSH bands matched."
            return True, "Passed hybrid Bloom filter check."

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'identity', 'band', or 'hybrid'.")


def build_fingerprint_bloom_filter(
    signatures_path: Union[str, Path],
    error_rate: float = 0.001,
    include_bands: bool = True,
    bands: int = 8
) -> FingerprintBloomFilter:
    """
    Builds a FingerprintBloomFilter from the signatures database.
    """
    signatures_path = Path(signatures_path)
    with open(signatures_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    identities = set()
    records = set()
    all_bands = set()

    rows = len(data[0]["signatures"]) // bands if data else 8

    for item in data:
        fname = item["file_name"]
        ident = FingerprintBloomFilter.extract_identity(fname)
        identities.add(ident)
        records.add(fname)

        if include_bands:
            sigs = item["signatures"]
            for i in range(bands):
                band_tuple = tuple(sigs[rows * i : rows * (i + 1)])
                all_bands.add(f"b{i}_{band_tuple}")

    # Build Identity Bloom Filter
    id_bf = BloomFilter(capacity=len(identities), error_rate=error_rate)
    id_bf.add_all(identities)

    # Build Record Bloom Filter
    rec_bf = BloomFilter(capacity=len(records), error_rate=error_rate)
    rec_bf.add_all(records)

    # Build Band Bloom Filter
    band_bf = None
    if include_bands:
        band_bf = BloomFilter(capacity=len(all_bands), error_rate=error_rate)
        band_bf.add_all(all_bands)

    return FingerprintBloomFilter(
        identity_filter=id_bf,
        band_filter=band_bf,
        record_filter=rec_bf
    )
