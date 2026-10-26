"""
Multiprocess Streaming Signature Extractor for 40 HDF5 Shards.

Processes 6,000,000 fingerprint observations in parallel across all CPU cores,
saving chunked, LZF-compressed signature shards with gallery/probe split metadata.

Train/Test split:
    Randomised per-identity. For each identity, exactly floor(n_obs * probe_ratio)
    observations are held out as probes, chosen uniformly at random across all
    orientations and quality levels. This ensures both gallery and probe sets
    contain a mixture of all orientations and qualities.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import h5py
import numpy as np

from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine


def process_single_shard(
    input_h5_path: str,
    output_h5_path: str,
    probe_ratio: float = 1.0 / 3.0,
    batch_size: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Extracts MinHash signatures for all observations in an HDF5 shard.

    The gallery/probe split is randomised per-identity: for each unique identity
    in the shard, floor(n_obs * probe_ratio) observations are randomly selected
    as probes; the rest go to the gallery. This mixes orientations and qualities
    in both splits.

    Args:
        input_h5_path: Path to dataset_XXX.h5
        output_h5_path: Path to write signatures_XXX.h5
        probe_ratio:   Fraction of each identity's observations held out as probes.
                       Default 1/3 (5 of 15 obs per identity).
        batch_size:    Read batch size for HDF5 streaming.
        seed:          RNG seed (controls both hash family and split).

    Returns:
        Dict of processing statistics.
    """
    t0 = time.perf_counter()
    engine = VectorizedMinHashEngine(num_hashes=64, seed=seed)
    rng = np.random.default_rng(seed)

    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)

    with h5py.File(input_h5_path, "r") as in_f:
        d_min = in_f["data/minutiae"]
        d_num = in_f["data/num_minutiae"]
        d_id  = in_f["metadata/synthetic_identity"]
        d_ori = in_f["metadata/orientation"]
        d_qual = in_f["metadata/quality"]

        n_samples = len(d_num)

        # Pre-allocate output arrays
        signatures   = np.empty((n_samples, 64), dtype=np.uint32)
        identities   = np.empty(n_samples, dtype=np.int64)
        orientations = np.empty(n_samples, dtype=np.int32)
        qualities    = np.empty(n_samples, dtype=np.int32)
        is_train     = np.ones(n_samples, dtype=np.bool_)   # default: gallery

        # Read all metadata in one pass (metadata arrays are small)
        identities[:]   = d_id[:]
        orientations[:] = d_ori[:]
        qualities[:]    = d_qual[:]

        # --- Randomised per-identity split ---
        unique_ids = np.unique(identities)
        for uid in unique_ids:
            mask = np.where(identities == uid)[0]
            n_obs = len(mask)
            n_probe = max(1, int(np.floor(n_obs * probe_ratio)))
            probe_local = rng.choice(n_obs, size=n_probe, replace=False)
            probe_global = mask[probe_local]
            is_train[probe_global] = False
        # -------------------------------------

        # Compute signatures batch-by-batch
        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            batch_n = end - start

            b_min = d_min[start:end]
            b_num = d_num[start:end]

            for i in range(batch_n):
                c = int(b_num[i])
                pts    = b_min[i, :c, :2]
                angles = b_min[i, :c, 2]
                types  = b_min[i, :c, 3].astype(np.int32)
                signatures[start + i] = engine.process_observation(pts, angles, types)

    # Write compressed HDF5 signature shard
    with h5py.File(output_h5_path, "w") as out_f:
        out_f.create_dataset(
            "data/signatures",
            data=signatures,
            chunks=(min(10000, n_samples), 64),
            compression="lzf",
        )
        out_f.create_dataset("metadata/synthetic_identity", data=identities,   compression="lzf")
        out_f.create_dataset("metadata/orientation",        data=orientations,  compression="lzf")
        out_f.create_dataset("metadata/quality",            data=qualities,     compression="lzf")
        out_f.create_dataset("metadata/is_train",           data=is_train,      compression="lzf")

        out_f.attrs["input_file"]  = os.path.basename(input_h5_path)
        out_f.attrs["num_samples"] = n_samples
        out_f.attrs["num_train"]   = int(np.sum(is_train))
        out_f.attrs["num_test"]    = int(np.sum(~is_train))
        out_f.attrs["probe_ratio"] = probe_ratio
        out_f.attrs["split_seed"]  = seed

    elapsed = time.perf_counter() - t0
    throughput = n_samples / elapsed if elapsed > 0 else 0.0

    return {
        "shard":               os.path.basename(input_h5_path),
        "num_samples":         n_samples,
        "num_train":           int(np.sum(is_train)),
        "num_test":            int(np.sum(~is_train)),
        "elapsed_sec":         round(elapsed, 2),
        "throughput_obs_sec":  round(throughput, 1),
    }


class ShardedSignatureDataset:
    """
    Seamless, memory-efficient interface for reading signatures and metadata across HDF5 shards.
    Supports scaling to hundreds of shards with zero-copy pre-allocated memory loading.
    """
    def __init__(self, shards_dir: str):
        self.shards_dir = Path(shards_dir)
        # Numerical sorting to safely handle shards beyond 999 (e.g. signatures_1000.h5)
        self.shard_files = sorted(
            self.shards_dir.glob("signatures_*.h5"),
            key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
        )
        if not self.shard_files:
            raise FileNotFoundError(f"No signature shards found in {shards_dir}")

        # Build shard index mapping
        self.shard_lengths = []
        self.total_samples = 0
        self.shard_starts = []

        for sf in self.shard_files:
            with h5py.File(sf, "r") as f:
                n = len(f["metadata/synthetic_identity"])
                self.shard_starts.append(self.total_samples)
                self.shard_lengths.append(n)
                self.total_samples += n

    def load_gallery(self, max_records: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads all Train/Gallery signatures and identities into memory.
        Uses pre-allocated arrays to avoid 2x RAM spike during concatenation.

        Returns:
            (gallery_signatures, gallery_identities)
        """
        # Pass 1: Quick inspection of counts to pre-allocate exact memory size
        shard_plan = []
        total_to_load = 0

        for sf in self.shard_files:
            with h5py.File(sf, "r") as f:
                is_train = f["metadata/is_train"][:]
                n_train = int(np.count_nonzero(is_train))
                if max_records and total_to_load + n_train > max_records:
                    take = max_records - total_to_load
                    shard_plan.append((sf, take, is_train))
                    total_to_load += take
                    break
                else:
                    shard_plan.append((sf, n_train, is_train))
                    total_to_load += n_train

        # Pre-allocate contiguous arrays once (eliminates 17+ GB memory spike at 100M scale)
        gallery_sigs = np.empty((total_to_load, 64), dtype=np.uint32)
        gallery_ids = np.empty(total_to_load, dtype=np.int64)

        # Pass 2: Load chunks directly into destination slices
        offset = 0
        for sf, take, is_train in shard_plan:
            with h5py.File(sf, "r") as f:
                s = f["data/signatures"][:][is_train][:take]
                i = f["metadata/synthetic_identity"][:][is_train][:take]
                gallery_sigs[offset:offset + take] = s
                gallery_ids[offset:offset + take] = i
                offset += take

        return gallery_sigs, gallery_ids

    def load_probes(self, max_records: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads Test/Probe signatures and metadata into memory with pre-allocated destination arrays.

        Returns:
            (probe_signatures, probe_identities, probe_orientations, probe_qualities)
        """
        shard_plan = []
        total_to_load = 0

        for sf in self.shard_files:
            with h5py.File(sf, "r") as f:
                is_test = ~f["metadata/is_train"][:]
                n_test = int(np.count_nonzero(is_test))
                if max_records and total_to_load + n_test > max_records:
                    take = max_records - total_to_load
                    shard_plan.append((sf, take, is_test))
                    total_to_load += take
                    break
                else:
                    shard_plan.append((sf, n_test, is_test))
                    total_to_load += n_test

        probe_sigs = np.empty((total_to_load, 64), dtype=np.uint32)
        probe_ids = np.empty(total_to_load, dtype=np.int64)
        probe_oris = np.empty(total_to_load, dtype=np.int32)
        probe_quals = np.empty(total_to_load, dtype=np.int32)

        offset = 0
        for sf, take, is_test in shard_plan:
            with h5py.File(sf, "r") as f:
                s = f["data/signatures"][:][is_test][:take]
                i = f["metadata/synthetic_identity"][:][is_test][:take]
                o = f["metadata/orientation"][:][is_test][:take]
                q = f["metadata/quality"][:][is_test][:take]

                probe_sigs[offset:offset + take] = s
                probe_ids[offset:offset + take] = i
                probe_oris[offset:offset + take] = o
                probe_quals[offset:offset + take] = q
                offset += take

        return probe_sigs, probe_ids, probe_oris, probe_quals
