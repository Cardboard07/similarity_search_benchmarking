"""
Multiprocess Streaming Signature Extractor for 40 HDF5 Shards.

Processes 6,000,000 fingerprint observations in parallel across all CPU cores,
saving chunked, LZF-compressed signature shards with gallery/probe split metadata.
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
    train_orientations: Tuple[int, ...] = (1, 2),
    batch_size: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Extracts MinHash signatures for all observations in an HDF5 shard.
    
    Args:
        input_h5_path: Path to dataset_XXX.h5
        output_h5_path: Path to write signatures_XXX.h5
        train_orientations: Orientations assigned to Gallery/Train (default 1 and 2 = 10 obs/ID)
        batch_size: Read batch size for HDF5 streaming
        seed: Random seed for hash family
        
    Returns:
        Dict of processing statistics.
    """
    t0 = time.perf_counter()
    engine = VectorizedMinHashEngine(num_hashes=64, seed=seed)

    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)

    with h5py.File(input_h5_path, "r") as in_f:
        d_min = in_f["data/minutiae"]
        d_num = in_f["data/num_minutiae"]
        d_id = in_f["metadata/synthetic_identity"]
        d_ori = in_f["metadata/orientation"]
        d_qual = in_f["metadata/quality"]

        n_samples = len(d_num)

        # Pre-allocate output arrays in memory for speed before writing to HDF5
        signatures = np.empty((n_samples, 64), dtype=np.uint32)
        identities = np.empty(n_samples, dtype=np.int64)
        orientations = np.empty(n_samples, dtype=np.int32)
        qualities = np.empty(n_samples, dtype=np.int32)
        is_train = np.empty(n_samples, dtype=np.bool_)

        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            batch_n = end - start

            b_min = d_min[start:end]
            b_num = d_num[start:end]
            b_id = d_id[start:end]
            b_ori = d_ori[start:end]
            b_qual = d_qual[start:end]

            identities[start:end] = b_id
            orientations[start:end] = b_ori
            qualities[start:end] = b_qual
            # Orientations in train_orientations belong to Gallery (Train)
            is_train[start:end] = np.isin(b_ori, train_orientations)

            for i in range(batch_n):
                c = int(b_num[i])
                pts = b_min[i, :c, :2]
                angles = b_min[i, :c, 2]
                types = b_min[i, :c, 3].astype(np.int32)

                signatures[start + i] = engine.process_observation(pts, angles, types)

    # Write out compressed HDF5 signature file
    with h5py.File(output_h5_path, "w") as out_f:
        d_sig = out_f.create_dataset(
            "data/signatures",
            data=signatures,
            chunks=(min(10000, n_samples), 64),
            compression="lzf"
        )
        out_f.create_dataset("metadata/synthetic_identity", data=identities, compression="lzf")
        out_f.create_dataset("metadata/orientation", data=orientations, compression="lzf")
        out_f.create_dataset("metadata/quality", data=qualities, compression="lzf")
        out_f.create_dataset("metadata/is_train", data=is_train, compression="lzf")

        out_f.attrs["input_file"] = os.path.basename(input_h5_path)
        out_f.attrs["num_samples"] = n_samples
        out_f.attrs["num_train"] = int(np.sum(is_train))
        out_f.attrs["num_test"] = int(np.sum(~is_train))

    elapsed = time.perf_counter() - t0
    throughput = n_samples / elapsed if elapsed > 0 else 0.0

    return {
        "shard": os.path.basename(input_h5_path),
        "num_samples": n_samples,
        "elapsed_sec": round(elapsed, 2),
        "throughput_obs_sec": round(throughput, 1)
    }


class ShardedSignatureDataset:
    """
    Seamless interface for reading signatures and metadata across all 40 shards.
    """
    def __init__(self, shards_dir: str):
        self.shards_dir = Path(shards_dir)
        self.shard_files = sorted(self.shards_dir.glob("signatures_*.h5"))
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
        Returns:
            (gallery_signatures, gallery_identities)
        """
        sigs_list = []
        ids_list = []
        total_loaded = 0

        for sf in self.shard_files:
            with h5py.File(sf, "r") as f:
                is_train = f["metadata/is_train"][:]
                s = f["data/signatures"][:][is_train]
                i = f["metadata/synthetic_identity"][:][is_train]

                if max_records and total_loaded + len(s) > max_records:
                    needed = max_records - total_loaded
                    sigs_list.append(s[:needed])
                    ids_list.append(i[:needed])
                    break
                else:
                    sigs_list.append(s)
                    ids_list.append(i)
                    total_loaded += len(s)

        gallery_sigs = np.vstack(sigs_list)
        gallery_ids = np.concatenate(ids_list)
        return gallery_sigs, gallery_ids

    def load_probes(self, max_records: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads Test/Probe signatures and metadata into memory.
        Returns:
            (probe_signatures, probe_identities, probe_orientations, probe_qualities)
        """
        sigs_list = []
        ids_list = []
        oris_list = []
        quals_list = []
        total_loaded = 0

        for sf in self.shard_files:
            with h5py.File(sf, "r") as f:
                is_test = ~f["metadata/is_train"][:]
                s = f["data/signatures"][:][is_test]
                i = f["metadata/synthetic_identity"][:][is_test]
                o = f["metadata/orientation"][:][is_test]
                q = f["metadata/quality"][:][is_test]

                if max_records and total_loaded + len(s) > max_records:
                    needed = max_records - total_loaded
                    sigs_list.append(s[:needed])
                    ids_list.append(i[:needed])
                    oris_list.append(o[:needed])
                    quals_list.append(q[:needed])
                    break
                else:
                    sigs_list.append(s)
                    ids_list.append(i)
                    oris_list.append(o)
                    quals_list.append(q)
                    total_loaded += len(s)

        return (
            np.vstack(sigs_list),
            np.concatenate(ids_list),
            np.concatenate(oris_list),
            np.concatenate(quals_list)
        )
