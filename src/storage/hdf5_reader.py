"""
HDF5 shard and dataset reader supporting random access, batch reads, and streaming iteration.
"""
from typing import Iterator
import h5py
import numpy as np

class HDF5ShardReader:
    """
    Reader for individual HDF5 shard files.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = h5py.File(filepath, 'r')
        self.d_minutiae = self.file['data/minutiae']
        self.d_num = self.file['data/num_minutiae']
        self.d_identity = self.file['metadata/synthetic_identity']
        self.d_orientation = self.file['metadata/orientation']
        self.d_quality = self.file['metadata/quality']
        self.d_seed = self.file['metadata/seed']
        self.num_samples = len(self.d_num)

    def get_sample(self, idx: int) -> dict:
        """
        Retrieves a single unpadded fingerprint observation.
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
            
        c = int(self.d_num[idx])
        raw_m = self.d_minutiae[idx, :c] # [c, 5]
        
        return {
            'identity': int(self.d_identity[idx]),
            'orientation': int(self.d_orientation[idx]),
            'quality': int(self.d_quality[idx]),
            'seed': int(self.d_seed[idx]),
            'num_minutiae': c,
            'pts': raw_m[:, :2],
            'angles': raw_m[:, 2],
            'types': raw_m[:, 3].astype(int),
            'qualities': raw_m[:, 4].astype(int)
        }

    def get_batch(self, start_idx: int, end_idx: int) -> dict:
        """
        Retrieves a slice of observations.
        """
        start_idx = max(0, start_idx)
        end_idx = min(self.num_samples, end_idx)
        
        return {
            'minutiae': self.d_minutiae[start_idx:end_idx],
            'num_minutiae': self.d_num[start_idx:end_idx],
            'identities': self.d_identity[start_idx:end_idx],
            'orientations': self.d_orientation[start_idx:end_idx],
            'qualities': self.d_quality[start_idx:end_idx],
            'seeds': self.d_seed[start_idx:end_idx]
        }

    def iter_samples(self, batch_size: int = 1000) -> Iterator[dict]:
        """
        Streamingly iterates through all samples in the shard chunk-by-chunk.
        """
        for start in range(0, self.num_samples, batch_size):
            end = min(self.num_samples, start + batch_size)
            batch = self.get_batch(start, end)
            n_in_batch = end - start
            for i in range(n_in_batch):
                c = int(batch['num_minutiae'][i])
                raw_m = batch['minutiae'][i, :c]
                yield {
                    'identity': int(batch['identities'][i]),
                    'orientation': int(batch['orientations'][i]),
                    'quality': int(batch['qualities'][i]),
                    'seed': int(batch['seeds'][i]),
                    'num_minutiae': c,
                    'pts': raw_m[:, :2],
                    'angles': raw_m[:, 2],
                    'types': raw_m[:, 3].astype(int),
                    'qualities': raw_m[:, 4].astype(int)
                }

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
