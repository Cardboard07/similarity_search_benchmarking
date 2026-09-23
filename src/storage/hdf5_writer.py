"""
High-performance streaming HDF5 shard writer.
Stores multi-million synthetic fingerprint datasets using fixed-width padded representation,
chunking, and LZF compression.
"""
from datetime import datetime, timezone
import os
import h5py
import numpy as np
from src.generation.pipeline import GenerationBatch

class HDF5ShardWriter:
    """
    Manages an individual HDF5 shard file with incremental appending and chunking.
    """
    def __init__(
        self,
        filepath: str,
        max_minutiae: int = 128,
        chunk_size: int = 1000,
        compression: str = "lzf",
        model_version: str = "1.0.0",
        master_seed: int = 42,
        config_hash: str = "default"
    ):
        self.filepath = filepath
        self.max_minutiae = max_minutiae
        self.chunk_size = chunk_size
        self.compression = compression
        self.model_version = model_version
        self.master_seed = master_seed
        self.config_hash = config_hash
        
        self.num_samples = 0
        self.file = None
        self._init_file()

    def _init_file(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        # Open in append/write mode
        self.file = h5py.File(self.filepath, 'a')
        
        if 'data' not in self.file:
            data_grp = self.file.create_group('data')
            meta_grp = self.file.create_group('metadata')
            gen_grp = self.file.create_group('generation')
            
            # Resizable datasets (maxshape=(None, ...))
            data_grp.create_dataset(
                'minutiae',
                shape=(0, self.max_minutiae, 5),
                maxshape=(None, self.max_minutiae, 5),
                dtype='float32',
                chunks=(self.chunk_size, self.max_minutiae, 5),
                compression=self.compression
            )
            data_grp.create_dataset(
                'num_minutiae',
                shape=(0,),
                maxshape=(None,),
                dtype='int32',
                chunks=(self.chunk_size,),
                compression=self.compression
            )
            
            meta_grp.create_dataset(
                'synthetic_identity',
                shape=(0,),
                maxshape=(None,),
                dtype='int32',
                chunks=(self.chunk_size,),
                compression=self.compression
            )
            meta_grp.create_dataset(
                'orientation',
                shape=(0,),
                maxshape=(None,),
                dtype='int32',
                chunks=(self.chunk_size,),
                compression=self.compression
            )
            meta_grp.create_dataset(
                'quality',
                shape=(0,),
                maxshape=(None,),
                dtype='int32',
                chunks=(self.chunk_size,),
                compression=self.compression
            )
            meta_grp.create_dataset(
                'seed',
                shape=(0,),
                maxshape=(None,),
                dtype='int64',
                chunks=(self.chunk_size,),
                compression=self.compression
            )
            
            # Shard root attributes
            self.file.attrs['model_version'] = self.model_version
            self.file.attrs['master_seed'] = self.master_seed
            self.file.attrs['config_hash'] = self.config_hash
            self.file.attrs['created_at'] = datetime.now(timezone.utc).isoformat()
            self.file.attrs['max_minutiae'] = self.max_minutiae
            self.file.attrs['quality_mapping'] = "{3: 'pristine', 7: 'good', 5: 'noisy', 1: 'bad', 9: 'very_bad'}"
            self.num_samples = 0
        else:
            self.num_samples = len(self.file['data/num_minutiae'])

    def append_batch(self, batch: GenerationBatch):
        """
        Appends a generated batch to the HDF5 shard datasets without keeping previous data in memory.
        """
        n_new = batch.num_observations
        if n_new == 0:
            return

        old_n = self.num_samples
        new_n = old_n + n_new

        data_grp = self.file['data']
        meta_grp = self.file['metadata']

        # Resize datasets
        data_grp['minutiae'].resize(new_n, axis=0)
        data_grp['num_minutiae'].resize(new_n, axis=0)
        meta_grp['synthetic_identity'].resize(new_n, axis=0)
        meta_grp['orientation'].resize(new_n, axis=0)
        meta_grp['quality'].resize(new_n, axis=0)
        meta_grp['seed'].resize(new_n, axis=0)

        # Write chunk slices
        data_grp['minutiae'][old_n:new_n] = batch.minutiae
        data_grp['num_minutiae'][old_n:new_n] = batch.num_minutiae
        meta_grp['synthetic_identity'][old_n:new_n] = batch.identities
        meta_grp['orientation'][old_n:new_n] = batch.orientations
        meta_grp['quality'][old_n:new_n] = batch.qualities
        meta_grp['seed'][old_n:new_n] = batch.seeds

        self.num_samples = new_n
        self.file.attrs['num_observations'] = new_n
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.attrs['num_observations'] = self.num_samples
            self.file.close()
            self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
