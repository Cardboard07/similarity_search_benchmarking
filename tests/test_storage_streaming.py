"""
Unit tests for HDF5 chunked storage, reader, and resumable manifest.
"""
import os
import tempfile
import numpy as np
import pytest
from src.generation.pipeline import GenerationBatch
from src.storage.hdf5_writer import HDF5ShardWriter
from src.storage.hdf5_reader import HDF5ShardReader
from src.storage.manifest import DatasetManifest, ShardInfo

def test_hdf5_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = os.path.join(tmpdir, "test_shard.h5")
        writer = HDF5ShardWriter(filepath=shard_path, max_minutiae=64, chunk_size=10, compression="lzf")

        # Create dummy batch
        n_obs = 15 # 1 identity * 15 obs
        minutiae = np.zeros((n_obs, 64, 5), dtype=np.float32)
        minutiae[:, :10, :2] = np.random.uniform(50, 300, size=(n_obs, 10, 2))
        minutiae[:, :10, 2] = 45.0
        minutiae[:, :10, 3] = 1
        
        batch = GenerationBatch(
            minutiae=minutiae,
            num_minutiae=np.full(n_obs, 10, dtype=np.int32),
            identities=np.full(n_obs, 101, dtype=np.int32),
            orientations=np.tile([1, 2, 3], 5),
            qualities=np.repeat([3, 7, 5, 1, 9], 3),
            seeds=np.full(n_obs, 42, dtype=np.int64),
            num_identities=1,
            num_observations=n_obs
        )

        writer.append_batch(batch)
        writer.close()

        # Read back
        with HDF5ShardReader(shard_path) as reader:
            assert reader.num_samples == 15
            sample0 = reader.get_sample(0)
            assert sample0['identity'] == 101
            assert sample0['num_minutiae'] == 10
            assert sample0['pts'].shape == (10, 2)
            assert pytest.approx(sample0['angles'][0], abs=1e-5) == 45.0
            assert sample0['types'][0] == 1

def test_manifest_resumability():
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = DatasetManifest(output_dir=tmpdir, master_seed=42)
        next_id, next_shard = manifest.get_resume_state(100)
        assert next_id == 1
        assert next_shard == 0

        # Record completed shard
        manifest.record_shard(ShardInfo(
            shard_id=0,
            filename="dataset_000.h5",
            num_observations=300,
            num_identities=20,
            identity_start=1,
            identity_end=20,
            status="completed"
        ))

        next_id, next_shard = manifest.get_resume_state(100)
        assert next_id == 21
        assert next_shard == 1
