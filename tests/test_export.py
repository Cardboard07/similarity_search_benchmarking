"""
Unit tests for JSON exporter module and filename / schema compatibility.
"""
import json
import os
import tempfile
import numpy as np
import pytest
from src.generation.pipeline import GenerationBatch
from src.storage.hdf5_writer import HDF5ShardWriter
from src.storage.manifest import DatasetManifest, ShardInfo
from src.storage.exporter import JSONExporter

def test_json_exporter_naming_and_schema():
    with tempfile.TemporaryDirectory() as tmpdir:
        shard_path = os.path.join(tmpdir, "dataset_000.h5")
        writer = HDF5ShardWriter(filepath=shard_path, max_minutiae=64, chunk_size=10, compression="lzf")

        # 1 identity * 15 observations
        n_obs = 15
        minutiae = np.zeros((n_obs, 64, 5), dtype=np.float32)
        minutiae[:, :5, :2] = np.array([[100, 150], [120, 180], [200, 220], [250, 300], [310, 350]])
        minutiae[:, :5, 2] = 180.0
        minutiae[:, :5, 3] = 2

        # 3 orientations x 5 qualities
        oris = []
        quals = []
        for o in [1, 2, 3]:
            for q in [3, 7, 5, 1, 9]:
                oris.append(o)
                quals.append(q)

        batch = GenerationBatch(
            minutiae=minutiae,
            num_minutiae=np.full(n_obs, 5, dtype=np.int32),
            identities=np.full(n_obs, 1001, dtype=np.int32),
            orientations=np.array(oris, dtype=np.int32),
            qualities=np.array(quals, dtype=np.int32),
            seeds=np.full(n_obs, 42, dtype=np.int64),
            num_identities=1,
            num_observations=n_obs
        )
        writer.append_batch(batch)
        writer.close()

        manifest = DatasetManifest(output_dir=tmpdir)
        manifest.record_shard(ShardInfo(
            shard_id=0,
            filename="dataset_000.h5",
            num_observations=15,
            num_identities=1,
            identity_start=1001,
            identity_end=1001,
            status="completed"
        ))

        export_dir = os.path.join(tmpdir, "exported")
        exporter = JSONExporter(manifest.manifest_path)
        exported_files = exporter.export_identity(identity=1001, export_dir=export_dir)

        assert len(exported_files) == 15
        
        # Verify naming convention: q<quality>_<identity>_<orientation>.json
        expected_sample = os.path.join(export_dir, "q3_1001_1.json")
        assert os.path.exists(expected_sample)

        # Verify JSON content matches exact schema
        with open(expected_sample, 'r') as fp:
            d = json.load(fp)

        assert d['file_name'] == "q3_1001_1.json"
        assert d['quality_folder'] == 3
        assert d['identity'] == "1001"
        assert d['impression'] == "1"
        assert d['image_size'] == {"x": 416, "y": 560}
        assert d['num_minutiae'] == 5
        assert len(d['minutiae']) == 5
        assert d['minutiae'][0]['x'] == 100
        assert d['minutiae'][0]['y'] == 150
        assert d['minutiae'][0]['angle'] == 180.0
        assert d['minutiae'][0]['type'] == 2
