"""
Performance benchmarking suite for million-scale synthetic fingerprint generation and storage.
Measures:
- Generation throughput (identities/sec, minutiae/sec)
- Process RAM memory footprint
- HDF5 write throughput and compression efficiency
- Random access vs sequential batch read latency
- Scalability across increasing batch sizes and chunk sizes
"""
import argparse
import os
import pickle
import sys
import time
import tempfile
import tracemalloc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.generation.pipeline import SyntheticGenerationPipeline
from src.storage.hdf5_writer import HDF5ShardWriter
from src.storage.hdf5_reader import HDF5ShardReader

def benchmark_generation_and_storage(models_path: str):
    print("=" * 70)
    print("MILLION-SCALE PERFORMANCE BENCHMARK SUITE")
    print("=" * 70)

    with open(models_path, 'rb') as f:
        bundle = pickle.load(f)

    pipeline = SyntheticGenerationPipeline(
        model_a=bundle['model_a'],
        angle_model=bundle['angle_model'],
        quality_model=bundle['quality_model'],
        max_minutiae=128
    )

    batch_sizes = [50, 200, 500]
    total_test_ids = 1000

    print("\n--- 1. Generation Throughput & Memory Scaling ---")
    for bs in batch_sizes:
        tracemalloc.start()
        t0 = time.time()
        total_obs = 0
        total_minutiae = 0

        for batch in pipeline.generate_streaming(total_identities=total_test_ids, batch_size=bs, master_seed=42):
            total_obs += batch.num_observations
            total_minutiae += int(np.sum(batch.num_minutiae))

        elapsed = time.time() - t0
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        id_throughput = total_test_ids / elapsed
        obs_throughput = total_obs / elapsed
        min_throughput = total_minutiae / elapsed
        peak_mb = peak_mem / (1024 * 1024)

        print(f"Batch Size: {bs:4d} | Elapsed: {elapsed:5.2f}s | "
              f"Identities/s: {id_throughput:6.1f} | "
              f"Obs/s: {obs_throughput:7.1f} | "
              f"Minutiae/s: {min_throughput:9.1f} | "
              f"Peak RAM: {peak_mb:5.1f} MB")

    print("\n--- 2. Storage Layout & Compression Benchmarking ---")
    compression_options = [
        ("None (Raw)", None),
        ("LZF (High Speed)", "lzf"),
        ("GZIP-4 (High Compression)", "gzip")
    ]
    test_obs = 15000 # 1000 identities * 15 obs

    # Generate test batch
    sample_batch = pipeline.generate_batch(start_identity=1, num_identities_in_batch=1000, master_seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        for comp_name, comp_filter in compression_options:
            fpath = os.path.join(tmpdir, f"bench_{comp_name.split()[0].lower()}.h5")
            
            # Measure Write Time
            t0 = time.time()
            writer = HDF5ShardWriter(
                filepath=fpath,
                max_minutiae=128,
                chunk_size=1000,
                compression=comp_filter
            )
            writer.append_batch(sample_batch)
            writer.close()
            t_write = time.time() - t0
            file_mb = os.path.getsize(fpath) / (1024 * 1024)
            kb_per_obs = (os.path.getsize(fpath) / test_obs) / 1024.0

            # Measure Random Access Read Time (100 random samples)
            t0 = time.time()
            with HDF5ShardReader(fpath) as reader:
                indices = np.random.choice(test_obs, 100, replace=False)
                for idx in indices:
                    _ = reader.get_sample(int(idx))
            t_random_read = (time.time() - t0) * 1000.0 # ms

            # Measure Sequential Batch Read Time
            t0 = time.time()
            with HDF5ShardReader(fpath) as reader:
                for chunk in reader.iter_samples(batch_size=1000):
                    pass
            t_seq_read = time.time() - t0

            print(f"Filter: {comp_name:25s} | File Size: {file_mb:5.2f} MB ({kb_per_obs:4.2f} KB/obs) | "
                  f"Write Time: {t_write:5.3f}s ({test_obs/t_write:6.0f} obs/s) | "
                  f"100 Random Reads: {t_random_read:5.1f}ms | "
                  f"Seq Read All: {t_seq_read:5.3f}s")

    print("\n--- 3. Scalability Projection for Millions of Fingerprints ---")
    one_million_obs_seconds = 1000000.0 / (test_obs / t_write)
    one_million_mb = 1000000.0 * (file_mb / test_obs)
    print(f"Projected time for 1,000,000 observations (66,666 identities): ~{one_million_obs_seconds/60:.1f} minutes")
    print(f"Projected disk storage for 1,000,000 observations (LZF):       ~{one_million_mb/1024:.2f} GB")
    print(f"Projected time for 10,000,000 observations (666,666 identities): ~{one_million_obs_seconds*10/3600:.1f} hours")
    print(f"Projected disk storage for 10,000,000 observations (LZF):      ~{one_million_mb*10/1024:.2f} GB")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Benchmark synthetic generation system.")
    parser.add_argument("--models-path", type=str, default="models/fitted_models.pkl", help="Path to fitted models")
    args = parser.parse_args()
    benchmark_generation_and_storage(args.models_path)

if __name__ == '__main__':
    main()
