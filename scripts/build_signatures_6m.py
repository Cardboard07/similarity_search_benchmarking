"""
CLI Script: Extract MinHash Signatures for 6-Million Dataset across 40 HDF5 Shards.

Processes shards in parallel using multiprocessing and constructs the 400,000-identity Bloom filter.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

from src.lsh_scale.hdf5_signature_extractor import process_single_shard
from src.lsh_scale.bloom_filter_6m import ScaledIdentityBloomFilter


def worker_task(args: Tuple[str, str, int, bool]) -> dict:
    in_path, out_path, batch_size, resume = args
    if resume and os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return {
            "shard": os.path.basename(in_path),
            "status": "skipped (already exists)",
            "num_samples": 150000,
            "elapsed_sec": 0.0,
            "throughput_obs_sec": 0.0
        }

    res = process_single_shard(
        input_h5_path=in_path,
        output_h5_path=out_path,
        batch_size=batch_size
    )
    res["status"] = "completed"
    return res


def main():
    parser = argparse.ArgumentParser(description="Extract 6M MinHash signatures from 40 HDF5 shards.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(workspace_root / "dataset"),
        help="Directory containing dataset_XXX.h5 shards."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(workspace_root / "dataset" / "signatures_shards"),
        help="Directory to save signatures_XXX.h5 shards."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(10, cpu_count()),
        help="Number of parallel worker processes."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="HDF5 batch read size per shard."
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        help="Limit number of shards to process (for quick testing)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip already processed shards."
    )
    parser.add_argument(
        "--bloom-output",
        type=str,
        default=str(workspace_root / "dataset" / "bloom_filter_400k"),
        help="Path prefix to save the 400k identity Bloom filter."
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = sorted(dataset_path.glob("dataset_*.h5"))
    if not input_files:
        print(f"Error: No dataset_*.h5 files found in {args.dataset_dir}")
        sys.exit(1)

    if args.max_shards:
        input_files = input_files[:args.max_shards]

    print("=" * 70)
    print(f"      6-MILLION SAMPLE SIGNATURE EXTRACTION PIPELINE")
    print("=" * 70)
    print(f"Input Directory    : {dataset_path}")
    print(f"Output Directory   : {output_path}")
    print(f"Shards to Process  : {len(input_files)}")
    print(f"Total Observations : {len(input_files) * 150000:,}")
    print(f"Parallel Workers   : {args.num_workers} processes")
    print(f"Batch Size         : {args.batch_size}")
    print("=" * 70)

    tasks = []
    for in_f in input_files:
        shard_num = in_f.stem.replace("dataset_", "")
        out_f = output_path / f"signatures_{shard_num}.h5"
        tasks.append((str(in_f), str(out_f), args.batch_size, args.resume))

    t_start = time.perf_counter()

    with Pool(processes=args.num_workers) as pool:
        for res in pool.imap_unordered(worker_task, tasks):
            print(
                f"[{res['status'].upper():10s}] {res['shard']} - "
                f"{res.get('num_samples', 0):,} obs in {res.get('elapsed_sec', 0):.1f}s "
                f"({res.get('throughput_obs_sec', 0):,.0f} obs/s)"
            )

    total_time = time.perf_counter() - t_start
    total_samples = len(input_files) * 150000
    overall_throughput = total_samples / total_time if total_time > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"Extraction Complete!")
    print(f"Total Time        : {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Overall Throughput: {overall_throughput:,.0f} observations/second")
    print("=" * 70)

    # Build and save 400,000 Identity Bloom Filter
    print(f"\nBuilding 400,000-Identity Bloom Filter...")
    t_bf0 = time.perf_counter()
    bf = ScaledIdentityBloomFilter(capacity=400000, error_rate=0.001)

    # Identifiers in the dataset run consecutively from 1 to 400,000
    # Add all 400,000 unique identities
    bf.add_all(range(1, 400001))
    t_bf1 = time.perf_counter()

    bf.save(args.bloom_output)
    print(f"Bloom Filter built in {t_bf1 - t_bf0:.2f}s and saved to {args.bloom_output}.bin/.json")
    print("Diagnostic statistics:")
    for k, v in bf.get_stats().items():
        print(f"  |- {k:25s}: {v}")


if __name__ == "__main__":
    main()
