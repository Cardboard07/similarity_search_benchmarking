"""
CLI script for large-scale streaming generation of synthetic fingerprints.
Generates millions of observations in buffered batches written directly to HDF5 shards.
Supports multiprocessing across CPU cores, resumability, deterministic seeding, and automatic manifest updates.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pickle
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.generation.pipeline import SyntheticGenerationPipeline
from src.storage.hdf5_writer import HDF5ShardWriter
from src.storage.manifest import DatasetManifest, ShardInfo

def generate_shard_worker(
    shard_id: int,
    identity_start: int,
    identity_end: int,
    master_seed: int,
    models_path: str,
    output_dir: str,
    batch_size: int = 500,
    max_minutiae: int = 128,
    chunk_size: int = 1000,
    compression: str = "lzf"
) -> dict:
    """
    Worker function executed in parallel processes.
    Generates a range of identities [identity_start, identity_end] and writes them directly to an HDF5 shard.
    """
    with open(models_path, 'rb') as f:
        bundle = pickle.load(f)

    pipeline = SyntheticGenerationPipeline(
        model_a=bundle['model_a'],
        angle_model=bundle['angle_model'],
        quality_model=bundle['quality_model'],
        max_minutiae=max_minutiae
    )

    filename = f"dataset_{shard_id:03d}.h5"
    filepath = os.path.join(output_dir, filename)
    
    # If resuming and shard file already exists with full observations, verify and reuse
    expected_obs = (identity_end - identity_start + 1) * 15
    if os.path.exists(filepath):
        try:
            import h5py
            with h5py.File(filepath, 'r') as hf:
                if len(hf['data/num_minutiae']) == expected_obs:
                    num_ids = identity_end - identity_start + 1
                    return {
                        'shard_id': shard_id,
                        'filename': filename,
                        'num_observations': expected_obs,
                        'num_identities': num_ids,
                        'identity_start': identity_start,
                        'identity_end': identity_end,
                        'status': 'completed'
                    }
        except Exception:
            pass # overwrite corrupt or incomplete file

    # Remove incomplete file if exists
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    writer = HDF5ShardWriter(
        filepath=filepath,
        max_minutiae=max_minutiae,
        chunk_size=chunk_size,
        compression=compression,
        master_seed=master_seed
    )

    curr = identity_start
    while curr <= identity_end:
        bs = min(batch_size, identity_end - curr + 1)
        batch = pipeline.generate_batch(
            start_identity=curr,
            num_identities_in_batch=bs,
            master_seed=master_seed
        )
        writer.append_batch(batch)
        curr += bs

    writer.close()
    num_ids = identity_end - identity_start + 1
    return {
        'shard_id': shard_id,
        'filename': filename,
        'num_observations': num_ids * 15,
        'num_identities': num_ids,
        'identity_start': identity_start,
        'identity_end': identity_end,
        'status': 'completed'
    }

def main():
    parser = argparse.ArgumentParser(description="Large-scale synthetic fingerprint generation.")
    parser.add_argument("--models-path", type=str, default="models/fitted_models.pkl", help="Path to serialized models")
    parser.add_argument("--num-identities", type=int, default=400000, help="Number of synthetic identities to generate")
    parser.add_argument("--batch-size", type=int, default=500, help="Number of identities per in-memory batch")
    parser.add_argument("--shard-size", type=int, default=150000, help="Observations per HDF5 shard file")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Master deterministic seed")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory for HDF5 shards")
    parser.add_argument("--resume", action="store_true", help="Resume generation from existing manifest")
    args = parser.parse_args()

    if not os.path.exists(args.models_path):
        raise FileNotFoundError(f"Fitted models not found at: {args.models_path}. Run fit_models.py first.")

    with open(args.models_path, 'rb') as f:
        bundle = pickle.load(f)

    config = bundle.get('config', {})
    storage_cfg = config.get('storage', {})
    max_minutiae = storage_cfg.get('max_minutiae', 128)
    compression = storage_cfg.get('compression', 'lzf')
    chunk_size = storage_cfg.get('chunk_size', 1000)

    os.makedirs(args.output_dir, exist_ok=True)
    manifest = DatasetManifest(output_dir=args.output_dir, master_seed=args.seed)

    # Calculate shard distribution
    # Each identity has 15 observations
    ids_per_shard = args.shard_size // 15
    total_shards = (args.num_identities + ids_per_shard - 1) // ids_per_shard
    total_observations = args.num_identities * 15

    print("=" * 70)
    print("MILLION-SCALE FINGERPRINT GENERATION")
    print(f"Total Identities:   {args.num_identities:,} (4 lakh)")
    print(f"Total Observations: {total_observations:,} (6 million samples across 15 variations)")
    print(f"Total Shards:       {total_shards} shards ({ids_per_shard:,} identities / {args.shard_size:,} obs each)")
    print(f"Parallel Workers:   {args.workers}")
    print(f"Output Directory:   {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    # Determine which shards need to be generated
    completed_shards = {s.shard_id for s in manifest.shards if s.status == "completed"}
    pending_tasks = []

    for s_id in range(total_shards):
        id_start = s_id * ids_per_shard + 1
        id_end = min((s_id + 1) * ids_per_shard, args.num_identities)
        
        # Check if already completed and file exists
        shard_fname = f"dataset_{s_id:03d}.h5"
        shard_path = os.path.join(args.output_dir, shard_fname)
        if args.resume and s_id in completed_shards and os.path.exists(shard_path):
            continue

        pending_tasks.append((
            s_id,
            id_start,
            id_end,
            args.seed,
            args.models_path,
            args.output_dir,
            args.batch_size,
            max_minutiae,
            chunk_size,
            compression
        ))

    already_done_identities = args.num_identities - sum(t[2] - t[1] + 1 for t in pending_tasks)
    print(f"Already completed: {already_done_identities:,} identities ({already_done_identities * 15:,} obs)")
    print(f"Remaining to run:  {args.num_identities - already_done_identities:,} identities ({len(pending_tasks)} shards)")

    if not pending_tasks:
        print("All shards are already completed! Nothing to generate.")
        return

    t0 = time.time()
    num_to_generate = sum(t[2] - t[1] + 1 for t in pending_tasks)
    pbar = tqdm(total=num_to_generate, desc="Generating synthetic fingerprints", unit="ids")

    if args.workers <= 1:
        # Sequential generation
        for task in pending_tasks:
            shard_info_dict = generate_shard_worker(*task)
            manifest.record_shard(ShardInfo(**shard_info_dict))
            pbar.update(shard_info_dict['num_identities'])
    else:
        # Multiprocess worker pool
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_shard = {
                executor.submit(generate_shard_worker, *task): task[0]
                for task in pending_tasks
            }
            for future in as_completed(future_to_shard):
                shard_info_dict = future.result()
                manifest.record_shard(ShardInfo(**shard_info_dict))
                pbar.update(shard_info_dict['num_identities'])

    pbar.close()
    elapsed = time.time() - t0
    total_generated_obs = num_to_generate * 15
    throughput_obs = total_generated_obs / elapsed if elapsed > 0 else 0
    throughput_ids = num_to_generate / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 70)
    print(f"GENERATION COMPLETE in {elapsed/60:.2f} minutes ({elapsed:.1f}s)!")
    print(f"Generated: {total_generated_obs:,} observations ({num_to_generate:,} identities)")
    print(f"Overall Throughput: {throughput_obs:,.1f} obs/s ({throughput_ids:,.1f} identities/s)")
    print(f"Manifest written to: {manifest.manifest_path}")
    print(f"Total dataset in manifest: {manifest.total_observations:,} observations ({manifest.total_identities:,} identities)")
    print("=" * 70)

if __name__ == '__main__':
    main()
