#!/usr/bin/env python3
"""
Master HPC Pipeline Orchestrator.

Orchestrates the entire high-performance fingerprint duplicate detection workflow
from million-scale generation and MinHash signature extraction to executing the
three canonical evaluation experiments.

Stages supported:
  - generate:        Generates synthetic fingerprint shards (up to 102M+ samples).
  - signatures:      Extracts 64-element MinHash signatures across all HDF5 shards.
  - exp1:            Experiment 1: LSH Band Bloom Filter parameter sweep (9 canonical configs).
  - exp2:            Experiment 2: Full-scale retrieval benchmark (MRR, MAP@10, Hit@1..50, quality breakdown).
  - exp3:            Experiment 3: 4-way architectural latency & throughput benchmark (Positive, Negative, Mixed).
  - all_experiments: Runs Exp 1, Exp 2, and Exp 3 sequentially on existing signatures.
  - all:             Runs end-to-end: generate -> signatures -> exp1 -> exp2 -> exp3.

Usage:
  # Run all 3 experiments on current dataset:
  python scripts/run_hpc_pipeline.py --stage all_experiments

  # Run quick smoke test on 50k samples:
  python scripts/run_hpc_pipeline.py --stage all_experiments --max-gallery 50000 --num-queries 20

  # Run full generation to 102M and evaluate:
  python scripts/run_hpc_pipeline.py --stage all --num-samples 102000000 --workers 64
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))


def check_prerequisites(stage: str, dataset_dir: Path, shards_dir: Path, models_path: Path) -> None:
    """Validates prerequisite files and directories before launching a stage."""
    if stage in ["generate", "all"]:
        if not models_path.exists():
            raise FileNotFoundError(
                f"Fitted models bundle not found at: {models_path}\n"
                "Please run: python scripts/fit_models.py first."
            )

    if stage in ["signatures", "all"]:
        ds_files = list(dataset_dir.glob("dataset_*.h5"))
        if not ds_files and stage != "all":
            raise FileNotFoundError(
                f"No dataset shards found in {dataset_dir}.\n"
                "Please run stage 'generate' first: python scripts/run_hpc_pipeline.py --stage generate"
            )

    if stage in ["exp1", "exp2", "exp3", "all_experiments"]:
        sig_files = list(shards_dir.glob("signatures_*.h5"))
        if not sig_files:
            raise FileNotFoundError(
                f"No signature shards found in {shards_dir}.\n"
                "Please run stage 'signatures' first: python scripts/run_hpc_pipeline.py --stage signatures"
            )


def run_command_stream(cmd: List[str], desc: str) -> float:
    """Executes a command synchronously, streaming output to stdout in real-time."""
    print("\n" + "=" * 90)
    print(f"  STARTING: {desc}")
    print(f"  Command:  {' '.join(cmd)}")
    print("=" * 90 + "\n")
    sys.stdout.flush()

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace_root),
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )
    ret = proc.wait()
    elapsed = time.perf_counter() - t0

    if ret != 0:
        print(f"\n[ERROR] Command failed with exit code {ret} after {elapsed:.1f}s.")
        sys.exit(ret)

    print(f"\n[COMPLETED] {desc} finished in {elapsed:.2f}s ({elapsed/60:.2f} min).\n")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Master HPC Pipeline Orchestrator for Duplicate Detection at Scale.")
    parser.add_argument(
        "--stage",
        type=str,
        default="all_experiments",
        choices=["generate", "signatures", "exp1", "exp2", "exp3", "all_experiments", "all"],
        help="Pipeline stage to execute (default: all_experiments)"
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Total observations to generate (e.g. 102000000)")
    parser.add_argument("--num-identities", type=int, default=None, help="Total identities to generate")
    parser.add_argument("--shard-size", type=int, default=150000, help="Observations per shard file")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8, help="Number of parallel worker processes/threads")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for signature extraction")
    parser.add_argument("--max-gallery", type=int, default=None, help="Max gallery records to index in experiments")
    parser.add_argument("--num-queries", type=int, default=100, help="Queries evaluated in Exp 1, Exp 2, and Exp 3")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory containing dataset shards")
    parser.add_argument("--shards-dir", type=str, default="dataset/signatures_shards", help="Directory containing signature shards")
    parser.add_argument("--models-path", type=str, default="models/fitted_models.pkl", help="Fitted models file")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save experiment JSON reports")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    dataset_path = (workspace_root / args.dataset_dir).resolve()
    shards_path = (workspace_root / args.shards_dir).resolve()
    models_path = (workspace_root / args.models_path).resolve()
    reports_path = (workspace_root / args.reports_dir).resolve()
    reports_path.mkdir(parents=True, exist_ok=True)

    check_prerequisites(args.stage, dataset_path, shards_path, models_path)

    print("*" * 90)
    print("                 DUPLICATE DETECTION HPC MASTER ORCHESTRATOR")
    print(f"Stage:                {args.stage.upper()}")
    print(f"CPU Workers:          {args.workers}")
    print(f"Dataset Directory:    {dataset_path}")
    print(f"Signatures Directory: {shards_path}")
    print(f"Reports Directory:    {reports_path}")
    if args.max_gallery:
        print(f"Gallery Limit:        {args.max_gallery:,} records")
    print("*" * 90)

    timings = {}
    total_start = time.perf_counter()

    # -------------------------------------------------------------
    # 1. Generation Stage
    # -------------------------------------------------------------
    if args.stage in ["generate", "all"]:
        gen_cmd = [
            sys.executable, "scripts/generate.py",
            "--models-path", str(models_path),
            "--output-dir", str(dataset_path),
            "--shard-size", str(args.shard_size),
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--resume"
        ]
        if args.num_samples is not None:
            gen_cmd.extend(["--num-samples", str(args.num_samples)])
        elif args.num_identities is not None:
            gen_cmd.extend(["--num-identities", str(args.num_identities)])

        timings["Generation"] = run_command_stream(gen_cmd, "Stage: Synthetic Fingerprint Generation")

    # -------------------------------------------------------------
    # 2. Signatures Stage
    # -------------------------------------------------------------
    if args.stage in ["signatures", "all"]:
        sig_cmd = [
            sys.executable, "scripts/build_signatures_6m.py",
            "--dataset-dir", str(dataset_path),
            "--output-dir", str(shards_path),
            "--num-workers", str(args.workers),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--resume"
        ]
        timings["Signatures"] = run_command_stream(sig_cmd, "Stage: MinHash Signature Extraction")

    # -------------------------------------------------------------
    # 3. Experiment 1: Band Pre-Filter Sweep
    # -------------------------------------------------------------
    if args.stage in ["exp1", "all_experiments", "all"]:
        exp1_out = reports_path / "band_prefilter_sweep.json"
        exp1_cmd = [
            sys.executable, "scripts/run_exp1_band_sweep.py",
            "--dataset-dir", str(dataset_path),
            "--shards-dir", str(shards_path),
            "--num-queries", str(args.num_queries),
            "--seed", str(args.seed),
            "--output", str(exp1_out)
        ]
        if args.max_gallery is not None:
            exp1_cmd.extend(["--max-gallery", str(args.max_gallery)])

        timings["Experiment 1 (Band Sweep)"] = run_command_stream(exp1_cmd, "Experiment 1: LSH Band Bloom Filter Parameter Sweep")

    # -------------------------------------------------------------
    # 4. Experiment 2: Retrieval Benchmark at Scale
    # -------------------------------------------------------------
    if args.stage in ["exp2", "all_experiments", "all"]:
        exp2_out = reports_path / "benchmark_full_scale.json"
        exp2_cmd = [
            sys.executable, "scripts/run_exp2_retrieval_benchmark.py",
            "--dataset-dir", str(dataset_path),
            "--shards-dir", str(shards_path),
            "--num-queries", str(args.num_queries),
            "--seed", str(args.seed),
            "--output", str(exp2_out)
        ]
        if args.max_gallery is not None:
            exp2_cmd.extend(["--max-gallery", str(args.max_gallery)])

        timings["Experiment 2 (Retrieval Benchmark)"] = run_command_stream(exp2_cmd, "Experiment 2: Full-Scale Standard Retrieval Benchmark")

    # -------------------------------------------------------------
    # 5. Experiment 3: 4-Way Latency & Throughput Benchmark
    # -------------------------------------------------------------
    if args.stage in ["exp3", "all_experiments", "all"]:
        exp3_out = reports_path / "latency_4way_results.json"
        exp3_cmd = [
            sys.executable, "scripts/run_exp3_latency_4way.py",
            "--dataset-dir", str(dataset_path),
            "--shards-dir", str(shards_path),
            "--num-queries", str(args.num_queries),
            "--seed", str(args.seed),
            "--output", str(exp3_out)
        ]
        if args.max_gallery is not None:
            exp3_cmd.extend(["--max-gallery", str(args.max_gallery)])

        timings["Experiment 3 (4-Way Latency)"] = run_command_stream(exp3_cmd, "Experiment 3: 4-Way Architectural Latency Benchmark")

    total_time = time.perf_counter() - total_start

    print("\n" + "=" * 90)
    print("                    HPC PIPELINE EXECUTION SUMMARY")
    print("=" * 90)
    for stage_name, el in timings.items():
        print(f"  {stage_name:<40} : {el:>8.2f}s  ({el/60:>5.2f} min)")
    print("-" * 90)
    print(f"  {'TOTAL PIPELINE RUNTIME':<40} : {total_time:>8.2f}s  ({total_time/60:>5.2f} min)")
    print("=" * 90)
    print(f"All reports successfully generated in: {reports_path.resolve()}\n")


if __name__ == "__main__":
    main()
