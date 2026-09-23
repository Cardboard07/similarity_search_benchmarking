"""
CLI validation suite for synthetic fingerprint minutiae.
Evaluates statistical fidelity, identity preservation, novelty against real fingerprints,
and produces comprehensive validation reports and figures.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.statistics.calibration import DatasetCalibrator
from src.statistics.spatial_stats import nearest_neighbor_distances, pairwise_distances
from src.storage.manifest import DatasetManifest
from src.storage.hdf5_reader import HDF5ShardReader
from src.validation.statistical import StatisticalValidator
from src.validation.identity import IdentityPreservationValidator
from src.validation.novelty import NoveltyValidator
from src.validation.scale_validation import StreamingScaleValidator
from src.validation.plots import (
    plot_count_distribution,
    plot_spatial_nn_distribution,
    plot_pairwise_distribution,
    plot_circular_polar_distribution,
    plot_synthetic_fingerprint_sample,
    plot_quality_degradation_curves
)

def main():
    parser = argparse.ArgumentParser(description="Validate synthetic fingerprint minutiae datasets.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--models-path", type=str, default="models/fitted_models.pkl", help="Path to fitted models")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Directory containing generated HDF5 dataset")
    parser.add_argument("--output-dir", type=str, default="reports", help="Directory to save reports and figures")
    parser.add_argument("--sample-identities", type=int, default=200, help="Number of synthetic identities to sample for deep geometric validation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading Real Pristine Population & Target Calibration")
    print("=" * 60)
    calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
    real_pristine = calibrator.load_all_pristine()
    
    real_counts = np.array([len(fp['pts']) for fp in real_pristine])
    real_nn = []
    real_pw = []
    real_angles = []
    real_types = []
    for fp in real_pristine:
        pts = fp['pts']
        if len(pts) > 0:
            real_nn.extend(nearest_neighbor_distances(pts, k=1))
            real_pw.extend(pairwise_distances(pts, max_pairs=30))
            real_angles.extend(fp['angles'])
            real_types.extend(fp['types'])

    real_nn = np.array(real_nn)
    real_pw = np.array(real_pw)
    real_angles = np.array(real_angles)
    real_types = np.array(real_types)

    print("=" * 60)
    print("STEP 2: Reading Synthetic Dataset via Manifest")
    print("=" * 60)
    manifest = DatasetManifest(args.dataset_dir)
    if not manifest.shards:
        raise RuntimeError(f"No shards found in dataset directory: {args.dataset_dir}. Run generate.py first.")

    # Sample representative identities uniformly across shards (up to 1,000 pristine identities)
    synth_pristine = []
    synth_by_identity = {} # id -> list of 15 obs
    
    completed_shards = [s for s in manifest.shards if s.status == "completed"]
    ids_per_shard_sample = max(1, 1000 // len(completed_shards))
    
    print(f"Sampling ~{ids_per_shard_sample} identities from each of {len(completed_shards)} shards...")
    for shard in completed_shards:
        shard_path = os.path.join(args.dataset_dir, shard.filename)
        with HDF5ShardReader(shard_path) as reader:
            # Each identity has 15 observations. Read first N identities in this shard
            sample_ids_count = min(ids_per_shard_sample, shard.num_identities)
            for id_idx in range(sample_ids_count):
                obs_offset = id_idx * 15
                id_samples = []
                for o_idx in range(15):
                    sample = reader.get_sample(obs_offset + o_idx)
                    id_samples.append(sample)
                    if sample['orientation'] == 1 and sample['quality'] == 3:
                        synth_pristine.append(sample)
                ident = id_samples[0]['identity']
                synth_by_identity[ident] = id_samples

    print(f"Loaded {len(synth_pristine)} synthetic pristine identities from {len(completed_shards)} HDF5 shards.")

    synth_counts = np.array([fp['num_minutiae'] for fp in synth_pristine])
    synth_nn = []
    synth_pw = []
    synth_angles = []
    synth_types = []
    for fp in synth_pristine:
        pts = fp['pts']
        if len(pts) > 0:
            synth_nn.extend(nearest_neighbor_distances(pts, k=1))
            synth_pw.extend(pairwise_distances(pts, max_pairs=30))
            synth_angles.extend(fp['angles'])
            synth_types.extend(fp['types'])

    synth_nn = np.array(synth_nn)
    synth_pw = np.array(synth_pw)
    synth_angles = np.array(synth_angles)
    synth_types = np.array(synth_types)

    print("\n" + "=" * 60)
    print("STEP 3: Quantitative Statistical Fidelity Tests")
    print("=" * 60)
    # 1. Minutiae count comparison
    comp_count = StatisticalValidator.compare_1d_continuous(real_counts, synth_counts, name="minutiae_count")
    # 2. Nearest neighbor distance comparison
    comp_nn = StatisticalValidator.compare_1d_continuous(real_nn, synth_nn, name="nearest_neighbor_distance")
    # 3. Pairwise distance comparison
    comp_pw = StatisticalValidator.compare_1d_continuous(real_pw, synth_pw, name="pairwise_distance")
    # 4. Discrete type comparison
    comp_type = StatisticalValidator.compare_discrete_proportions(real_types, synth_types, name="minutiae_types")
    # 5. Circular angles comparison
    comp_angles = StatisticalValidator.compare_circular_distributions(real_angles, synth_angles, name="orientations")

    print(f"Minutiae Count:     KS={comp_count.ks_statistic:.4f} (p={comp_count.ks_p_value:.4f}), Wasserstein={comp_count.wasserstein_distance:.3f}, Overlap={comp_count.histogram_intersection*100:.1f}%")
    print(f"NN Distance (1-NN): KS={comp_nn.ks_statistic:.4f} (p={comp_nn.ks_p_value:.4f}), Wasserstein={comp_nn.wasserstein_distance:.3f}, Overlap={comp_nn.histogram_intersection*100:.1f}%")
    print(f"Pairwise Distance:  KS={comp_pw.ks_statistic:.4f} (p={comp_pw.ks_p_value:.4f}), Wasserstein={comp_pw.wasserstein_distance:.3f}, Overlap={comp_pw.histogram_intersection*100:.1f}%")
    print(f"Types Ratio:        Real Type 1={comp_type['real_type1_ratio']:.3f}, Synth Type 1={comp_type['synthetic_type1_ratio']:.3f}, JS={comp_type['jensen_shannon_divergence']:.4f}")
    print(f"Orientations:       Real Mean={comp_angles['real_circular_mean_deg']:.1f} deg, Synth Mean={comp_angles['synthetic_circular_mean_deg']:.1f} deg, Overlap={comp_angles['circular_histogram_overlap']*100:.1f}%")

    print("\n" + "=" * 60)
    print("STEP 4: Identity Preservation Verification across 15 Observations")
    print("=" * 60)
    id_validator = IdentityPreservationValidator()
    eval_ids = list(synth_by_identity.keys())[:min(args.sample_identities, len(synth_by_identity))]
    eval_obs_groups = [synth_by_identity[i] for i in eval_ids]
    
    id_pres_results = id_validator.compute_intra_vs_inter_class_scores(eval_obs_groups)
    print(f"Intra-Identity Match Rate: Mean={id_pres_results['intra_match_rate_mean']*100:.1f}%, Min={id_pres_results['intra_match_rate_min']*100:.1f}%")
    print(f"Inter-Identity Match Rate: Mean={id_pres_results['inter_match_rate_mean']*100:.1f}%, Max={id_pres_results['inter_match_rate_max']*100:.1f}%")
    print(f"Separability d-prime:      {id_pres_results['d_prime_separability']:.2f} (High separability confirms identity preservation)")

    print("\n" + "=" * 60)
    print("STEP 5: Novelty & Anti-Memorization Analysis against 1,000 Real Fingerprints")
    print("=" * 60)
    novelty_val = NoveltyValidator(real_pristine_fingerprints=real_pristine, copy_threshold=0.65)
    sample_pristine = synth_pristine[:min(args.sample_identities, len(synth_pristine))]
    novelty_summary = novelty_val.evaluate_batch_novelty(sample_pristine)
    print(f"Identities evaluated:      {novelty_summary['num_evaluated']}")
    print(f"Novelty Percentage:        {novelty_summary['novel_percentage']:.1f}% (All novel: {novelty_summary['all_novel']})")
    print(f"Max Similarity to Real:    {novelty_summary['max_similarity_observed']*100:.1f}% (Threshold: {novelty_summary['copy_threshold']*100:.1f}%)")
    print(f"Mean Similarity to Real:   {novelty_summary['mean_similarity_to_nearest']*100:.1f}%")

    # Generate figures
    print("\n" + "=" * 60)
    print("STEP 6: Generating Diagnostic Figures")
    print("=" * 60)
    plot_count_distribution(real_counts, synth_counts, os.path.join(figures_dir, "minutiae_count_comparison.png"))
    plot_spatial_nn_distribution(real_nn, synth_nn, os.path.join(figures_dir, "nearest_neighbor_comparison.png"))
    plot_pairwise_distribution(real_pw, synth_pw, os.path.join(figures_dir, "pairwise_distance_comparison.png"))
    plot_circular_polar_distribution(real_angles, synth_angles, os.path.join(figures_dir, "orientation_polar_comparison.png"))

    # Plot sample synthetic fingerprint
    sample_id = synth_pristine[0]
    plot_synthetic_fingerprint_sample(
        sample_id['pts'],
        sample_id['angles'],
        sample_id['types'],
        title=f"Synthetic Pristine Fingerprint (ID: {sample_id['identity']}, N={sample_id['num_minutiae']})",
        output_path=os.path.join(figures_dir, "synthetic_fingerprint_sample.png")
    )

    # Save full machine-readable validation report
    report = {
        'count_comparison': comp_count.__dict__,
        'nn_comparison': comp_nn.__dict__,
        'pairwise_comparison': comp_pw.__dict__,
        'type_comparison': comp_type,
        'angles_comparison': comp_angles,
        'identity_preservation': id_pres_results,
        'novelty_analysis': novelty_summary,
        'success_criteria': {
            'novelty_passed': bool(novelty_summary['all_novel']),
            'statistical_count_passed': bool(comp_count.wasserstein_distance < 3.0),
            'spatial_nn_passed': bool(comp_nn.wasserstein_distance < 4.0),
            'identity_preservation_passed': bool(id_pres_results['d_prime_separability'] > 3.0)
        }
    }

    report_path = os.path.join(args.output_dir, "validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation report saved to: {report_path}")
    print(f"Diagnostic figures saved to: {figures_dir}")
    print("=" * 60)
    print("VALIDATION STATUS: " + ("ALL CRITERIA PASSED!" if all(report['success_criteria'].values()) else "SOME METRICS WARN"))
    print("=" * 60)

if __name__ == '__main__':
    main()
