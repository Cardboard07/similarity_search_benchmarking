"""
CLI script to fit Model A, Model B_angle, and Model B_quality on the real fingerprint dataset.
Saves model parameters, calibration summaries, and diagnostic plots.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np

from src.statistics.calibration import DatasetCalibrator
from src.models.spatial_model import (
    InhomogeneousPoissonInhibitionModel,
    AdaptiveKDESpatialModel,
    GaussianMixtureSpatialModel
)
from src.models.model_a import ModelA
from src.models.angle_model import AngleModel
from src.models.quality_model import QualityModel
from src.validation.plots import (
    plot_count_distribution,
    plot_spatial_nn_distribution,
    plot_circular_polar_distribution,
    plot_quality_degradation_curves
)

def main():
    parser = argparse.ArgumentParser(description="Fit statistical fingerprint minutiae models.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save fitted models")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = config['dataset']['data_dir']
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "diagnostic_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Calibrating Real Fingerprint Dataset")
    print("=" * 60)
    calibrator = DatasetCalibrator(data_dir)
    calib_results = calibrator.run_full_calibration()
    
    # Save calibration summary
    calib_json = {k: v for k, v in calib_results.items() if k != 'angle_correspondence_results'}
    summary_path = os.path.join(args.output_dir, "calibration_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(calib_json, f, indent=2)
    print(f"Calibration summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("STEP 2: Evaluating Spatial Model Candidates on Held-Out Fingerprints")
    print("=" * 60)
    pristine_fps = calibrator.load_all_pristine()
    np.random.seed(42)
    indices = np.random.permutation(len(pristine_fps))
    split = int(0.8 * len(pristine_fps))
    train_fps = [pristine_fps[i] for i in indices[:split]]
    test_fps = [pristine_fps[i] for i in indices[split:]]
    test_pts = np.vstack([fp['pts'] for fp in test_fps if len(fp['pts']) > 0])

    candidates = {
        'poisson_inhibition': InhomogeneousPoissonInhibitionModel(min_separation=8.5),
        'adaptive_kde': AdaptiveKDESpatialModel(min_separation=8.5),
        'gmm': GaussianMixtureSpatialModel(n_components=8, min_separation=8.5)
    }

    train_pts = [fp['pts'] for fp in train_fps if len(fp['pts']) > 0]
    best_name = None
    best_ll = -np.inf
    print("Cross-validation log-likelihood on 200 held-out fingerprints:")
    for name, model in candidates.items():
        model.fit(train_pts)
        ll = model.evaluate_log_likelihood(test_pts)
        print(f"  Candidate '{name}': Mean log-likelihood = {ll:.4f}")
        if ll > best_ll:
            best_ll = ll
            best_name = name

    print(f"Selected Best Spatial Model: '{best_name}'")
    chosen_spatial_model = candidates[best_name]
    # Refit on all 1,000 pristine fingerprints
    all_pristine_pts = [fp['pts'] for fp in pristine_fps if len(fp['pts']) > 0]
    chosen_spatial_model.fit(all_pristine_pts)

    print("\n" + "=" * 60)
    print("STEP 3: Fitting Model A (Identity Generator)")
    print("=" * 60)
    model_a = ModelA(spatial_model=chosen_spatial_model, min_separation=8.5)
    model_a.fit(pristine_fps)
    print("Model A successfully fitted.")

    print("\n" + "=" * 60)
    print("STEP 4: Fitting Model B_angle (Impression Model)")
    print("=" * 60)
    angle_model = AngleModel()
    angle_model.fit(calib_results['angle_correspondence_results'])
    print(f"Model B_angle fitted: tx_std={angle_model.tx_std:.3f}, ty_std={angle_model.ty_std:.3f}, rot_std={angle_model.rot_std_deg:.3f} deg, scale_std={angle_model.scale_std:.4f}")

    print("\n" + "=" * 60)
    print("STEP 5: Fitting Model B_quality (Quality Degradation Model)")
    print("=" * 60)
    quality_model = QualityModel(use_spatial_field=True)
    quality_model.fit_from_stats(calib_results['quality_stats'])
    print("Model B_quality fitted across levels 7, 5, 1, 9.")

    # Save fitted models bundle
    models_bundle = {
        'model_a': model_a,
        'angle_model': angle_model,
        'quality_model': quality_model,
        'config': config,
        'best_spatial_model_name': best_name
    }
    model_save_path = os.path.join(args.output_dir, "fitted_models.pkl")
    with open(model_save_path, 'wb') as f:
        pickle.dump(models_bundle, f)
    print(f"\nFitted models successfully serialized to: {model_save_path}")

    # Generate initial diagnostic plots
    print("\nGenerating model calibration diagnostic plots...")
    # Generate 100 sample pristine fingerprints from Model A for visual sanity check
    synth_counts = [model_a.generate_identity(seed=1000 + i).num_minutiae for i in range(100)]
    real_counts = np.array([len(fp['pts']) for fp in pristine_fps])
    plot_count_distribution(
        real_counts=real_counts,
        synth_counts=np.array(synth_counts),
        output_path=os.path.join(plots_dir, "count_distribution_fit.png")
    )
    print("Model fitting pipeline complete!")

if __name__ == '__main__':
    main()
