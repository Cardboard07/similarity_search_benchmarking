import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import pickle
import numpy as np
from src.statistics.calibration import DatasetCalibrator
from src.statistics.spatial_stats import nearest_neighbor_distances, pairwise_distances

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
pristine_fps = calibrator.load_all_pristine()

real_counts = [len(fp['pts']) for fp in pristine_fps]
real_pts = [fp['pts'] for fp in pristine_fps if len(fp['pts']) > 0]
real_nn = [nearest_neighbor_distances(p, k=1) for p in real_pts]
real_nn_flat = np.concatenate(real_nn)

print("REAL PRISTINE (q3_1):")
print(f"Minutiae count: mean={np.mean(real_counts):.2f}, std={np.std(real_counts):.2f}, min={np.min(real_counts)}, max={np.max(real_counts)}")
print(f"NN distance: mean={np.mean(real_nn_flat):.2f}px, median={np.median(real_nn_flat):.2f}px, min={np.min(real_nn_flat):.2f}px, 5th={np.percentile(real_nn_flat, 5):.2f}px")

# Check fitted model_a in models/fitted_models.pkl
with open("models/fitted_models.pkl", "rb") as f:
    bundle = pickle.load(f)
model_a = bundle['model_a']

synth_fps = [model_a.generate_identity(seed=1000 + i) for i in range(200)]
synth_counts = [s.num_minutiae for s in synth_fps]
synth_nn = [nearest_neighbor_distances(s.pts, k=1) for s in synth_fps]
synth_nn_flat = np.concatenate(synth_nn)

print("\nMODEL A SYNTHETIC PRISTINE:")
print(f"Minutiae count: mean={np.mean(synth_counts):.2f}, std={np.std(synth_counts):.2f}, min={np.min(synth_counts)}, max={np.max(synth_counts)}")
print(f"NN distance: mean={np.mean(synth_nn_flat):.2f}px, median={np.median(synth_nn_flat):.2f}px, min={np.min(synth_nn_flat):.2f}px, 5th={np.percentile(synth_nn_flat, 5):.2f}px")
