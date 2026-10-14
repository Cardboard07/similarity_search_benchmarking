import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.statistics.calibration import DatasetCalibrator
from src.models.correspondence import MinutiaMatcher
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)

# Check distribution of pos errors and ang errors in REAL data
all_real_pos_errs = []
all_real_ang_errs = []

matcher = MinutiaMatcher(spatial_gate=18.0, angular_gate_deg=35.0)
for ident in range(1, 51):
    fp3 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
    fq = calibrator.load_fingerprint_file(f"q7_{ident}_1.json")
    if fp3 and fq:
        res = matcher.match(fp3['pts'], fp3['angles'], fp3['types'], fq['pts'], fq['angles'], fq['types'])
        all_real_pos_errs.extend(res.spatial_distances)
        all_real_ang_errs.extend(res.angular_differences)

pos_arr = np.array(all_real_pos_errs)
ang_arr = np.array(all_real_ang_errs)

print("REAL q7 vs q3 error percentiles:")
print("Pos error percentiles (px):")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}%: {np.percentile(pos_arr, p):.3f} px")

print("Ang error percentiles (deg):")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}%: {np.percentile(ang_arr, p):.3f} deg")

# Controlled ablation on a single real print to see what drops Token Jaccard from 0.71 to 0.37
fp = calibrator.load_fingerprint_file("q3_1_1.json")
pts = fp['pts']
angles = fp['angles']
types = fp['types']
base_tokens = tokenizer.extract_relative_tokens(pts, angles, types)

# Case 1: Real q7 file directly
fp_q7 = calibrator.load_fingerprint_file("q7_1_1.json")
q7_tokens = tokenizer.extract_relative_tokens(fp_q7['pts'], fp_q7['angles'], fp_q7['types'])
print("\nControlled Ablations on Identity 1:")
print(f"Real q7 token Jaccard: {len(base_tokens & q7_tokens) / len(base_tokens | q7_tokens):.4f}")

# Case 2: Only empirical position noise (resampled from pos_arr with random directions)
rng = np.random.default_rng(42)
sampled_pos_mags = rng.choice(pos_arr, size=len(pts))
angles_dir = rng.uniform(0, 2*np.pi, size=len(pts))
pts_pos_only = pts + np.column_stack([sampled_pos_mags * np.cos(angles_dir), sampled_pos_mags * np.sin(angles_dir)])
tok_pos_only = tokenizer.extract_relative_tokens(pts_pos_only, angles, types)
print(f"Empirical Pos noise ONLY: {len(base_tokens & tok_pos_only) / len(base_tokens | tok_pos_only):.4f}")

# Case 3: Gaussian position noise with std = 1.36 px
pts_gauss_pos = pts + rng.normal(0, 1.36, size=pts.shape)
tok_gauss_pos = tokenizer.extract_relative_tokens(pts_gauss_pos, angles, types)
print(f"Gaussian Pos noise (std=1.36) ONLY: {len(base_tokens & tok_gauss_pos) / len(base_tokens | tok_gauss_pos):.4f}")

# Case 4: Empirical Angular noise ONLY
sampled_ang_diffs = rng.choice(ang_arr, size=len(pts)) * rng.choice([-1, 1], size=len(pts))
angles_only = (angles + sampled_ang_diffs) % 360.0
tok_ang_only = tokenizer.extract_relative_tokens(pts, angles_only, types)
print(f"Empirical Ang noise ONLY: {len(base_tokens & tok_ang_only) / len(base_tokens | tok_ang_only):.4f}")

# Case 5: Both empirical pos + ang noise
tok_both_emp = tokenizer.extract_relative_tokens(pts_pos_only, angles_only, types)
print(f"Empirical Pos + Ang noise: {len(base_tokens & tok_both_emp) / len(base_tokens | tok_both_emp):.4f}")

# Case 6: Adding false minutiae
print("\nFalse minutiae effect:")
for false_m in [1, 2, 3]:
    h_idx = rng.choice(len(pts), size=false_m)
    f_pts = pts[h_idx] + rng.normal(0, 12.0, size=(false_m, 2))
    f_ang = angles[h_idx]
    f_ty = types[h_idx]
    pts_with_f = np.vstack([pts, f_pts])
    ang_with_f = np.concatenate([angles, f_ang])
    ty_with_f = np.concatenate([types, f_ty])
    tok_f = tokenizer.extract_relative_tokens(pts_with_f, ang_with_f, ty_with_f)
    print(f"  With {false_m} false minutiae added: Jaccard = {len(base_tokens & tok_f) / len(base_tokens | tok_f):.4f}")
