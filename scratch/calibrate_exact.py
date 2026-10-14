import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.statistics.calibration import DatasetCalibrator
from src.models.correspondence import MinutiaMatcher
from src.statistics.circular_stats import fit_von_mises

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
matcher = MinutiaMatcher(spatial_gate=18.0, angular_gate_deg=35.0)

print("Exact calibration across all 1000 identities:")

# Angle model (impressions)
imp_tx, imp_ty, imp_rot, imp_ang_diff = [], [], [], []
for ident in range(1, 1001):
    fp1 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
    if not fp1: continue
    for ori in [2, 3]:
        fpo = calibrator.load_fingerprint_file(f"q3_{ident}_{ori}.json")
        if not fpo: continue
        res = matcher.match(
            pts_a=fp1['pts'], angles_a=fp1['angles'], types_a=fp1['types'],
            pts_b=fpo['pts'], angles_b=fpo['angles'], types_b=fpo['types']
        )
        imp_tx.append(res.estimated_tx)
        imp_ty.append(res.estimated_ty)
        imp_rot.append(res.estimated_rotation_deg)
        imp_ang_diff.extend(res.angular_differences)

print(f"AngleModel tx_std: {np.std(imp_tx):.4f}")
print(f"AngleModel ty_std: {np.std(imp_ty):.4f}")
print(f"AngleModel rot_std: {np.std(imp_rot):.4f}")
print(f"AngleModel ang diff: count={len(imp_ang_diff)}, mean={np.mean(imp_ang_diff):.4f}, median={np.median(imp_ang_diff):.4f}, 95th={np.percentile(imp_ang_diff, 95):.4f}")
_, imp_kappa = fit_von_mises(np.array(imp_ang_diff))
print(f"AngleModel unclipped von Mises kappa: {imp_kappa:.1f}")

# Quality model
for q in [7, 5, 1, 9]:
    survs, falses, pos_diffs, ang_diffs = [], [], [], []
    for ident in range(1, 1001):
        fp3 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
        fq = calibrator.load_fingerprint_file(f"q{q}_{ident}_1.json")
        if not (fp3 and fq): continue
        res = matcher.match(
            pts_a=fp3['pts'], angles_a=fp3['angles'], types_a=fp3['types'],
            pts_b=fq['pts'], angles_b=fq['angles'], types_b=fq['types']
        )
        survs.append(res.survival_rate)
        falses.append(res.false_minutiae_count)
        pos_diffs.extend(res.spatial_distances)
        ang_diffs.extend(res.angular_differences)
        
    pos_arr = np.array(pos_diffs)
    ang_arr = np.array(ang_diffs)
    # 2D Gaussian std from Euclidean distance: mean(r) = std * sqrt(pi/2) => std = mean / 1.2533
    pos_std = float(np.mean(pos_arr) / np.sqrt(np.pi / 2.0))
    _, kappa = fit_von_mises(ang_arr)
    print(f"\nQuality q{q}:")
    print(f"  surv_rate: {np.mean(survs):.4f}")
    print(f"  false_rate: {np.mean(falses):.4f}")
    print(f"  pos_err mean: {np.mean(pos_arr):.4f}px, median: {np.median(pos_arr):.4f}px, implied sigma_pos: {pos_std:.4f}px")
    print(f"  ang_err mean: {np.mean(ang_arr):.4f} deg, median: {np.median(ang_arr):.4f} deg, fitted kappa: {kappa:.1f}")
