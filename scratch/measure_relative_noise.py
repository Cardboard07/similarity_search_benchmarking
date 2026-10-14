import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from src.statistics.calibration import DatasetCalibrator

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")

# Measure pairwise relative distance error: |d_real(i, j) - d_noisy(i, j)|
rel_dist_errors = []
rel_ang_errors = []
abs_pos_errors = []
abs_ang_errors = []

for ident in range(1, 21):
    f3 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
    f7 = calibrator.load_fingerprint_file(f"q7_{ident}_1.json")
    if not (f3 and f7): continue
    
    # Match minutiae by nearest distance
    p3 = f3['pts']
    p7 = f7['pts']
    a3 = f3['angles']
    a7 = f7['angles']
    
    # For points that match within 5 px
    from scipy.spatial.distance import cdist
    dmat = cdist(p3, p7)
    matches = []
    for i in range(len(p3)):
        j = np.argmin(dmat[i])
        if dmat[i, j] < 5.0:
            matches.append((i, j))
            abs_pos_errors.append(dmat[i, j])
            ang_diff = abs(a3[i] - a7[j])
            ang_diff = min(ang_diff, 360.0 - ang_diff)
            abs_ang_errors.append(ang_diff)
            
    # For pairs of matched minutiae
    for idx_a in range(len(matches)):
        for idx_b in range(idx_a + 1, len(matches)):
            i1, j1 = matches[idx_a]
            i2, j2 = matches[idx_b]
            dist3 = np.linalg.norm(p3[i1] - p3[i2])
            dist7 = np.linalg.norm(p7[j1] - p7[j2])
            if dist3 < 60.0: # Local neighbors (within 60 px)
                rel_dist_errors.append(abs(dist3 - dist7))
                delta3 = (a3[i2] - a3[i1]) % 360.0
                delta7 = (a7[j2] - a7[j1]) % 360.0
                rel_ang = abs(delta3 - delta7)
                rel_ang = min(rel_ang, 360.0 - rel_ang)
                rel_ang_errors.append(rel_ang)

print("Absolute errors on individual minutiae:")
print(f"  Pos error: mean = {np.mean(abs_pos_errors):.3f} px, median = {np.median(abs_pos_errors):.3f} px")
print(f"  Ang error: mean = {np.mean(abs_ang_errors):.3f} deg, median = {np.median(abs_ang_errors):.3f} deg")

print("\nRelative errors between local neighbors (< 60 px):")
print(f"  Relative dist change: mean = {np.mean(rel_dist_errors):.3f} px, median = {np.median(rel_dist_errors):.3f} px, 90th = {np.percentile(rel_dist_errors, 90):.3f} px")
print(f"  Relative ang change: mean = {np.mean(rel_ang_errors):.3f} deg, median = {np.median(rel_ang_errors):.3f} deg, 90th = {np.percentile(rel_ang_errors, 90):.3f} deg")
