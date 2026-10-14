import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scipy.spatial.distance import cdist
from src.statistics.calibration import DatasetCalibrator
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)

for q, target_disp, target_ang in [(7, 1.67, 1.87), (5, 1.63, 1.83), (1, 1.37, 1.24), (9, 1.02, 0.84)]:
    jacs = []
    tot_disps = []
    tot_angs = []
    real_jacs = []
    
    for ident in range(1, 26):
        fp3 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
        fpq = calibrator.load_fingerprint_file(f"q{q}_{ident}_1.json")
        if not (fp3 and fpq): continue
        pts = fp3['pts']
        angles = fp3['angles']
        types = fp3['types']
        base_tok = tokenizer.extract_relative_tokens(pts, angles, types)
        real_tok = tokenizer.extract_relative_tokens(fpq['pts'], fpq['angles'], fpq['types'])
        real_jacs.append(len(base_tok & real_tok) / len(base_tok | real_tok))
        
        rng = np.random.default_rng(42 + ident * 101 + q)
        n_ctrl = 5
        cx = np.linspace(20, 396, n_ctrl)
        cy = np.linspace(20, 470, n_ctrl)
        grid_x, grid_y = np.meshgrid(cx, cy)
        ctrl_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
        
        # Scale factor for RBF smoothing: RBF interpolation dampens peak by ~0.4
        # So to get mean displacement = target_disp:
        # ctrl_disp ~ target_disp * 2.2
        ctrl_disp = rng.normal(0.0, target_disp * 2.1, size=ctrl_pts.shape)
        dmat = cdist(pts, ctrl_pts)
        weights = np.exp(-0.5 * (dmat / 75.0)**2)
        weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-12
        smooth_disp = (weights @ ctrl_disp) / weights_sum
        
        # Jitter: 0.35 px independent
        indep_jitter = rng.normal(0.0, 0.35, size=pts.shape)
        deg_pts = pts + smooth_disp + indep_jitter
        
        # Angular field:
        ctrl_ang = rng.normal(0.0, target_ang * 2.1, size=len(ctrl_pts))
        smooth_ang = (weights @ ctrl_ang) / weights_sum.squeeze()
        indep_ang = rng.normal(0.0, 0.45, size=len(pts))
        deg_angles = (angles + smooth_ang + indep_ang) % 360.0
        
        bin_step = 360.0 / 256.0
        deg_angles = np.round(deg_angles / bin_step) * bin_step % 360.0
        
        tok_sim = tokenizer.extract_relative_tokens(deg_pts, deg_angles, types)
        jacs.append(len(base_tok & tok_sim) / len(base_tok | tok_sim))
        tot_disps.append(np.mean(np.linalg.norm(smooth_disp + indep_jitter, axis=1)))
        
        ang_d = np.abs((deg_angles - angles + 180.0) % 360.0 - 180.0)
        tot_angs.append(np.mean(ang_d))
        
    print(f"Quality q{q}:")
    print(f"  Real Token Jaccard:   {np.mean(real_jacs):.4f}")
    print(f"  Synth Token Jaccard:  {np.mean(jacs):.4f}")
    print(f"  Total Disp: target={target_disp:.2f}px, actual={np.mean(tot_disps):.2f}px")
    print(f"  Total Ang:  target={target_ang:.2f}deg, actual={np.mean(tot_angs):.2f}deg")
