import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from scipy.spatial.distance import cdist
from src.models.quality_model import QualityModel, QualityParameters
from src.statistics.calibration import DatasetCalibrator
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine
from src.models.correspondence import MinutiaMatcher

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)
matcher = MinutiaMatcher(spatial_gate=18.0, angular_gate_deg=35.0)

# Build improved QualityModel
qm = QualityModel()
qm.params = {
    3: QualityParameters(quality_level=3, pos_noise_std=0.0, angle_noise_kappa=100000.0, survival_rate=1.0, false_minutiae_rate=0.0),
    7: QualityParameters(quality_level=7, pos_noise_std=0.80, angle_noise_kappa=686.0, survival_rate=0.990, false_minutiae_rate=0.51),
    5: QualityParameters(quality_level=5, pos_noise_std=0.78, angle_noise_kappa=533.0, survival_rate=0.988, false_minutiae_rate=0.60),
    1: QualityParameters(quality_level=1, pos_noise_std=0.65, angle_noise_kappa=994.0, survival_rate=0.982, false_minutiae_rate=0.82),
    9: QualityParameters(quality_level=9, pos_noise_std=0.49, angle_noise_kappa=1301.0, survival_rate=0.981, false_minutiae_rate=0.86),
}

# Override apply_quality_degradation
def new_apply(self, pts, angles, types, quality_level, rng):
    n = len(pts)
    if n == 0 or quality_level == 3:
        return pts.copy(), angles.copy(), types.copy(), np.zeros(n, dtype=np.int32)
    qp = self.params.get(quality_level, self.params[7])
    
    # 1. Survival
    b_dists = np.min(np.column_stack([pts[:, 0] - 10.0, 405.0 - pts[:, 0], pts[:, 1] - 10.0, 485.0 - pts[:, 1]]), axis=1)
    boundary_factor = np.clip(b_dists / 50.0, 0.95, 1.0)
    p_surv = np.clip(qp.survival_rate * boundary_factor, 0.92, 0.999)
    u_surv = rng.uniform(0.0, 1.0, size=n)
    survived_mask = u_surv < p_surv
    if np.sum(survived_mask) < int(0.85 * n):
        survived_indices = np.argsort(p_surv)[-int(0.85 * n):]
        survived_mask[:] = False
        survived_mask[survived_indices] = True
        
    surv_pts = pts[survived_mask].copy()
    surv_angles = angles[survived_mask].copy()
    surv_types = types[survived_mask].copy()
    n_surv = len(surv_pts)
    
    # 2. Smooth Fields
    if n_surv > 5:
        n_ctrl = 5
        cx = np.linspace(20, 396, n_ctrl)
        cy = np.linspace(20, 470, n_ctrl)
        grid_x, grid_y = np.meshgrid(cx, cy)
        ctrl_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
        
        dmat = cdist(surv_pts, ctrl_pts)
        weights = np.exp(-0.5 * (dmat / 75.0)**2)
        weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-12
        
        # Smooth displacement
        ctrl_disp = rng.normal(0.0, qp.pos_noise_std * 2.1, size=ctrl_pts.shape)
        smooth_disp = (weights @ ctrl_disp) / weights_sum
        
        # Smooth angle
        ang_scale_rad = 1.0 / np.sqrt(qp.angle_noise_kappa)
        ang_scale_deg = np.degrees(ang_scale_rad)
        ctrl_ang = rng.normal(0.0, ang_scale_deg * 1.5, size=len(ctrl_pts))
        smooth_ang = (weights @ ctrl_ang) / weights_sum.squeeze()
    else:
        smooth_disp = np.zeros((n_surv, 2))
        smooth_ang = np.zeros(n_surv)
        
    # Jitter
    indep_jitter = rng.normal(0.0, 0.35, size=(n_surv, 2))
    deg_pts = surv_pts + smooth_disp + indep_jitter
    deg_pts[:, 0] = np.clip(deg_pts[:, 0], 10.0, 405.0)
    deg_pts[:, 1] = np.clip(deg_pts[:, 1], 10.0, 485.0)
    
    indep_ang = rng.normal(0.0, 0.40, size=n_surv)
    deg_angles = (surv_angles + smooth_ang + indep_ang) % 360.0
    
    # False minutiae
    n_false = int(rng.poisson(qp.false_minutiae_rate))
    if n_false > 0 and n_surv > 0:
        h_idx = rng.choice(n_surv, size=n_false, replace=True)
        f_offset = rng.normal(0.0, 15.0, size=(n_false, 2))
        f_pts = deg_pts[h_idx] + f_offset
        f_pts[:, 0] = np.clip(f_pts[:, 0], 10.0, 405.0)
        f_pts[:, 1] = np.clip(f_pts[:, 1], 10.0, 485.0)
        f_ang = (deg_angles[h_idx] + rng.normal(0.0, 8.0, size=n_false)) % 360.0
        f_types = rng.choice([1, 2], size=n_false, p=[0.4, 0.6])
        all_pts = np.vstack([deg_pts, f_pts])
        all_angles = np.concatenate([deg_angles, f_ang])
        all_types = np.concatenate([surv_types, f_types])
    else:
        all_pts, all_angles, all_types = deg_pts, deg_angles, surv_types
        
    bin_step = 360.0 / 256.0
    all_angles = np.round(all_angles / bin_step) * bin_step % 360.0
    return all_pts.astype(np.float32), all_angles.astype(np.float32), all_types.astype(np.int32), np.zeros(len(all_pts), dtype=np.int32)

QualityModel.apply_quality_degradation = new_apply

print("Testing new QualityModel on 50 identities:")
for q in [7, 5, 1, 9]:
    r_jacs, s_jacs, r_mhs, s_mhs, r_disps, s_disps, r_angs, s_angs = [], [], [], [], [], [], [], []
    for ident in range(1, 51):
        fp3 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
        fpq = calibrator.load_fingerprint_file(f"q{q}_{ident}_1.json")
        if not (fp3 and fpq): continue
        t3 = tokenizer.extract_relative_tokens(fp3['pts'], fp3['angles'], fp3['types'])
        tq = tokenizer.extract_relative_tokens(fpq['pts'], fpq['angles'], fpq['types'])
        s3 = tokenizer.compute_signature(t3)
        sq = tokenizer.compute_signature(tq)
        r_jacs.append(len(t3 & tq) / len(t3 | tq))
        r_mhs.append(np.mean(s3 == sq))
        
        # Simulate
        rng = np.random.default_rng(1000 + ident * 17 + q)
        sim_pts, sim_angles, sim_types, _ = qm.apply_quality_degradation(fp3['pts'], fp3['angles'], fp3['types'], q, rng)
        tsim = tokenizer.extract_relative_tokens(sim_pts, sim_angles, sim_types)
        ssim = tokenizer.compute_signature(tsim)
        s_jacs.append(len(t3 & tsim) / len(t3 | tsim))
        s_mhs.append(np.mean(s3 == ssim))
        
        res_r = matcher.match(fp3['pts'], fp3['angles'], fp3['types'], fpq['pts'], fpq['angles'], fpq['types'])
        res_s = matcher.match(fp3['pts'], fp3['angles'], fp3['types'], sim_pts, sim_angles, sim_types)
        if len(res_r.spatial_distances) > 0: r_disps.extend(res_r.spatial_distances)
        if len(res_s.spatial_distances) > 0: s_disps.extend(res_s.spatial_distances)
        if len(res_r.angular_differences) > 0: r_angs.extend(res_r.angular_differences)
        if len(res_s.angular_differences) > 0: s_angs.extend(res_s.angular_differences)
        
    print(f"\n--- Quality Level q{q} ---")
    print(f"  Pos Disp:    Real = {np.mean(r_disps):.3f}px | Synth = {np.mean(s_disps):.3f}px")
    print(f"  Ang Diff:    Real = {np.mean(r_angs):.3f}deg | Synth = {np.mean(s_angs):.3f}deg")
    print(f"  Jaccard:     Real = {np.mean(r_jacs):.4f}  | Synth = {np.mean(s_jacs):.4f}")
    print(f"  MinHash Eq:  Real = {np.mean(r_mhs):.4f}  | Synth = {np.mean(s_mhs):.4f}")
