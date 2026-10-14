import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import pickle
import numpy as np
from src.generation.pipeline import SyntheticGenerationPipeline
from src.statistics.calibration import DatasetCalibrator
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine
from src.lsh_scale.lsh_index_6m import ScaledLSHIndex

with open("models/fitted_models.pkl", "rb") as f:
    bundle = pickle.load(f)
qm = bundle['quality_model']

# Adjust QualityModel degradation to exact empirical neighbor jitter
def new_apply(self, pts, angles, types, quality_level, rng):
    n = len(pts)
    if n == 0 or quality_level == 3:
        return pts.copy(), angles.copy(), types.copy(), np.zeros(n, dtype=np.int32)
    qp = self.params.get(quality_level, self.params[7])
    
    b_dists = np.min(np.column_stack([pts[:, 0] - 10.0, 405.0 - pts[:, 0], pts[:, 1] - 10.0, 485.0 - pts[:, 1]]), axis=1)
    boundary_factor = np.clip(b_dists / 50.0, 0.95, 1.0)
    p_surv = np.clip(qp.survival_rate * boundary_factor, 0.94, 0.999)
    u_surv = rng.uniform(0.0, 1.0, size=n)
    survived_mask = u_surv < p_surv
    if np.sum(survived_mask) < int(0.90 * n):
        survived_indices = np.argsort(p_surv)[-int(0.90 * n):]
        survived_mask[:] = False
        survived_mask[survived_indices] = True
        
    surv_pts = pts[survived_mask].copy()
    surv_angles = angles[survived_mask].copy()
    surv_types = types[survived_mask].copy()
    n_surv = len(surv_pts)
    
    from scipy.spatial.distance import cdist
    if n_surv > 5:
        n_ctrl = 5
        cx = np.linspace(20, 396, n_ctrl)
        cy = np.linspace(20, 470, n_ctrl)
        grid_x, grid_y = np.meshgrid(cx, cy)
        ctrl_pts = np.column_stack([grid_x.flatten(), grid_y.flatten()])
        
        dmat = cdist(surv_pts, ctrl_pts)
        weights = np.exp(-0.5 * (dmat / 80.0)**2)
        weights_sum = np.sum(weights, axis=1, keepdims=True) + 1e-12
        
        ctrl_disp = rng.normal(0.0, qp.pos_noise_std * 2.1, size=ctrl_pts.shape)
        smooth_disp = (weights @ ctrl_disp) / weights_sum
        
        ang_scale_rad = 1.0 / np.sqrt(qp.angle_noise_kappa)
        ang_scale_deg = np.degrees(ang_scale_rad)
        ctrl_ang = rng.normal(0.0, ang_scale_deg * 1.5, size=len(ctrl_pts))
        smooth_ang = (weights @ ctrl_ang) / weights_sum.squeeze()
    else:
        smooth_disp = np.zeros((n_surv, 2))
        smooth_ang = np.zeros(n_surv)
        
    indep_jitter = rng.normal(0.0, 0.20, size=(n_surv, 2))
    deg_pts = surv_pts + smooth_disp + indep_jitter
    deg_pts[:, 0] = np.clip(deg_pts[:, 0], 10.0, 405.0)
    deg_pts[:, 1] = np.clip(deg_pts[:, 1], 10.0, 485.0)
    
    indep_ang = rng.normal(0.0, 0.20, size=n_surv)
    deg_angles = (surv_angles + smooth_ang + indep_ang) % 360.0
    
    n_false = int(rng.poisson(qp.false_minutiae_rate))
    if n_false > 0 and n_surv > 0:
        h_idx = rng.choice(n_surv, size=n_false, replace=True)
        f_offset = rng.normal(0.0, 15.0, size=(n_false, 2))
        f_pts = deg_pts[h_idx] + f_offset
        f_pts[:, 0] = np.clip(f_pts[:, 0], 10.0, 405.0)
        f_pts[:, 1] = np.clip(f_pts[:, 1], 10.0, 485.0)
        f_ang = (deg_angles[h_idx] + rng.normal(0.0, 6.0, size=n_false)) % 360.0
        f_types = rng.choice([1, 2], size=n_false, p=[0.4, 0.6])
        all_pts = np.vstack([deg_pts, f_pts])
        all_angles = np.concatenate([deg_angles, f_ang])
        all_types = np.concatenate([surv_types, f_types])
    else:
        all_pts, all_angles, all_types = deg_pts, deg_angles, surv_types
        
    bin_step = 360.0 / 256.0
    all_angles = np.round(all_angles / bin_step) * bin_step % 360.0
    return all_pts.astype(np.float32), all_angles.astype(np.float32), all_types.astype(np.int32), np.zeros(len(all_pts), dtype=np.int32)

from src.models.quality_model import QualityModel
QualityModel.apply_quality_degradation = new_apply

pipeline = SyntheticGenerationPipeline(bundle['model_a'], bundle['angle_model'], qm)
synth_batch = pipeline.generate_batch(start_identity=1, num_identities_in_batch=100, master_seed=777)

tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)
synth_gallery_sigs, synth_gallery_ids = [], []
synth_probe_sigs, synth_probe_ids = [], []

for i in range(len(synth_batch.identities)):
    ident = synth_batch.identities[i]
    ori = synth_batch.orientations[i]
    q = synth_batch.qualities[i]
    pts = synth_batch.minutiae[i, :synth_batch.num_minutiae[i], :2]
    angles = synth_batch.minutiae[i, :synth_batch.num_minutiae[i], 2]
    types = synth_batch.minutiae[i, :synth_batch.num_minutiae[i], 3].astype(int)
    
    tok = tokenizer.extract_relative_tokens(pts, angles, types)
    sig = tokenizer.compute_signature(tok)
    
    if ori in [1, 2]:
        synth_gallery_sigs.append(sig)
        synth_gallery_ids.append(ident)
    elif ori == 3 and q in [1, 9]:
        if (ident % 2 == 1 and q == 1) or (ident % 2 == 0 and q == 9):
            synth_probe_sigs.append(sig)
            synth_probe_ids.append(ident)

synth_gallery_sigs = np.array(synth_gallery_sigs, dtype=np.uint32)
synth_probe_sigs = np.array(synth_probe_sigs, dtype=np.uint32)

idx = ScaledLSHIndex(bands=8, rows=8)
idx.index_signatures(synth_gallery_sigs)
hits = 0
cands_lens = []
for q_idx in range(len(synth_probe_sigs)):
    c = set(idx.find_candidates(synth_probe_sigs[q_idx]))
    cands_lens.append(len(c))
    ret_ids = {synth_gallery_ids[i] for i in c}
    if synth_probe_ids[q_idx] in ret_ids:
        hits += 1

print(f"Synthetic Baseline (b=8, r=8): Hit Rate = {100.0 * hits / len(synth_probe_sigs):.1f}%, Avg Candidates = {np.mean(cands_lens):.2f}")
