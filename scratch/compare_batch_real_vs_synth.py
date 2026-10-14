import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import pickle
import numpy as np
from src.generation.pipeline import SyntheticGenerationPipeline
from src.statistics.calibration import DatasetCalibrator
from src.models.correspondence import MinutiaMatcher
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine
from src.statistics.spatial_stats import nearest_neighbor_distances, pairwise_distances

print("=" * 70)
print("ROUND 1: DIRECT COMPARISON OF REAL DATA VS RECALIBRATED SYNTHETIC DATA")
print("=" * 70)

# 1. Load fitted models & pipeline
with open("models/fitted_models.pkl", "rb") as f:
    bundle = pickle.load(f)
model_a = bundle['model_a']
angle_model = bundle['angle_model']
quality_model = bundle['quality_model']

pipeline = SyntheticGenerationPipeline(model_a, angle_model, quality_model)
tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)
matcher = MinutiaMatcher(spatial_gate=18.0, angular_gate_deg=35.0)

# Generate small batch: 100 identities (1500 observations)
batch = pipeline.generate_batch(start_identity=1, num_identities_in_batch=100, master_seed=42)
print(f"Generated synthetic batch: {batch.num_identities} identities, {batch.num_observations} observations")

# Load real dataset
calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")

# -------------------------------------------------------------
# A. Spatial & Minutiae Population Statistics (Pristine q3_1)
# -------------------------------------------------------------
real_pristine = calibrator.load_all_pristine()[:100]
real_counts = [len(fp['pts']) for fp in real_pristine]
real_pts_list = [fp['pts'] for fp in real_pristine]
real_nn = np.concatenate([nearest_neighbor_distances(p, k=1) for p in real_pts_list])
real_pw = np.concatenate([pairwise_distances(p, max_pairs=30) for p in real_pts_list])

# Filter synthetic pristine (quality=3, orientation=1)
synth_pristine_mask = (batch.qualities == 3) & (batch.orientations == 1)
synth_pristine_indices = np.where(synth_pristine_mask)[0]
synth_counts = batch.num_minutiae[synth_pristine_indices]
synth_pts_list = [batch.minutiae[idx, :batch.num_minutiae[idx], :2] for idx in synth_pristine_indices]
synth_angles_list = [batch.minutiae[idx, :batch.num_minutiae[idx], 2] for idx in synth_pristine_indices]
synth_types_list = [batch.minutiae[idx, :batch.num_minutiae[idx], 3].astype(int) for idx in synth_pristine_indices]
synth_nn = np.concatenate([nearest_neighbor_distances(p, k=1) for p in synth_pts_list])
synth_pw = np.concatenate([pairwise_distances(p, max_pairs=30) for p in synth_pts_list])

print("\n[A] PRISTINE POPULATION COMPARISON (q3, ori 1):")
print(f"{'Metric':<28} | {'Real Data':<18} | {'Synthetic Data':<18}")
print("-" * 70)
print(f"{'Minutiae Count (Mean ± Std)':<28} | {np.mean(real_counts):.2f} ± {np.std(real_counts):.2f}        | {np.mean(synth_counts):.2f} ± {np.std(synth_counts):.2f}")
print(f"{'Minutiae Count [Min, Max]':<28} | [{np.min(real_counts)}, {np.max(real_counts)}]             | [{np.min(synth_counts)}, {np.max(synth_counts)}]")
print(f"{'NN Distance (Mean ± Std)':<28} | {np.mean(real_nn):.2f} ± {np.std(real_nn):.2f} px     | {np.mean(synth_nn):.2f} ± {np.std(synth_nn):.2f} px")
print(f"{'NN Distance (Median)':<28} | {np.median(real_nn):.2f} px            | {np.median(synth_nn):.2f} px")
print(f"{'Pairwise Distance (Mean)':<28} | {np.mean(real_pw):.2f} px           | {np.mean(synth_pw):.2f} px")

# -------------------------------------------------------------
# B. Impression / Orientation Variations (ori 1 vs ori 2, 3)
# -------------------------------------------------------------
# Real impressions
real_imp_disp, real_imp_ang, real_imp_jac, real_imp_mh = [], [], [], []
for ident in range(1, 101):
    fp1 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
    if not fp1: continue
    t1 = tokenizer.extract_relative_tokens(fp1['pts'], fp1['angles'], fp1['types'])
    s1 = tokenizer.compute_signature(t1)
    for ori in [2, 3]:
        fpo = calibrator.load_fingerprint_file(f"q3_{ident}_{ori}.json")
        if not fpo: continue
        to = tokenizer.extract_relative_tokens(fpo['pts'], fpo['angles'], fpo['types'])
        so = tokenizer.compute_signature(to)
        real_imp_jac.append(len(t1 & to) / len(t1 | to) if len(t1 | to) > 0 else 0.0)
        real_imp_mh.append(np.mean(s1 == so))
        res = matcher.match(fp1['pts'], fp1['angles'], fp1['types'], fpo['pts'], fpo['angles'], fpo['types'])
        if len(res.spatial_distances) > 0: real_imp_disp.extend(res.spatial_distances)
        if len(res.angular_differences) > 0: real_imp_ang.extend(res.angular_differences)

# Synthetic impressions
synth_imp_disp, synth_imp_ang, synth_imp_jac, synth_imp_mh = [], [], [], []
for i in range(100):
    # Find ori 1 q3
    mask1 = (batch.identities == i + 1) & (batch.orientations == 1) & (batch.qualities == 3)
    idx1 = np.where(mask1)[0][0]
    p1 = batch.minutiae[idx1, :batch.num_minutiae[idx1], :2]
    a1 = batch.minutiae[idx1, :batch.num_minutiae[idx1], 2]
    ty1 = batch.minutiae[idx1, :batch.num_minutiae[idx1], 3].astype(int)
    t1 = tokenizer.extract_relative_tokens(p1, a1, ty1)
    s1 = tokenizer.compute_signature(t1)
    
    for ori in [2, 3]:
        masko = (batch.identities == i + 1) & (batch.orientations == ori) & (batch.qualities == 3)
        idxo = np.where(masko)[0][0]
        po = batch.minutiae[idxo, :batch.num_minutiae[idxo], :2]
        ao = batch.minutiae[idxo, :batch.num_minutiae[idxo], 2]
        tyo = batch.minutiae[idxo, :batch.num_minutiae[idxo], 3].astype(int)
        to = tokenizer.extract_relative_tokens(po, ao, tyo)
        so = tokenizer.compute_signature(to)
        synth_imp_jac.append(len(t1 & to) / len(t1 | to) if len(t1 | to) > 0 else 0.0)
        synth_imp_mh.append(np.mean(s1 == so))
        res = matcher.match(p1, a1, ty1, po, ao, tyo)
        if len(res.spatial_distances) > 0: synth_imp_disp.extend(res.spatial_distances)
        if len(res.angular_differences) > 0: synth_imp_ang.extend(res.angular_differences)

print("\n[B] IMPRESSION VARIATIONS COMPARISON (q3_1 vs q3_2, q3_3):")
print(f"{'Metric':<28} | {'Real Data':<18} | {'Synthetic Data':<18}")
print("-" * 70)
print(f"{'Minutiae Disp (Mean)':<28} | {np.mean(real_imp_disp):.3f} px            | {np.mean(synth_imp_disp):.3f} px")
print(f"{'Angular Diff (Mean)':<28} | {np.mean(real_imp_ang):.3f} deg          | {np.mean(synth_imp_ang):.3f} deg")
print(f"{'Token Jaccard (Mean)':<28} | {np.mean(real_imp_jac):.4f}             | {np.mean(synth_imp_jac):.4f}")
print(f"{'MinHash Collision (Mean)':<28} | {np.mean(real_imp_mh):.4f}             | {np.mean(synth_imp_mh):.4f}")

# -------------------------------------------------------------
# C. Quality Degradation Variations (q3 vs q7, q5, q1, q9)
# -------------------------------------------------------------
print("\n[C] QUALITY DEGRADATIONS COMPARISON:")
for q in [7, 5, 1, 9]:
    # Real
    r_surv, r_pos, r_ang, r_jac, r_mh = [], [], [], [], []
    for ident in range(1, 101):
        fp3 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
        fq = calibrator.load_fingerprint_file(f"q{q}_{ident}_1.json")
        if not (fp3 and fq): continue
        t3 = tokenizer.extract_relative_tokens(fp3['pts'], fp3['angles'], fp3['types'])
        tq = tokenizer.extract_relative_tokens(fq['pts'], fq['angles'], fq['types'])
        s3 = tokenizer.compute_signature(t3)
        sq = tokenizer.compute_signature(tq)
        r_jac.append(len(t3 & tq) / len(t3 | tq) if len(t3 | tq) > 0 else 0.0)
        r_mh.append(np.mean(s3 == sq))
        res = matcher.match(fp3['pts'], fp3['angles'], fp3['types'], fq['pts'], fq['angles'], fq['types'])
        r_surv.append(res.survival_rate)
        if len(res.spatial_distances) > 0: r_pos.extend(res.spatial_distances)
        if len(res.angular_differences) > 0: r_ang.extend(res.angular_differences)
        
    # Synth
    s_surv, s_pos, s_ang, s_jac, s_mh = [], [], [], [], []
    for i in range(100):
        mask3 = (batch.identities == i + 1) & (batch.orientations == 1) & (batch.qualities == 3)
        idx3 = np.where(mask3)[0][0]
        p3 = batch.minutiae[idx3, :batch.num_minutiae[idx3], :2]
        a3 = batch.minutiae[idx3, :batch.num_minutiae[idx3], 2]
        ty3 = batch.minutiae[idx3, :batch.num_minutiae[idx3], 3].astype(int)
        t3 = tokenizer.extract_relative_tokens(p3, a3, ty3)
        s3 = tokenizer.compute_signature(t3)
        
        maskq = (batch.identities == i + 1) & (batch.orientations == 1) & (batch.qualities == q)
        idxq = np.where(maskq)[0][0]
        pq = batch.minutiae[idxq, :batch.num_minutiae[idxq], :2]
        aq = batch.minutiae[idxq, :batch.num_minutiae[idxq], 2]
        tyq = batch.minutiae[idxq, :batch.num_minutiae[idxq], 3].astype(int)
        tq = tokenizer.extract_relative_tokens(pq, aq, tyq)
        sq = tokenizer.compute_signature(tq)
        s_jac.append(len(t3 & tq) / len(t3 | tq) if len(t3 | tq) > 0 else 0.0)
        s_mh.append(np.mean(s3 == sq))
        res = matcher.match(p3, a3, ty3, pq, aq, tyq)
        s_surv.append(res.survival_rate)
        if len(res.spatial_distances) > 0: s_pos.extend(res.spatial_distances)
        if len(res.angular_differences) > 0: s_ang.extend(res.angular_differences)

    print(f"--- Quality Level q{q} ---")
    print(f"  Survival Rate:        Real = {np.mean(r_surv):.4f} | Synth = {np.mean(s_surv):.4f}")
    print(f"  Pos Jitter (Mean):    Real = {np.mean(r_pos):.3f}px | Synth = {np.mean(s_pos):.3f}px")
    print(f"  Angular Diff (Mean):  Real = {np.mean(r_ang):.3f}°  | Synth = {np.mean(s_ang):.3f}°")
    print(f"  Token Jaccard (Mean): Real = {np.mean(r_jac):.4f} | Synth = {np.mean(s_jac):.4f}")
    print(f"  MinHash Eq Rate:      Real = {np.mean(r_mh):.4f} | Synth = {np.mean(s_mh):.4f}")

# -------------------------------------------------------------
# D. Inter-Identity (Uniqueness Check)
# -------------------------------------------------------------
synth_inter_jacs, synth_inter_mh = [], []
for i in range(1, 41):
    m_a = (batch.identities == i) & (batch.orientations == 1) & (batch.qualities == 3)
    idx_a = np.where(m_a)[0][0]
    p_a = batch.minutiae[idx_a, :batch.num_minutiae[idx_a], :2]
    a_a = batch.minutiae[idx_a, :batch.num_minutiae[idx_a], 2]
    t_a = batch.minutiae[idx_a, :batch.num_minutiae[idx_a], 3].astype(int)
    tok_a = tokenizer.extract_relative_tokens(p_a, a_a, t_a)
    sig_a = tokenizer.compute_signature(tok_a)
    
    for j in range(41, 81):
        m_b = (batch.identities == j) & (batch.orientations == 1) & (batch.qualities == 3)
        idx_b = np.where(m_b)[0][0]
        p_b = batch.minutiae[idx_b, :batch.num_minutiae[idx_b], :2]
        a_b = batch.minutiae[idx_b, :batch.num_minutiae[idx_b], 2]
        t_b = batch.minutiae[idx_b, :batch.num_minutiae[idx_b], 3].astype(int)
        tok_b = tokenizer.extract_relative_tokens(p_b, a_b, t_b)
        sig_b = tokenizer.compute_signature(tok_b)
        
        synth_inter_jacs.append(len(tok_a & tok_b) / len(tok_a | tok_b) if len(tok_a | tok_b) > 0 else 0.0)
        synth_inter_mh.append(np.mean(sig_a == sig_b))

print("\n[D] INTER-IDENTITY UNIQUENESS (DIFFERENT IDENTITIES):")
print(f"{'Metric':<28} | {'Real Data':<18} | {'Synthetic Data':<18}")
print("-" * 70)
print(f"{'Inter-Identity Jaccard (Mean)':<28} | {0.0205:<18.4f} | {np.mean(synth_inter_jacs):<18.4f}")
print(f"{'Inter-Identity Jaccard (Max)':<28}  | {0.0821:<18.4f} | {np.max(synth_inter_jacs):<18.4f}")
print(f"{'Inter-Identity MinHash Eq':<28}     | {0.0212:<18.4f} | {np.mean(synth_inter_mh):<18.4f}")
print(f"{'Inter-Identity MinHash Max':<28}    | {0.1250:<18.4f} | {np.max(synth_inter_mh):<18.4f}")
