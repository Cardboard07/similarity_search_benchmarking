import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import json
import numpy as np
from src.statistics.calibration import DatasetCalibrator
from src.models.correspondence import MinutiaMatcher
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)
matcher = MinutiaMatcher(spatial_gate=18.0, angular_gate_deg=35.0)

print("Calibrator indexed files:", len(calibrator.file_map))

# 1. Measure pristine impressions across 200 identities
trans_x, trans_y, rot_degs, scale_vals = [], [], [], []
pos_disps, ang_disps = [], []
jaccards_ori = []
minhash_matches_ori = []

for ident in range(1, 201):
    fp1 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
    if not fp1: continue
    tok1 = tokenizer.extract_relative_tokens(fp1['pts'], fp1['angles'], fp1['types'])
    sig1 = tokenizer.compute_signature(tok1)
    
    for ori in [2, 3]:
        fpo = calibrator.load_fingerprint_file(f"q3_{ident}_{ori}.json")
        if not fpo: continue
        toko = tokenizer.extract_relative_tokens(fpo['pts'], fpo['angles'], fpo['types'])
        sigo = tokenizer.compute_signature(toko)
        
        # Jaccard
        inter = len(tok1 & toko)
        union = len(tok1 | toko)
        jac = inter / union if union > 0 else 0.0
        jaccards_ori.append(jac)
        minhash_matches_ori.append(np.mean(sig1 == sigo))
        
        res = matcher.match(
            pts_a=fp1['pts'], angles_a=fp1['angles'], types_a=fp1['types'],
            pts_b=fpo['pts'], angles_b=fpo['angles'], types_b=fpo['types']
        )
        trans_x.append(res.estimated_tx)
        trans_y.append(res.estimated_ty)
        rot_degs.append(res.estimated_rotation_deg)
        scale_vals.append(res.estimated_scale)
        if len(res.spatial_distances) > 0:
            pos_disps.extend(res.spatial_distances)
        if len(res.angular_differences) > 0:
            ang_disps.extend(res.angular_differences)

print("=" * 60)
print("REAL PRISTINE IMPRESSIONS (q3_1 vs q3_2, q3_3):")
print(f"tx: mean={np.mean(trans_x):.4f}, std={np.std(trans_x):.4f}")
print(f"ty: mean={np.mean(trans_y):.4f}, std={np.std(trans_y):.4f}")
print(f"rot: mean={np.mean(rot_degs):.4f} deg, std={np.std(rot_degs):.4f} deg, max={np.max(np.abs(rot_degs)):.4f}")
print(f"scale: mean={np.mean(scale_vals):.5f}, std={np.std(scale_vals):.5f}")
print(f"Minutiae Pos Disp: mean={np.mean(pos_disps):.3f}px, median={np.median(pos_disps):.3f}px, 95th={np.percentile(pos_disps, 95):.3f}px")
print(f"Minutiae Ang Disp: mean={np.mean(ang_disps):.3f} deg, median={np.median(ang_disps):.3f} deg, 95th={np.percentile(ang_disps, 95):.3f} deg, max={np.max(ang_disps):.3f} deg")
print(f"Token Jaccard: mean={np.mean(jaccards_ori):.4f}, median={np.median(jaccards_ori):.4f}, min={np.min(jaccards_ori):.4f}, max={np.max(jaccards_ori):.4f}")
print(f"MinHash Eq Rate: mean={np.mean(minhash_matches_ori):.4f}")

# 2. Measure quality degradations across 200 identities
print("=" * 60)
print("REAL QUALITY DEGRADATIONS (q3 vs q7, q5, q1, q9):")
for q in [7, 5, 1, 9]:
    survs, falses, q_pos, q_ang, q_jacs, q_mh = [], [], [], [], [], []
    for ident in range(1, 201):
        fp3 = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
        fq = calibrator.load_fingerprint_file(f"q{q}_{ident}_1.json")
        if not (fp3 and fq): continue
        tok3 = tokenizer.extract_relative_tokens(fp3['pts'], fp3['angles'], fp3['types'])
        tokq = tokenizer.extract_relative_tokens(fq['pts'], fq['angles'], fq['types'])
        sig3 = tokenizer.compute_signature(tok3)
        sigq = tokenizer.compute_signature(tokq)
        
        inter = len(tok3 & tokq)
        union = len(tok3 | tokq)
        q_jacs.append(inter / union if union > 0 else 0.0)
        q_mh.append(np.mean(sig3 == sigq))
        
        res = matcher.match(
            pts_a=fp3['pts'], angles_a=fp3['angles'], types_a=fp3['types'],
            pts_b=fq['pts'], angles_b=fq['angles'], types_b=fq['types']
        )
        survs.append(res.survival_rate)
        falses.append(res.false_minutiae_count)
        if len(res.spatial_distances) > 0:
            q_pos.extend(res.spatial_distances)
        if len(res.angular_differences) > 0:
            q_ang.extend(res.angular_differences)
            
    print(f"Quality q{q}:")
    print(f"  Survival rate: mean={np.mean(survs):.4f}, min={np.min(survs):.4f}")
    print(f"  False minutiae count: mean={np.mean(falses):.3f}, max={np.max(falses)}")
    print(f"  Pos Disp: mean={np.mean(q_pos):.3f}px, median={np.median(q_pos):.3f}px, 95th={np.percentile(q_pos, 95):.3f}px")
    print(f"  Ang Disp: mean={np.mean(q_ang):.3f} deg, median={np.median(q_ang):.3f} deg, 95th={np.percentile(q_ang, 95):.3f} deg, max={np.max(q_ang):.3f} deg")
    print(f"  Token Jaccard: mean={np.mean(q_jacs):.4f}, median={np.median(q_jacs):.4f}")
    print(f"  MinHash Eq Rate: mean={np.mean(q_mh):.4f}")

# 3. Measure cross-variation (worst case: ori 3 with q1/q9 vs ori 1 q3)
print("=" * 60)
print("REAL CROSS VARIATION (ori 1 q3 vs ori 3 q1/q9):")
cross_jacs, cross_mh = [], []
for ident in range(1, 201):
    fp_ref = calibrator.load_fingerprint_file(f"q3_{ident}_1.json")
    for q in [1, 9]:
        fp_worst = calibrator.load_fingerprint_file(f"q{q}_{ident}_3.json")
        if not (fp_ref and fp_worst): continue
        tok_ref = tokenizer.extract_relative_tokens(fp_ref['pts'], fp_ref['angles'], fp_ref['types'])
        tok_worst = tokenizer.extract_relative_tokens(fp_worst['pts'], fp_worst['angles'], fp_worst['types'])
        sig_ref = tokenizer.compute_signature(tok_ref)
        sig_worst = tokenizer.compute_signature(tok_worst)
        
        inter = len(tok_ref & tok_worst)
        union = len(tok_ref | tok_worst)
        cross_jacs.append(inter / union if union > 0 else 0.0)
        cross_mh.append(np.mean(sig_ref == sig_worst))
print(f"Worst Case Cross-variation (ori 3 q1/9 vs ref):")
print(f"  Token Jaccard: mean={np.mean(cross_jacs):.4f}, median={np.median(cross_jacs):.4f}, min={np.min(cross_jacs):.4f}")
print(f"  MinHash Eq Rate: mean={np.mean(cross_mh):.4f}")

# 4. Measure Inter-identity (different people)
print("=" * 60)
print("REAL INTER-IDENTITY (DIFFERENT IDENTITIES):")
inter_jacs, inter_mh = [], []
for i in range(1, 51):
    fp_a = calibrator.load_fingerprint_file(f"q3_{i}_1.json")
    for j in range(51, 101):
        fp_b = calibrator.load_fingerprint_file(f"q3_{j}_1.json")
        if not (fp_a and fp_b): continue
        tok_a = tokenizer.extract_relative_tokens(fp_a['pts'], fp_a['angles'], fp_a['types'])
        tok_b = tokenizer.extract_relative_tokens(fp_b['pts'], fp_b['angles'], fp_b['types'])
        sig_a = tokenizer.compute_signature(tok_a)
        sig_b = tokenizer.compute_signature(tok_b)
        inter = len(tok_a & tok_b)
        union = len(tok_a | tok_b)
        inter_jacs.append(inter / union if union > 0 else 0.0)
        inter_mh.append(np.mean(sig_a == sig_b))
print(f"Inter-Identity Jaccard: mean={np.mean(inter_jacs):.4f}, 99th={np.percentile(inter_jacs, 99):.4f}, max={np.max(inter_jacs):.4f}")
print(f"Inter-Identity MinHash Eq Rate: mean={np.mean(inter_mh):.4f}, max={np.max(inter_mh):.4f}")
