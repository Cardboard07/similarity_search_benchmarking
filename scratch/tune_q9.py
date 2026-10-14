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

# Set exact empirical parameters for q9 and q1
# In real data, q9 is captured with pos_err 1.0 px and ang_err 0.8 deg
qm.params[9].pos_noise_std = 0.35
qm.params[9].angle_noise_kappa = 2800.0
qm.params[9].survival_rate = 0.985

# q1 has pos_err 1.37 px, ang_err 1.24 deg
qm.params[1].pos_noise_std = 0.50
qm.params[1].angle_noise_kappa = 1800.0
qm.params[1].survival_rate = 0.985

# q7 has pos_err 1.67 px, ang_err 1.87 deg
qm.params[7].pos_noise_std = 0.65
qm.params[7].angle_noise_kappa = 1200.0

# q5 has pos_err 1.63 px, ang_err 1.83 deg
qm.params[5].pos_noise_std = 0.65
qm.params[5].angle_noise_kappa = 1100.0

pipeline = SyntheticGenerationPipeline(bundle['model_a'], bundle['angle_model'], qm)
synth_batch = pipeline.generate_batch(start_identity=1, num_identities_in_batch=100, master_seed=999)

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
        # Take 100 probes matching real (alternating q1/q9 like real test)
        if (ident % 2 == 1 and q == 1) or (ident % 2 == 0 and q == 9):
            synth_probe_sigs.append(sig)
            synth_probe_ids.append(ident)

synth_gallery_sigs = np.array(synth_gallery_sigs, dtype=np.uint32)
synth_probe_sigs = np.array(synth_probe_sigs, dtype=np.uint32)

print(f"Synthetic Gallery: {len(synth_gallery_sigs)}, Probes: {len(synth_probe_sigs)}")

for b, r, name in [(8, 8, "Baseline (b=8, r=8)"), (16, 4, "High Sensitivity (b=16, r=4)"), (12, 5, "Balanced (b=12, r=5)")]:
    idx = ScaledLSHIndex(bands=b, rows=r)
    idx.index_signatures(synth_gallery_sigs)
    hits = 0
    cands_lens = []
    for q_idx in range(len(synth_probe_sigs)):
        c = set(idx.find_candidates(synth_probe_sigs[q_idx]))
        cands_lens.append(len(c))
        ret_ids = {synth_gallery_ids[i] for i in c}
        if synth_probe_ids[q_idx] in ret_ids:
            hits += 1
    print(f"{name}: Hit Rate = {100.0 * hits / len(synth_probe_sigs):.1f}%, Avg Cands = {np.mean(cands_lens):.2f}")
