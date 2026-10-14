import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import pickle
import numpy as np
from src.generation.pipeline import SyntheticGenerationPipeline
from src.statistics.calibration import DatasetCalibrator
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine
from src.lsh_scale.lsh_index_6m import ScaledLSHIndex

print("=" * 75)
print("ROUND 2: LSH RETRIEVAL PERFORMANCE BENCHMARK (REAL DATA VS SYNTHETIC DATA)")
print("=" * 75)

tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)
calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")

# 1. Prepare Real Dataset (100 identities, 1000 gallery, 100 probes)
print("Loading Real Fingerprints...")
real_gallery_sigs = []
real_gallery_ids = []
real_probe_sigs = []
real_probe_ids = []

for ident in range(1, 101):
    # Gallery: ori 1 and ori 2 across all qualities (10 samples per identity)
    for ori in [1, 2]:
        for q in [3, 7, 5, 1, 9]:
            fp = calibrator.load_fingerprint_file(f"q{q}_{ident}_{ori}.json")
            if fp:
                tok = tokenizer.extract_relative_tokens(fp['pts'], fp['angles'], fp['types'])
                sig = tokenizer.compute_signature(tok)
                real_gallery_sigs.append(sig)
                real_gallery_ids.append(ident)
                
    # Probe: ori 3 with difficult quality (q1 or q9) as challenging real-world queries
    q_probe = 1 if ident % 2 == 1 else 9
    fp_probe = calibrator.load_fingerprint_file(f"q{q_probe}_{ident}_3.json")
    if fp_probe:
        tok_p = tokenizer.extract_relative_tokens(fp_probe['pts'], fp_probe['angles'], fp_probe['types'])
        sig_p = tokenizer.compute_signature(tok_p)
        real_probe_sigs.append(sig_p)
        real_probe_ids.append(ident)

real_gallery_sigs = np.array(real_gallery_sigs, dtype=np.uint32)
real_gallery_ids = np.array(real_gallery_ids, dtype=np.int32)
real_probe_sigs = np.array(real_probe_sigs, dtype=np.uint32)
real_probe_ids = np.array(real_probe_ids, dtype=np.int32)

print(f"Real Data prepared: Gallery = {len(real_gallery_sigs)} samples, Probes = {len(real_probe_sigs)} queries")

# 2. Prepare Synthetic Dataset (100 fresh identities generated with master_seed=999)
print("\nGenerating Fresh Synthetic Fingerprints...")
with open("models/fitted_models.pkl", "rb") as f:
    bundle = pickle.load(f)
pipeline = SyntheticGenerationPipeline(bundle['model_a'], bundle['angle_model'], bundle['quality_model'])

synth_batch = pipeline.generate_batch(start_identity=1, num_identities_in_batch=100, master_seed=999)

synth_gallery_sigs = []
synth_gallery_ids = []
synth_probe_sigs = []
synth_probe_ids = []

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
        synth_probe_sigs.append(sig)
        synth_probe_ids.append(ident)

synth_gallery_sigs = np.array(synth_gallery_sigs, dtype=np.uint32)
synth_gallery_ids = np.array(synth_gallery_ids, dtype=np.int32)
synth_probe_sigs = np.array(synth_probe_sigs, dtype=np.uint32)
synth_probe_ids = np.array(synth_probe_ids, dtype=np.int32)

print(f"Synthetic Data prepared: Gallery = {len(synth_gallery_sigs)} samples, Probes = {len(synth_probe_sigs)} queries")

# 3. Benchmark LSH Configurations on both datasets
lsh_configs = [
    {"name": "Standard Baseline (b=8, r=8)", "b": 8, "r": 8},
    {"name": "High Sensitivity (b=16, r=4)", "b": 16, "r": 4},
    {"name": "Balanced Tuned (b=12, r=5)", "b": 12, "r": 5},
    {"name": "High Precision (b=4, r=16)", "b": 4, "r": 16},
]

print("\n" + "=" * 75)
print(f"{'LSH Configuration':<30} | {'Metric':<18} | {'Real Data':<10} | {'Synth Data':<10}")
print("=" * 75)

for cfg in lsh_configs:
    b, r = cfg["b"], cfg["r"]
    
    # Real LSH Index
    real_index = ScaledLSHIndex(bands=b, rows=r)
    real_index.index_signatures(real_gallery_sigs)
    
    real_hits = 0
    real_candidates_list = []
    real_zero_cand = 0
    for q_idx in range(len(real_probe_sigs)):
        cands = real_index.find_candidates(real_probe_sigs[q_idx])
        cands_set = set(cands)
        real_candidates_list.append(len(cands_set))
        if len(cands_set) == 0:
            real_zero_cand += 1
        true_id = real_probe_ids[q_idx]
        retrieved_ids = {real_gallery_ids[c] for c in cands_set}
        if true_id in retrieved_ids:
            real_hits += 1
            
    real_hit_rate = 100.0 * real_hits / len(real_probe_sigs)
    real_avg_cands = np.mean(real_candidates_list)
    real_zero_rate = 100.0 * real_zero_cand / len(real_probe_sigs)
    
    # Synthetic LSH Index
    synth_index = ScaledLSHIndex(bands=b, rows=r)
    synth_index.index_signatures(synth_gallery_sigs)
    
    synth_hits = 0
    synth_candidates_list = []
    synth_zero_cand = 0
    for q_idx in range(len(synth_probe_sigs)):
        cands = synth_index.find_candidates(synth_probe_sigs[q_idx])
        cands_set = set(cands)
        synth_candidates_list.append(len(cands_set))
        if len(cands_set) == 0:
            synth_zero_cand += 1
        true_id = synth_probe_ids[q_idx]
        retrieved_ids = {synth_gallery_ids[c] for c in cands_set}
        if true_id in retrieved_ids:
            synth_hits += 1
            
    synth_hit_rate = 100.0 * synth_hits / len(synth_probe_sigs)
    synth_avg_cands = np.mean(synth_candidates_list)
    synth_zero_rate = 100.0 * synth_zero_cand / len(synth_probe_sigs)
    
    print(f"{cfg['name']:<30} | {'Hit Rate (Recall)':<18} | {real_hit_rate:>9.1f}% | {synth_hit_rate:>9.1f}%")
    print(f"{'':<30} | {'Avg Candidates':<18} | {real_avg_cands:>9.2f}  | {synth_avg_cands:>9.2f}")
    print(f"{'':<30} | {'Zero Candidate %':<18} | {real_zero_rate:>9.1f}% | {synth_zero_rate:>9.1f}%")
    print("-" * 75)
