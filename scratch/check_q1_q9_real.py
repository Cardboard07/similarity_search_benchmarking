import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from src.statistics.calibration import DatasetCalibrator
from src.lsh_scale.vectorized_minhash import VectorizedMinHashEngine
from src.lsh_scale.lsh_index_6m import ScaledLSHIndex
import numpy as np

calibrator = DatasetCalibrator("fingerprint_dataset/parsed_data")
tokenizer = VectorizedMinHashEngine(num_hashes=64, seed=42)

for q_probe in [7, 5, 1, 9]:
    gallery_sigs, gallery_ids = [], []
    probe_sigs, probe_ids = [], []
    for ident in range(1, 101):
        for ori in [1, 2]:
            for q in [3, 7, 5, 1, 9]:
                fp = calibrator.load_fingerprint_file(f"q{q}_{ident}_{ori}.json")
                if fp:
                    tok = tokenizer.extract_relative_tokens(fp['pts'], fp['angles'], fp['types'])
                    gallery_sigs.append(tokenizer.compute_signature(tok))
                    gallery_ids.append(ident)
        fp_p = calibrator.load_fingerprint_file(f"q{q_probe}_{ident}_3.json")
        if fp_p:
            tok = tokenizer.extract_relative_tokens(fp_p['pts'], fp_p['angles'], fp_p['types'])
            probe_sigs.append(tokenizer.compute_signature(tok))
            probe_ids.append(ident)
            
    idx = ScaledLSHIndex(bands=8, rows=8)
    idx.index_signatures(np.array(gallery_sigs, dtype=np.uint32))
    hits = 0
    cands_list = []
    for i in range(len(probe_sigs)):
        c = set(idx.find_candidates(probe_sigs[i]))
        cands_list.append(len(c))
        ret = {gallery_ids[j] for j in c}
        if probe_ids[i] in ret:
            hits += 1
    print(f"Real Probe q{q_probe}: Hit Rate = {100.0 * hits / len(probe_sigs):.1f}%, Avg Cands = {np.mean(cands_list):.2f}")
