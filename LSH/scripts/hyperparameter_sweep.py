import os
import math
import struct
from pathlib import Path
from statistics import mean, stdev
import time
import itertools
import json

def parse_fmr(path):
    with open(path, "rb") as f:
        data = f.read()

    num_views = data[22]
    offset = 24
    views = []

    for _ in range(num_views):
        num_minutiae = data[offset + 3]
        offset += 4
        minutiae = []
        for _ in range(num_minutiae):
            chunk = data[offset:offset + 6]
            x_raw, y_raw, angle, quality = struct.unpack(">HHBB", chunk)
            mtype = (x_raw >> 14) & 0x03
            x = x_raw & 0x3FFF
            y = y_raw & 0x3FFF
            angle = angle * (360.0 / 256.0)

            minutiae.append({
                "x": x,
                "y": y,
                "angle": angle,
                "type": mtype,
                "quality": quality
            })
            offset += 6
        views.append(minutiae)
    return views[0]

def finger_id(path):
    return Path(path).stem.split("_")[0]

def precompute_dataset(dataset_dir, target_folders, n_files_per_folder=50):
    cache = {}
    identities = {}
    for q in target_folders:
        folder = dataset_dir / q
        files = sorted(folder.glob("*.ist"))[:n_files_per_folder]
        cache[q] = {}
        identities[q] = {}
        for f in files:
            minutiae = parse_fmr(f)
            n = len(minutiae)
            all_i_neighbors = []
            for i in range(n):
                xi, yi, ai, ti = minutiae[i]["x"], minutiae[i]["y"], minutiae[i]["angle"], minutiae[i]["type"]
                dists = []
                for j in range(n):
                    if i == j:
                        continue
                    xj, yj, aj, tj = minutiae[j]["x"], minutiae[j]["y"], minutiae[j]["angle"], minutiae[j]["type"]
                    d = math.hypot(xj - xi, yj - yi)
                    abs_dir = math.degrees(math.atan2(yj - yi, xj - xi)) % 360
                    rel_dir = (abs_dir - ai) % 360
                    delta = (aj - ai) % 360
                    dists.append((d, rel_dir, delta, ti, tj))
                dists.sort(key=lambda x: x[0])
                all_i_neighbors.append(dists)
                
            neighbors_by_k = {}
            for k in [2, 3, 4, 5, 6, 8]:
                k_list = []
                for dists in all_i_neighbors:
                    k_list.extend(dists[:k])
                neighbors_by_k[k] = k_list
            cache[q][f] = neighbors_by_k
            identities[q][f] = finger_id(f)
            
    return cache, identities

def main():
    dataset_dir = Path(r"C:\Users\hrish\Projects\research_projects\duplicate_detection\fingerprint_dataset")
    target_folders = ["1", "3", "5", "7", "9"]
    n_files = 50
    
    print(f"Loading and precomputing dataset for folders {target_folders} ({n_files} files per folder)...", flush=True)
    cache, identities = precompute_dataset(dataset_dir, target_folders, n_files)
    print("Dataset precomputed successfully!", flush=True)
    
    # Precompute pairwise indices
    folder_pairs = {}
    for q in target_folders:
        file_list = list(cache[q].keys())
        ident = identities[q]
        gen_pairs = []
        imp_pairs = []
        for i in range(len(file_list)):
            fa = file_list[i]
            ia = ident[fa]
            for j in range(i + 1, len(file_list)):
                fb = file_list[j]
                ib = ident[fb]
                if ia == ib:
                    gen_pairs.append((fa, fb))
                else:
                    imp_pairs.append((fa, fb))
        folder_pairs[q] = (file_list, gen_pairs, imp_pairs)
        
    ks = [2, 3, 4, 5, 6]
    dist_bins = [10, 15, 20, 30]
    dir_bins = [20, 30, 45]
    orient_bins = [20, 30, 45]
    include_types = [False, True]
    metrics = ["jaccard", "min_overlap"]
    
    grid = list(itertools.product(ks, dist_bins, dir_bins, orient_bins, include_types, metrics))
    print(f"Sweeping {len(grid)} hyperparameter combinations...", flush=True)
    
    start_time = time.time()
    results = []
    
    for idx, (k, dist_bin, dir_bin, orient_bin, include_type, metric) in enumerate(grid):
        eers = []
        dprimes = []
        
        for q in target_folders:
            file_list, gen_pairs, imp_pairs = folder_pairs[q]
            
            token_sets = {}
            for f in file_list:
                rel_list = cache[q][f][k]
                tokens = set()
                for d, rel_dir, delta, ti, tj in rel_list:
                    db = int(d // dist_bin)
                    rb = int(rel_dir // dir_bin)
                    ob = int(delta // orient_bin)
                    if include_type:
                        tokens.add((db, rb, ob, ti, tj))
                    else:
                        tokens.add((db, rb, ob))
                token_sets[f] = tokens
                
            genuine = []
            for fa, fb in gen_pairs:
                sa = token_sets[fa]
                sb = token_sets[fb]
                la, lb = len(sa), len(sb)
                if not la and not lb:
                    sim = 1.0
                elif not la or not lb:
                    sim = 0.0
                else:
                    inter = len(sa & sb)
                    if metric == "jaccard":
                        sim = inter / len(sa | sb)
                    elif metric == "min_overlap":
                        sim = inter / min(la, lb)
                genuine.append(sim)
                
            impostor = []
            for fa, fb in imp_pairs:
                sa = token_sets[fa]
                sb = token_sets[fb]
                la, lb = len(sa), len(sb)
                if not la and not lb:
                    sim = 1.0
                elif not la or not lb:
                    sim = 0.0
                else:
                    inter = len(sa & sb)
                    if metric == "jaccard":
                        sim = inter / len(sa | sb)
                    elif metric == "min_overlap":
                        sim = inter / min(la, lb)
                impostor.append(sim)
                
            ng, ni = len(genuine), len(impostor)
            mg = mean(genuine) if ng > 0 else 0.0
            mi = mean(impostor) if ni > 0 else 0.0
            sg = stdev(genuine) if ng > 1 else 1e-5
            si = stdev(impostor) if ni > 1 else 1e-5
            denom = math.sqrt(0.5 * (sg**2 + si**2))
            dp = (mg - mi) / (denom if denom > 1e-5 else 1e-5)
            
            all_s = sorted(list(set(genuine + impostor)))
            step = max(1, len(all_s) // 30)
            threshold_samples = all_s[::step]
            b_eer = 1.0
            for t in threshold_samples:
                fnmr = sum(1 for s in genuine if s < t) / ng
                fmr = sum(1 for s in impostor if s >= t) / ni
                eer = (fnmr + fmr) / 2.0
                if abs(fnmr - fmr) < abs(b_eer - 0.0):
                    b_eer = max(fnmr, fmr)
                    
            eers.append(b_eer)
            dprimes.append(dp)
            
        me = mean(eers)
        md = mean(dprimes)
        score = md - 10.0 * me
        
        results.append({
            "config": {
                "k": k,
                "dist_bin": dist_bin,
                "dir_bin": dir_bin,
                "orient_bin": orient_bin,
                "include_type": include_type,
                "metric": metric
            },
            "mean_eer": me,
            "mean_dprime": md,
            "score": score,
            "folder_eers": eers,
            "folder_dprimes": dprimes
        })
        
        if (idx + 1) % 50 == 0 or idx == len(grid) - 1:
            print(f"Evaluated {idx + 1}/{len(grid)} configs in {time.time() - start_time:.1f}s...", flush=True)
            
    results.sort(key=lambda r: r["score"], reverse=True)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted sweep in {elapsed:.2f} seconds!", flush=True)
    print("=" * 70, flush=True)
    print("TOP 10 BEST HYPERPARAMETER CONFIGURATIONS ACROSS FOLDERS 1,3,5,7,9:", flush=True)
    print("=" * 70, flush=True)
    
    for rank, res in enumerate(results[:10], 1):
        cfg = res["config"]
        print(f"#{rank:02d} | Score: {res['score']:.4f} | EER: {res['mean_eer']*100:.2f}% | d': {res['mean_dprime']:.3f} | Metric: {cfg['metric']}", flush=True)
        print(f"     Params -> k: {cfg['k']}, dist_bin: {cfg['dist_bin']}, dir_bin: {cfg['dir_bin']}, orient_bin: {cfg['orient_bin']}, include_type: {cfg['include_type']}", flush=True)
        print(f"     Folder EERs [1,3,5,7,9]: {[round(e*100, 2) for e in res['folder_eers']]}", flush=True)
        print(f"     Folder d'   [1,3,5,7,9]: {[round(d, 2) for d in res['folder_dprimes']]}", flush=True)
        print()
        
    best = results[0]
    print("=" * 70, flush=True)
    print("WINNING CONFIGURATION:", flush=True)
    print(json.dumps(best, indent=2), flush=True)
    print("=" * 70, flush=True)
    
    out_file = Path(r"C:\Users\hrish\.gemini\antigravity-ide\brain\630e4a86-35b6-4e42-99f5-ef75ec3a8a24\sweep_results.json")
    with open(out_file, "w") as f:
        json.dump(results[:20], f, indent=2)
    print(f"Top 20 sweep results saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
