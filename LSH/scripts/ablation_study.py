import os
import math
import struct
from pathlib import Path
from statistics import mean, stdev
import numpy as np
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

def extract_tokens(minutiae, k=3, dist_bin=15, dir_bin=30, orient_bin=30, include_type=False):
    tokens = set()
    n = len(minutiae)
    for i in range(n):
        xi = minutiae[i]["x"]
        yi = minutiae[i]["y"]
        ai = minutiae[i]["angle"]
        ti = minutiae[i]["type"]

        dists = []
        for j in range(n):
            if i == j:
                continue
            xj = minutiae[j]["x"]
            yj = minutiae[j]["y"]
            d = math.hypot(xj - xi, yj - yi)
            dists.append((d, j))

        dists.sort(key=lambda x: x[0])

        for d, j in dists[:k]:
            xj = minutiae[j]["x"]
            yj = minutiae[j]["y"]
            aj = minutiae[j]["angle"]
            tj = minutiae[j]["type"]

            abs_dir = math.degrees(math.atan2(yj - yi, xj - xi)) % 360
            rel_dir = (abs_dir - ai) % 360
            delta = (aj - ai) % 360

            if include_type:
                token = (
                    int(d // dist_bin),
                    int(rel_dir // dir_bin),
                    int(delta // orient_bin),
                    ti,
                    tj
                )
            else:
                token = (
                    int(d // dist_bin),
                    int(rel_dir // dir_bin),
                    int(delta // orient_bin),
                )
            tokens.add(token)
    return tokens

def compute_similarity(sa, sb, metric="jaccard"):
    la, lb = len(sa), len(sb)
    if not la and not lb:
        return 1.0
    if not la or not lb:
        return 0.0
    inter = len(sa & sb)
    if metric == "jaccard":
        return inter / len(sa | sb)
    elif metric == "min_overlap":
        return inter / min(la, lb)
    return inter / len(sa | sb)

def compute_metrics(genuine, impostor):
    g = np.sort(np.array(genuine, dtype=np.float32))
    imp = np.sort(np.array(impostor, dtype=np.float32))
    n_g, n_i = len(g), len(imp)
    
    if n_g == 0 or n_i == 0:
        return 0.5, 0.0
        
    thresholds = np.unique(np.concatenate([g, imp]))
    fnmr = np.searchsorted(g, thresholds, side='left') / float(n_g)
    fmr = 1.0 - (np.searchsorted(imp, thresholds, side='left') / float(n_i))
    diff = np.abs(fnmr - fmr)
    idx = np.argmin(diff)
    best_eer = float((fnmr[idx] + fmr[idx]) / 2.0)
    
    mg, mi = float(np.mean(g)), float(np.mean(imp))
    sg = float(np.std(g, ddof=1)) if n_g > 1 else 1e-5
    si = float(np.std(imp, ddof=1)) if n_i > 1 else 1e-5
    denom = math.sqrt(0.5 * (sg**2 + si**2))
    dp = (mg - mi) / (denom if denom > 1e-5 else 1e-5)
    return best_eer, dp, mg, mi

def evaluate_ablation_config(target_folders, files_by_folder, params, metric="jaccard"):
    eers = []
    dprimes = []
    gen_means = []
    imp_means = []
    
    for q in target_folders:
        files = files_by_folder[q]
        token_sets = {}
        identities = {}
        for f in files:
            minutiae = parse_fmr(f)
            token_sets[f] = extract_tokens(minutiae, **params)
            identities[f] = finger_id(f)
            
        genuine = []
        impostor = []
        file_list = list(token_sets.keys())
        
        for i in range(len(file_list)):
            a = file_list[i]
            sa = token_sets[a]
            for j in range(i + 1, len(file_list)):
                b = file_list[j]
                sb = token_sets[b]
                sim = compute_similarity(sa, sb, metric=metric)
                if identities[a] == identities[b]:
                    genuine.append(sim)
                else:
                    impostor.append(sim)
                    
        eer, dp, mg, mi = compute_metrics(genuine, impostor)
        eers.append(eer)
        dprimes.append(dp)
        gen_means.append(mg)
        imp_means.append(mi)
        
    return {
        "mean_eer": mean(eers),
        "mean_dprime": mean(dprimes),
        "gen_mean": mean(gen_means),
        "imp_mean": mean(imp_means)
    }

def main():
    dataset_dir = Path(r"C:\Users\hrish\Projects\research_projects\duplicate_detection\fingerprint_dataset")
    target_folders = ["1", "3", "5", "7", "9"]
    
    # Define Split: HELD-OUT TEST SPLIT (identities > 500, e.g. 501-700)
    print("Loading HELD-OUT TEST SPLIT (Identities 501 to 650) across folders 1, 3, 5, 7, 9...")
    files_by_folder = {}
    for q in target_folders:
        folder = dataset_dir / q
        all_files = sorted(folder.glob("*.ist"))
        test_files = [f for f in all_files if int(finger_id(f)) > 500][:150]
        files_by_folder[q] = test_files
        print(f"Folder {q}: {len(test_files)} test files loaded.")
        
    # Baseline Parameters
    base_params = {"k": 3, "dist_bin": 15, "dir_bin": 30, "orient_bin": 30, "include_type": False}
    base_metric = "jaccard"
    
    # Ablations
    ablations = [
        ("Baseline", base_params, base_metric),
        ("Ablation 1: k=2 (vs k=3)", {**base_params, "k": 2}, base_metric),
        ("Ablation 2: dist_bin=10 (vs 15)", {**base_params, "dist_bin": 10}, base_metric),
        ("Ablation 3: dir_bin=20 & orient_bin=20 (vs 30)", {**base_params, "dir_bin": 20, "orient_bin": 20}, base_metric),
        ("Ablation 4: include_type=True (vs False)", {**base_params, "include_type": True}, base_metric),
        ("Ablation 5: metric=min_overlap (vs Jaccard)", base_params, "min_overlap"),
        ("FULL TUNED COMBINATION", {"k": 2, "dist_bin": 10, "dir_bin": 20, "orient_bin": 20, "include_type": True}, "min_overlap")
    ]
    
    print("\n==================================================================================")
    print("RUNNING ABLATION STUDY ON HELD-OUT TEST SPLIT")
    print("==================================================================================")
    
    results = []
    for name, params, metric in ablations:
        res = evaluate_ablation_config(target_folders, files_by_folder, params, metric=metric)
        results.append((name, res))
        print(f"\n[{name}]")
        print(f"  Mean d'   : {res['mean_dprime']:.3f}")
        print(f"  Mean EER  : {res['mean_eer']*100:.2f}%")
        print(f"  Gen Mean  : {res['gen_mean']:.4f}")
        print(f"  Imp Mean  : {res['imp_mean']:.4f}")
        
    print("\n==================================================================================")
    print("SUMMARY ABLATION TABLE")
    print("==================================================================================")
    print(f"{'Configuration':<45} | {'d (d-prime)':<10} | {'EER (%)':<8} | {'Gen Mean':<9} | {'Imp Mean':<9}")
    print("-" * 90)
    for name, res in results:
        print(f"{name:<45} | {res['mean_dprime']:<10.3f} | {res['mean_eer']*100:<8.2f} | {res['gen_mean']:<9.4f} | {res['imp_mean']:<9.4f}")

if __name__ == "__main__":
    main()
