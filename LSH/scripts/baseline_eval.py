import os
import math
import struct
from pathlib import Path
from statistics import mean, median, stdev
import time

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

def relative_tokens(minutiae, k=3, dist_bin=15, dir_bin=30, orient_bin=30, include_type=False, min_quality=0, max_dist=None):
    if min_quality > 0:
        minutiae = [m for m in minutiae if m["quality"] >= min_quality]

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
            if max_dist is not None and d > max_dist:
                continue
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

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def finger_id(path):
    return Path(path).stem.split("_")[0]

def compute_eer(genuine, impostor):
    gen_sorted = sorted(genuine)
    imp_sorted = sorted(impostor)
    
    all_scores = sorted(list(set(genuine + impostor)))
    if not all_scores:
        return 0.5, 0.0
    
    n_gen = len(genuine)
    n_imp = len(impostor)
    
    best_eer = 1.0
    best_thresh = 0.0
    
    for t in all_scores:
        fnmr = sum(1 for s in genuine if s < t) / n_gen
        fmr = sum(1 for s in impostor if s >= t) / n_imp
        
        eer = (fnmr + fmr) / 2.0
        if abs(fnmr - fmr) < abs(best_eer - 0.0):
            best_eer = max(fnmr, fmr)
            best_thresh = t
            
    return best_eer, best_thresh

def compute_d_prime(genuine, impostor):
    m_g, s_g = mean(genuine), stdev(genuine) if len(genuine) > 1 else 1e-5
    m_i, s_i = mean(impostor), stdev(impostor) if len(impostor) > 1 else 1e-5
    denom = math.sqrt(0.5 * (s_g**2 + s_i**2))
    return (m_g - m_i) / (denom if denom > 1e-5 else 1e-5)

def evaluate_folder(folder_path, n_files=150, params=None):
    if params is None:
        params = {}
    folder = Path(folder_path)
    files = sorted(folder.glob("*.ist"))[:n_files]
    
    token_sets = {}
    identities = {}
    for f in files:
        minutiae = parse_fmr(f)
        token_sets[f] = relative_tokens(minutiae, **params)
        identities[f] = finger_id(f)
        
    genuine = []
    impostor = []
    file_list = list(token_sets.keys())
    
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            a = file_list[i]
            b = file_list[j]
            sim = jaccard(token_sets[a], token_sets[b])
            if identities[a] == identities[b]:
                genuine.append(sim)
            else:
                impostor.append(sim)
                
    eer, thresh = compute_eer(genuine, impostor)
    d_prime = compute_d_prime(genuine, impostor)
    
    return {
        "n_files": len(files),
        "n_gen": len(genuine),
        "n_imp": len(impostor),
        "gen_mean": mean(genuine),
        "imp_mean": mean(impostor),
        "gen_std": stdev(genuine) if len(genuine) > 1 else 0.0,
        "imp_std": stdev(impostor) if len(impostor) > 1 else 0.0,
        "eer": eer,
        "d_prime": d_prime,
    }

if __name__ == "__main__":
    dataset_dir = Path(r"C:\Users\hrish\Projects\research_projects\duplicate_detection\fingerprint_dataset")
    folders = ["1", "3", "5", "7", "9"]
    
    print("=== BASELINE EVALUATION (Default Parameters: k=3, dist_bin=15, dir_bin=30, orient_bin=30) ===")
    overall_eers = []
    overall_dprimes = []
    
    for q in folders:
        fpath = dataset_dir / q
        res = evaluate_folder(fpath, n_files=150)
        print(f"Folder {q}: EER={res['eer']*100:.2f}%, d'={res['d_prime']:.3f}, GenMean={res['gen_mean']:.4f}, ImpMean={res['imp_mean']:.4f}")
        overall_eers.append(res['eer'])
        overall_dprimes.append(res['d_prime'])
        
    print(f"\nOverall Mean EER: {mean(overall_eers)*100:.2f}% | Overall Mean d': {mean(overall_dprimes):.3f}")
