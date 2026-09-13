import os
import math
import struct
from pathlib import Path
from statistics import mean, median, stdev
from collections import defaultdict
import random

import matplotlib.pyplot as plt


# -----------------------------
# Parser
# -----------------------------

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


# -----------------------------
# Relative Tokens (Tuned Hyperparameters)
# -----------------------------

def relative_tokens(
        minutiae,
        k=2,
        dist_bin=10,
        dir_bin=20,
        orient_bin=20,
        include_type=True):

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


# -----------------------------
# Similarity Metrics
# -----------------------------

def min_overlap_similarity(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# -----------------------------
# Finger ID
# -----------------------------

def finger_id(path):
    """
    Assumes filenames like
    1_1.ist
    1_2.ist
    1_3.ist
    25_4.ist

    Everything before '_' is treated as
    the fingerprint identity.
    """
    return Path(path).stem.split("_")[0]


# -----------------------------
# Main Experiment
# -----------------------------

def experiment(folder, n_files=100, metric="min_overlap", show_plot=False):
    folder = Path(folder)
    files = sorted(folder.glob("*.ist"))
    files = files[:n_files]

    if len(files) == 0:
        raise RuntimeError(f"No .ist files found in {folder}")

    print(f"--- Folder {folder.name}: Loaded {len(files)} files ---")

    token_sets = {}
    identities = {}

    for f in files:
        minutiae = parse_fmr(f)
        token_sets[f] = relative_tokens(minutiae)
        identities[f] = finger_id(f)

    genuine = []
    impostor = []

    genuine_pairs = []
    impostor_pairs = []

    file_list = list(token_sets.keys())

    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            a = file_list[i]
            b = file_list[j]

            if metric == "min_overlap":
                sim = min_overlap_similarity(token_sets[a], token_sets[b])
            else:
                sim = jaccard(token_sets[a], token_sets[b])

            if identities[a] == identities[b]:
                genuine.append(sim)
                genuine_pairs.append((sim, a.name, b.name))
            else:
                impostor.append(sim)
                impostor_pairs.append((sim, a.name, b.name))

    def report(name, values):
        print("=" * 50)
        print(name)
        print("=" * 50)
        print(f"Pairs   : {len(values)}")
        print(f"Mean    : {mean(values):.4f}")
        print(f"Median  : {median(values):.4f}")
        print(f"Std Dev : {stdev(values):.4f}")
        print(f"Min     : {min(values):.4f}")
        print(f"Max     : {max(values):.4f}")
        print()

    report("GENUINE", genuine)
    report("IMPOSTOR", impostor)

    # Compute d-prime separation index
    mg, mi = mean(genuine), mean(impostor)
    sg, si = stdev(genuine), stdev(impostor)
    denom = math.sqrt(0.5 * (sg**2 + si**2))
    d_prime = (mg - mi) / (denom if denom > 1e-5 else 1e-5)
    print(f"Separation Index (d'): {d_prime:.3f}\n")

    if show_plot:
        plt.figure(figsize=(12, 2.8))
        plt.scatter(
            genuine,
            [1 + random.uniform(-0.05, 0.05) for _ in genuine],
            s=12,
            alpha=0.6,
            label="Genuine"
        )
        plt.scatter(
            impostor,
            [0 + random.uniform(-0.05, 0.05) for _ in impostor],
            s=12,
            alpha=0.35,
            label="Impostor"
        )
        plt.yticks([0, 1], ["Impostor", "Genuine"])
        plt.xlabel(f"Similarity ({metric})")
        plt.xlim(0, 1)
        plt.grid(axis="x", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"d_prime": d_prime, "gen_mean": mg, "imp_mean": mi}


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    base_dir = Path(r"C:\Users\hrish\Projects\research_projects\duplicate_detection\fingerprint_dataset")
    target_folders = ["1", "3", "5", "7", "9"]
    
    print("==================================================")
    print("EVALUATING TUNED PARSER ACROSS QUALITY FOLDERS (1, 3, 5, 7, 9)")
    print("==================================================\n")

    for q in target_folders:
        folder_path = base_dir / q
        experiment(folder=folder_path, n_files=100, metric="min_overlap", show_plot=False)