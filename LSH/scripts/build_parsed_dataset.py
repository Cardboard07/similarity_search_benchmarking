import os
import math
import struct
from pathlib import Path
import json
import shutil
import random

def parse_fmr(path):
    with open(path, "rb") as f:
        data = f.read()

    size_x = struct.unpack(">H", data[14:16])[0]
    size_y = struct.unpack(">H", data[16:18])[0]
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
            angle_deg = angle * (360.0 / 256.0)

            minutiae.append({
                "x": x,
                "y": y,
                "angle": angle_deg,
                "type": mtype,
                "quality": quality
            })
            offset += 6
        views.append(minutiae)
    return size_x, size_y, views[0]

def extract_relative_tokens(minutiae, k=2, dist_bin=10, dir_bin=20, orient_bin=20, include_type=True):
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

    return sorted(list(tokens))

def parse_and_save_dataset():
    dataset_dir = Path(r"C:\Users\hrish\Projects\research_projects\duplicate_detection\fingerprint_dataset")
    target_folders = ["1", "3", "5", "7", "9"]
    
    output_dir = dataset_dir / "parsed_data"
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    
    # Clean existing directories if present
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
        
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Group files by fingerprint identity (e.g. "1", "2", ... "1000")
    identity_files = {}
    for q in target_folders:
        folder = dataset_dir / q
        ist_files = sorted(folder.glob("*.ist"))
        for f in ist_files:
            stem = f.stem  # e.g., "100_1"
            parts = stem.split("_")
            ident_str = parts[0]
            impression_str = parts[1] if len(parts) > 1 else "1"
            
            if ident_str not in identity_files:
                identity_files[ident_str] = []
            identity_files[ident_str].append((q, impression_str, f))
            
    print(f"Total unique identities found: {len(identity_files)}")
    
    # Split strategy per identity:
    # For every identity (1..1000), split its 15 (quality, impression) files into train (approx 70%) and test (approx 30%)
    train_count = 0
    test_count = 0
    
    for ident_str, file_records in sorted(identity_files.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        # Deterministic shuffle per identity so split is balanced & reproducible across orientations/qualities
        seed_val = int(ident_str) if ident_str.isdigit() else hash(ident_str)
        rng = random.Random(seed_val)
        
        # Sort records by (quality, impression) then shuffle with seed
        sorted_records = sorted(file_records, key=lambda r: (int(r[0]), int(r[1]) if r[1].isdigit() else r[1]))
        shuffled_records = sorted_records.copy()
        rng.shuffle(shuffled_records)
        
        # Allocate 70% to train, 30% to test per identity
        n_total = len(shuffled_records)
        n_train = max(1, int(round(n_total * 0.70)))
        
        train_records = shuffled_records[:n_train]
        test_records = shuffled_records[n_train:]
        
        for q, imp, fpath in train_records:
            size_x, size_y, minutiae = parse_fmr(fpath)
            tokens = extract_relative_tokens(minutiae)
            out_filename = f"q{q}_{ident_str}_{imp}.json"
            
            record = {
                "file_name": out_filename,
                "quality_folder": int(q),
                "identity": ident_str,
                "impression": imp,
                "image_size": {"x": size_x, "y": size_y},
                "num_minutiae": len(minutiae),
                "minutiae": minutiae,
                "relative_tokens": tokens
            }
            
            with open(train_dir / out_filename, "w") as out_f:
                json.dump(record, out_f, indent=2)
            train_count += 1

        for q, imp, fpath in test_records:
            size_x, size_y, minutiae = parse_fmr(fpath)
            tokens = extract_relative_tokens(minutiae)
            out_filename = f"q{q}_{ident_str}_{imp}.json"
            
            record = {
                "file_name": out_filename,
                "quality_folder": int(q),
                "identity": ident_str,
                "impression": imp,
                "image_size": {"x": size_x, "y": size_y},
                "num_minutiae": len(minutiae),
                "minutiae": minutiae,
                "relative_tokens": tokens
            }
            
            with open(test_dir / out_filename, "w") as out_f:
                json.dump(record, out_f, indent=2)
            test_count += 1


    print(f"\nDataset construction complete!")
    print(f"Total Fingerprint Identities: {len(identity_files)} (all 1000 present in both Train and Test)")
    print(f"Parsed files in TRAIN directory ({train_dir}): {train_count}")
    print(f"Parsed files in TEST directory  ({test_dir}): {test_count}")
    print(f"Total parsed records: {train_count + test_count}")

if __name__ == "__main__":
    parse_and_save_dataset()
