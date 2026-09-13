import random
import json
from pathlib import Path
import hashlib
from tqdm import tqdm
folder = Path(r".\fingerprint_dataset\parsed_data\train")
write_f=r".\data_structure\signatures.json"

def make_hash_family(n, seed=None):
    rng = random.Random(seed)
    p = 4294967311  
    hash_funcs = []
    for _ in range(n):
        a = rng.randint(1, p - 1) 
        b = rng.randint(0, p - 1)
        hash_funcs.append((a, b, p))
    return hash_funcs

#this is converthing shingles to a number using a hash function, here collisions shouldn't matter too much
def token_to_int(token):
    b = repr(token).encode()
    return int.from_bytes(
        hashlib.blake2b(b, digest_size=8).digest(),
        "big"
    )


def minhash_signature(shingle_set, hash_funcs):
    sig = []
    for (a, b, p) in hash_funcs:
        best = None
        for token in shingle_set:
            x = token_to_int(token)
            h = (a * x + b) % p
            if best is None or h < best:
                best = h
        sig.append(best)
    return sig


hash_funcs = make_hash_family(n=64, seed=42)

if __name__ == "__main__":
    sigs = []
    for file in tqdm(folder.iterdir()):
        if file.is_file():
            with open(file, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
                sigs.append({
                    'file_name': doc_data['file_name'],
                    'signatures': minhash_signature(doc_data['relative_tokens'], hash_funcs)
                })

    with open(write_f, 'w') as d:
        json.dump(sigs, d, indent=4)
 