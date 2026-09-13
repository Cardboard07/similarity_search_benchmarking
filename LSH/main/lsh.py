import os
import json
from pathlib import Path

_base_dir = Path(__file__).resolve().parent.parent
write_f = _base_dir / "data_structure" / "signatures.json"
if not write_f.exists():
    write_f = Path(r".\data_structure\signatures.json")

data=[]
with open(write_f, "r", encoding="utf-8") as f:
    data = json.load(f)

bands=16    
 
def lsh(data,bands=bands):
    dicts=[]
    rows = len(data[0]['signatures']) // bands
    for i in range(bands):
        dicts.append({})
        for j in range(len(data)):
            band=tuple(data[j]['signatures'][rows*i : rows*(i+1)])
            if band in dicts[i]:
                dicts[i][band].append(data[j])
            else:
                dicts[i][band]=[]
                dicts[i][band].append(data[j])
    return dicts

def find_cand(dicts,data):
    candidates = []
    bands=len(dicts)
    seen = set()
    rows = len(data['signatures']) // bands
    for i in range(len(dicts)):
        band=tuple(data['signatures'][rows*i : rows*(i+1)])
        if band in dicts[i]:
           for record in dicts[i][band]:
                if record["file_name"] not in seen:
                    candidates.append(record)
                    seen.add(record["file_name"])
    return candidates

dicts = lsh(data, bands=8)

