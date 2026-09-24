from __future__ import annotations
from itertools import combinations

def make_pairs(items:list[dict])->tuple[list[tuple[dict,dict]],list[tuple[dict,dict]]]:
    genuine=[]; impostor=[]
    for a,b in combinations(items,2):
        if a.get("subject_id") is not None and a.get("subject_id")==b.get("subject_id"): genuine.append((a,b))
        elif a.get("subject_id") is not None and b.get("subject_id") is not None: impostor.append((a,b))
    return genuine,impostor
