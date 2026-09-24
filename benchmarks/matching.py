from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class MatchPair:
    pred_index:int
    truth_index:int
    distance_px:float

def angular_error(a:float,b:float)->float:
    d=(float(a)-float(b))%math.pi
    return min(d,math.pi-d)

def greedy_match(pred:list[dict],truth:list[dict],radius:float)->list[MatchPair]:
    candidates=[]; r2=float(radius)**2
    for pi,p in enumerate(pred):
        for gi,g in enumerate(truth):
            if p.get("type")!=g.get("type"): continue
            dx=float(p["x"])-float(g["x"]); dy=float(p["y"])-float(g["y"])
            d2=dx*dx+dy*dy
            if d2<=r2: candidates.append((d2,pi,gi))
    used_p=set(); used_g=set(); pairs=[]
    for d2,pi,gi in sorted(candidates):
        if pi in used_p or gi in used_g: continue
        used_p.add(pi); used_g.add(gi); pairs.append(MatchPair(pi,gi,math.sqrt(d2)))
    return pairs
