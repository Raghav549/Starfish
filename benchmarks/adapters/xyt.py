from __future__ import annotations
from pathlib import Path

def parse_xyt(text: str) -> list[dict]:
    points=[]
    for lineno, raw in enumerate(text.splitlines(),1):
        line=raw.strip()
        if not line or line.startswith("#"): continue
        parts=line.split()
        if len(parts)<4: raise ValueError(f"Invalid XYT line {lineno}: expected x y theta quality")
        x,y,theta,quality=map(float,parts[:4])
        points.append({"x":x,"y":y,"angle":theta,"quality":quality})
    return points

def read_xyt(path: Path)->list[dict]:
    return parse_xyt(path.read_text(encoding="utf-8"))
