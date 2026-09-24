from __future__ import annotations
from pathlib import Path
from .xyt import read_xyt
from .efs_ebts import load_json_export

def load_truth(path:Path,fmt:str|None=None)->dict:
    fmt=fmt or path.suffix.lower().lstrip(".")
    if fmt=="json": return load_json_export(path)
    if fmt=="xyt": return {path.with_suffix(".png").name:read_xyt(path)}
    raise ValueError(f"Unsupported ground-truth format: {fmt}. Use json or xyt.")
