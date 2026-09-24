from __future__ import annotations
from pathlib import Path
from .xyt import read_xyt

def xyt_to_truth(images_dir:Path,xyt_dir:Path)->dict:
    truth={}
    for p in sorted(xyt_dir.glob("*.xyt")):
        candidates=[images_dir/(p.stem+ext) for ext in (".png",".jpg",".jpeg",".bmp",".tif",".tiff")]
        image=next((c for c in candidates if c.exists()),None)
        if image is not None: truth[image.name]=read_xyt(p)
    return truth
