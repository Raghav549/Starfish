from __future__ import annotations
from pathlib import Path
from .xyt import read_xyt

def xyt_to_truth(images_dir:Path,xyt_dir:Path)->dict:
    truth={}
    for p in sorted(xyt_dir.glob("*.xyt")):
        for ext in (".png",".jpg",".jpeg",".bmp",".tif",".tiff"):
            image=images_dir/(p.stem+ext)
            if image.exists():
                truth[image.name]=read_xyt(p)
                break
    return truth
