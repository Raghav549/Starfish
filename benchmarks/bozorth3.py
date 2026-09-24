from __future__ import annotations
import csv,shutil,subprocess
from pathlib import Path

class Bozorth3Unavailable(RuntimeError): pass

def find_bozorth3(explicit:str|None=None)->str:
    candidate=explicit or shutil.which("bozorth3")
    if not candidate: raise Bozorth3Unavailable("bozorth3 executable not found; install NIST NBIS and pass --bozorth3 /path/to/bozorth3")
    return candidate

def score_pair(exe:str,a:Path,b:Path)->float:
    proc=subprocess.run([exe,str(a),str(b)],check=True,capture_output=True,text=True)
    for token in reversed(proc.stdout.strip().split()):
        try: return float(token)
        except ValueError: pass
    raise ValueError(f"Could not parse BOZORTH3 output: {proc.stdout!r}")

def write_scores(path:Path,rows:list[dict])->None:
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["kind","a","b","score"]); w.writeheader(); w.writerows(rows)
