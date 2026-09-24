from __future__ import annotations
from pathlib import Path
import json,re
from typing import Any

TYPE_MAP={
    "ending":"ending","ridgeending":"ending","ridge_ending":"ending",
    "bifurcation":"bifurcation","ridgebifurcation":"bifurcation","ridge_bifurcation":"bifurcation",
}

def _feature_type(value:Any)->str|None:
    key=re.sub(r"[^a-z0-9_]+","",str(value).lower().replace(" ","_"))
    return TYPE_MAP.get(key)

def normalize_feature(feature:dict[str,Any])->dict[str,Any]:
    f={str(k).lower():v for k,v in feature.items()}
    typ=_feature_type(f.get("type",f.get("feature_type",f.get("minutia_type"))))
    if typ is None: raise ValueError(f"Unsupported minutia type: {feature.get('type')!r}")
    x=f.get("x",f.get("x_coordinate",f.get("xcoord")))
    y=f.get("y",f.get("y_coordinate",f.get("ycoord")))
    if x is None or y is None: raise ValueError("Minutia is missing x/y coordinates")
    out={"x":float(x),"y":float(y),"type":typ}
    angle=f.get("angle",f.get("theta",f.get("orientation")))
    if angle is not None: out["angle"]=float(angle)
    return out

def records_to_truth(records:dict[str,list[dict[str,Any]]])->dict[str,list[dict[str,Any]]]:
    return {str(name):[normalize_feature(f) for f in features] for name,features in records.items()}

def load_json_export(path:Path)->dict[str,list[dict[str,Any]]]:
    data=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict): raise ValueError("Annotation export must be an object keyed by image name")
    return records_to_truth(data)
