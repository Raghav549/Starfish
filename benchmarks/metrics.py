from __future__ import annotations
import math
import numpy as np
from .matching import MatchPair, angular_error

def summarize(pred:list[dict],truth:list[dict],pairs:list[MatchPair])->dict:
    tp=len(pairs); fp=len(pred)-tp; fn=len(truth)-tp
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    f1=2*precision*recall/max(precision+recall,1e-12)
    loc=[p.distance_px**2 for p in pairs]
    angles=[angular_error(pred[p.pred_index]["angle"],truth[p.truth_index]["angle"])
            for p in pairs if "angle" in pred[p.pred_index] and "angle" in truth[p.truth_index]]
    return {"tp":tp,"fp":fp,"fn":fn,"precision":float(precision),"recall":float(recall),"f1":float(f1),
            "localization_rmse_px":float(np.sqrt(np.mean(loc))) if loc else None,
            "orientation_mae_rad":float(np.mean(angles)) if angles else None,"matched_pairs":tp}
