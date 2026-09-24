from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np

from app.pipeline import FingerprintPipeline


def load_ground_truth(path: Path):
    with path.open() as f:
        return json.load(f)


def match_minutiae(pred, gt, radius):
    if not pred or not gt:
        return 0, len(pred), len(gt)
    used = set()
    tp = 0
    for p in pred:
        best = None
        best_d = radius
        for i, g in enumerate(gt):
            if i in used or p["type"] != g["type"]:
                continue
            d = float(np.hypot(p["x"] - g["x"], p["y"] - g["y"]))
            if d <= best_d:
                best_d = d
                best = i
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def evaluate(image_dir: Path, truth: dict, radius: float):
    pipe = FingerprintPipeline()
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "loc_sum": 0.0, "loc_n": 0}
    for name, gt in truth.items():
        image = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        result = pipe.process(image)
        pred = result["minutiae"]
        tp, fp, fn = match_minutiae(pred, gt, radius)
        rows.append({
            "image": name,
            "predicted": len(pred),
            "ground_truth": len(gt),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "quality_status": result["quality"]["status"],
            "coherence_p50": result["quality"]["coherence_p50"],
            "ridge_frequency_median": result["quality"]["ridge_frequency_median"],
        })
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    summary = {"precision": precision, "recall": recall, "f1": f1, **totals, "localization_rmse_px": (
        float(np.sqrt(totals["loc_sum"] / totals["loc_n"])) if totals["loc_n"] else None
    )}
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=Path("benchmark-results.json"))
    args = ap.parse_args()
    summary, rows = evaluate(args.images, load_ground_truth(args.truth), args.radius)
    args.out.write_text(json.dumps({"summary": summary, "images": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
