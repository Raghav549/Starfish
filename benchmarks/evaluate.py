from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from app.pipeline import FingerprintPipeline


def load_ground_truth(path: Path):
    return json.loads(path.read_text())


def angular_error(a: float, b: float) -> float:
    d = (float(a) - float(b)) % np.pi
    return float(min(d, np.pi - d))


def greedy_match(pred, gt, radius):
    pairs = []
    candidates = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if p.get("type") != g.get("type"):
                continue
            d = float(np.hypot(float(p["x"]) - float(g["x"]), float(p["y"]) - float(g["y"])))
            if d <= radius:
                candidates.append((d, pi, gi))
    used_p, used_g = set(), set()
    for d, pi, gi in sorted(candidates):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); pairs.append((pi, gi, d))
    return pairs


def evaluate(image_dir: Path, truth: dict, radius: float):
    pipe = FingerprintPipeline()
    rows = []
    total_tp = total_fp = total_fn = 0
    sq_err = []
    angle_err = []
    for name, gt in truth.items():
        image = cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            rows.append({"image": name, "status": "missing_image"})
            continue
        result = pipe.process(image)
        pred = result["minutiae"]
        pairs = greedy_match(pred, gt, radius)
        matched_p = {p for p, _, _ in pairs}
        matched_g = {g for _, g, _ in pairs}
        total_tp += len(pairs)
        total_fp += len(pred) - len(matched_p)
        total_fn += len(gt) - len(matched_g)
        for pi, gi, d in pairs:
            sq_err.append(d * d)
            if "angle" in gt[gi] and "angle" in pred[pi]:
                angle_err.append(angular_error(pred[pi]["angle"], gt[gi]["angle"]))
        rows.append({
            "image": name,
            "predicted": len(pred),
            "ground_truth": len(gt),
            "tp": len(pairs),
            "fp": len(pred) - len(matched_p),
            "fn": len(gt) - len(matched_g),
            "quality_status": result["quality"]["status"],
            "coherence_p50": result["quality"]["coherence_p50"],
            "ridge_frequency_median": result["quality"]["ridge_frequency_median"],
        })

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    summary = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "localization_rmse_px": float(np.sqrt(np.mean(sq_err))) if sq_err else None,
        "orientation_mae_rad": float(np.mean(angle_err)) if angle_err else None,
        "matched_pairs": len(sq_err),
        "radius_px": float(radius),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate Starfish minutiae against labeled ground truth.")
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
