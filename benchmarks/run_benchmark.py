from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from evaluate import load_ground_truth, evaluate


def run(images: Path, truth: Path, out_dir: Path, radius: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    summary, rows = evaluate(images, load_ground_truth(truth), radius)
    elapsed = time.perf_counter() - t0
    summary["elapsed_seconds"] = elapsed
    summary["images_evaluated"] = len(rows)
    summary["images_per_second"] = len(rows) / elapsed if elapsed > 0 else None
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "per_image.csv").open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run a reproducible Starfish benchmark.")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark-results"))
    args = ap.parse_args()
    summary = run(args.images, args.truth, args.out_dir, args.radius)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
