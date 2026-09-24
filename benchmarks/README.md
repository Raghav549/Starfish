# Starfish biometric validation

This directory contains reproducible evaluation code. It deliberately does not fabricate benchmark numbers: a run requires real labeled ground truth.

## Metrics
The evaluator reports:
- minutiae precision, recall and F1
- coordinate localization RMSE for matched minutiae
- orientation mean absolute error when labeled angles are supplied
- TP/FP/FN counts per image
- quality diagnostics alongside each image

The matching protocol uses one-to-one greedy assignment within a configurable pixel radius and requires minutiae type agreement.

## Ground-truth format
See `schema.json`. Each image maps to an array of labeled ridge endings/bifurcations with x/y coordinates; optional orientation is in radians.

## Run
From repository root:

```bash
PYTHONPATH=backend python benchmarks/evaluate.py \
  --images /path/to/images \
  --truth /path/to/truth.json \
  --radius 8 \
  --out benchmark-results.json
```

## Dataset discipline
Use a dataset whose license permits the intended evaluation. Record sensor type, nominal resolution, dataset split, preprocessing, and evaluation radius. Keep subject-disjoint train/test splits where learning is introduced.

## Independent reference baselines
For external validation, compare against documented reference systems where their licensing and environment permit, including NIST NBIS MINDTCT and BOZORTH3. NIST describes MINDTCT as a minutiae detector with local quality assessment and BOZORTH3 as a minutiae-based matcher. NFIQ 2 is the NIST reference implementation for ISO/IEC 29794-4 fingerprint quality. See RESEARCH.md for references.

## Interpretation
A visual enhancement is not an accuracy measurement. No benchmark score is claimed until a labeled dataset is actually supplied and evaluated.
