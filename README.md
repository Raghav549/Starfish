# Starfish — Advanced Fingerprint Imaging & Minutiae Analysis

Starfish processes user-supplied fingerprint images for enhancement, ridge analysis, minutiae extraction, global-pattern diagnostics, quality assessment, visualization, and template export.

## Architecture
- Frontend: React + TypeScript + Vite
- API: Python + FastAPI
- Processing: NumPy, OpenCV, scikit-image
- Native layer: C++17 reserved for accelerated kernels
- Tests: pytest
- No fabricated fingerprint images are bundled.

## Current pipeline
Decode → normalize → foreground segmentation → orientation/coherence estimation → local ridge-frequency/STFT-style estimation → contextual Gabor enhancement → local Sauvola binarization → skeletonization → spur pruning → crossing-number candidates → ridge-support validation → spatial non-maximum suppression → minutiae quality → Poincaré-style singular-point candidates → structural template.

## Validation
The repository includes a reproducible ground-truth evaluator in `benchmarks/evaluate.py`. It computes one-to-one minutiae precision, recall, F1, coordinate localization RMSE, and orientation MAE when labeled orientations are available. Ground-truth format is defined in `benchmarks/schema.json`.

Run evaluation with:

```bash
PYTHONPATH=backend python benchmarks/evaluate.py --images /path/to/images --truth /path/to/truth.json --radius 8
```

For independent comparison, NIST NBIS documents MINDTCT as a minutiae detector and BOZORTH3 as a minutiae-based matcher; NFIQ 2 is the NIST reference implementation associated with ISO/IEC 29794-4 fingerprint image quality.

No benchmark accuracy number is fabricated or claimed without labeled data. A real evaluation dataset is required to produce empirical scores.

## Research basis
See `RESEARCH.md` for the literature basis. The implementation follows established fingerprint image-processing families including contextual enhancement, STFT analysis, directional fields/singular points, minutiae extraction/matching and biometric quality assessment.

## Limitations
Enhancement cannot recover information absent from the captured image. A visually sharper image is not proof of biometric correctness. Accuracy requires subject-disjoint labeled evaluation and, for recognition claims, matcher-level testing on the intended sensor/domain.
