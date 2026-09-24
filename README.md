# Starfish — Advanced Fingerprint Minutiae Extraction

Starfish processes user-supplied fingerprint images for enhancement, ridge analysis, minutiae extraction, quality assessment, visualization, and template export.

## Architecture
- Frontend: React + TypeScript + Vite
- API: Python + FastAPI
- Processing: NumPy, OpenCV, scikit-image
- Native layer: C++17 reserved for accelerated kernels
- Tests: pytest
- No fabricated fingerprint images are bundled.

## Pipeline
Decode → normalize → foreground segmentation → orientation estimation → Gabor enhancement → binarization → skeletonization → crossing-number minutiae detection → false-minutiae suppression → orientation → visualization.

## Run
Backend:
```bash
cd backend
python -m venv .venv
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```


## Research basis

See [RESEARCH.md](RESEARCH.md) for the algorithmic literature guiding Starfish, including contextual enhancement, STFT orientation/frequency estimation, directional fields, singular points, wavelet/Wiener alternatives, minutiae extraction/matching, deep minutiae detection and NIST quality assessment.

See [benchmarks/README.md](benchmarks/README.md) for the evaluation protocol.

## Research references
Starfish development is informed by peer-reviewed fingerprint enhancement, orientation/frequency estimation, directional field, minutiae and quality literature. See [RESEARCH.md](RESEARCH.md).

External references include Hong-Wan-Jain contextual enhancement, Chikkerur-Cartwright-Govindaraju STFT analysis, NFIQ 2 / ISO 29794-4, NIST NBIS tools, and MinutiaeNet. These guide implementation; they are not a substitute for independent biometric validation.
