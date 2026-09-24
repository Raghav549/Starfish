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
