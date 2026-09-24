from pathlib import Path
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from .pipeline import FingerprintPipeline

BASE = Path(__file__).resolve().parents[1]
OUTPUTS = BASE / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Starfish Fingerprint Engine", version="0.1.0")
pipeline = FingerprintPipeline()

@app.get("/health")
def health():
    return {"status": "ok", "service": "starfish"}

@app.post("/api/v1/extract")
async def extract(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file.")
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    result = pipeline.process(image)
    job_id = uuid.uuid4().hex
    for name in ("enhanced", "skeleton", "overlay"):
        cv2.imwrite(str(OUTPUTS / f"{job_id}_{name}.png"), result[name])
    return {
        "job_id": job_id,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "quality": result["quality"],
        "counts": result["counts"],
        "minutiae": result["minutiae"],
        "artifacts": {
            "enhanced": f"/api/v1/artifacts/{job_id}/enhanced",
            "skeleton": f"/api/v1/artifacts/{job_id}/skeleton",
            "overlay": f"/api/v1/artifacts/{job_id}/overlay"
        }
    }

@app.get("/api/v1/artifacts/{job_id}/{kind}")
def artifact(job_id: str, kind: str):
    if kind not in {"enhanced", "skeleton", "overlay"}:
        raise HTTPException(status_code=404)
    path = OUTPUTS / f"{job_id}_{kind}.png"
    if not path.exists():
        raise HTTPException(status_code=404)
    return StreamingResponse(path.open("rb"), media_type="image/png")
