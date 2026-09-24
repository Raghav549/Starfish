from pathlib import Path
import json, uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from .pipeline import FingerprintPipeline

BASE=Path(__file__).resolve().parents[1]
OUTPUTS=BASE/"outputs"
OUTPUTS.mkdir(parents=True,exist_ok=True)

app=FastAPI(title="Starfish Fingerprint Engine",version="0.6.0")
pipeline=FingerprintPipeline()
MAX_UPLOAD_BYTES=4_000_000
MAX_SIDE=1600

@app.get("/health")
def health():
    return {"status":"ok","service":"starfish","version":"0.6.0"}

async def decode_image(file:UploadFile)->np.ndarray:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415,"Upload a PNG, JPEG, or WEBP image.")
    raw=await file.read(MAX_UPLOAD_BYTES+1)
    if len(raw)>MAX_UPLOAD_BYTES:
        raise HTTPException(413,"IMAGE_TOO_LARGE: upload must be 4 MB or smaller after client compression.")
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(400,"Could not decode the uploaded image.")
    h,w=image.shape
    scale=min(1.0,MAX_SIDE/max(h,w))
    if scale<1:
        image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    return image

@app.post("/api/v1/extract")
async def extract(file:UploadFile=File(...)):
    image=await decode_image(file)
    try:
        result=pipeline.process(image)
    except Exception as exc:
        raise HTTPException(422,f"PROCESSING_ERROR: {type(exc).__name__}: {exc}") from exc
    job=uuid.uuid4().hex
    for name in ("enhanced","mask","skeleton","overlay"):
        if not cv2.imwrite(str(OUTPUTS/f"{job}_{name}.png"),result[name]):
            raise HTTPException(500,f"Could not write {name} artifact.")
    (OUTPUTS/f"{job}_template.json").write_text(json.dumps(result["template"],indent=2),encoding="utf-8")
    return {"job_id":job,"width":int(image.shape[1]),"height":int(image.shape[0]),"quality":result["quality"],"counts":result["counts"],"minutiae":result["minutiae"],"template":result["template"],"artifacts":{k:f"/api/v1/artifacts/{job}/{k}" for k in ("enhanced","mask","skeleton","overlay","template")}}

@app.get("/api/v1/artifacts/{job_id}/{kind}")
def artifact(job_id:str,kind:str):
    if kind=="template":
        p=OUTPUTS/f"{job_id}_template.json"
        if not p.exists(): raise HTTPException(404,"Artifact not found.")
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
    if kind not in {"enhanced","mask","skeleton","overlay"}:
        raise HTTPException(404,"Artifact not found.")
    p=OUTPUTS/f"{job_id}_{kind}.png"
    if not p.exists(): raise HTTPException(404,"Artifact not found.")
    return StreamingResponse(p.open("rb"),media_type="image/png")

@app.get("/api/v1/download/{job_id}/{kind}")
def download(job_id:str,kind:str):
    if kind=="template":
        p=OUTPUTS/f"{job_id}_template.json"; media="application/json"; name=f"starfish_{job_id}_template.json"
    elif kind in {"enhanced","mask","skeleton","overlay"}:
        p=OUTPUTS/f"{job_id}_{kind}.png"; media="image/png"; name=f"starfish_{job_id}_{kind}.png"
    else:
        raise HTTPException(404,"Artifact not found.")
    if not p.exists(): raise HTTPException(404,"Artifact not found.")
    return FileResponse(p,media_type=media,filename=name)
