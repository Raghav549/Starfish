import base64
import json, uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from .pipeline import FingerprintPipeline

app=FastAPI(title="Starfish Fingerprint Engine",version="0.8.0")
pipeline=FingerprintPipeline()
MAX_UPLOAD_BYTES=4_000_000
MAX_SIDE=1600

@app.get("/")
def root():
    return {"service":"starfish","status":"ok","api":"/health"}

@app.get("/health")
def health():
    return {"status":"ok","service":"starfish","version":"0.8.0"}

async def decode_image(file:UploadFile)->np.ndarray:
    raw=await file.read(MAX_UPLOAD_BYTES+1)
    if len(raw)>MAX_UPLOAD_BYTES:
        raise HTTPException(413,"IMAGE_TOO_LARGE: upload must be 4 MB or smaller after compression.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415,"Upload a PNG, JPEG, or WEBP image.")
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(400,"Could not decode the uploaded image.")
    h,w=image.shape
    scale=min(1.0,MAX_SIDE/max(h,w))
    if scale<1:
        image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    return image

def image_data_url(image:np.ndarray,kind:str)->str:
    if image.ndim==2:
        ok,data=cv2.imencode(".png",image,[cv2.IMWRITE_PNG_COMPRESSION,6])
        mime="image/png"
    else:
        ok,data=cv2.imencode(".jpg",image,[cv2.IMWRITE_JPEG_QUALITY,82])
        mime="image/jpeg"
    if not ok:
        raise RuntimeError(f"artifact encoding failed: {kind}")
    return f"data:{mime};base64,{base64.b64encode(data.tobytes()).decode('ascii')}"

@app.post("/api/v1/extract")
async def extract(file:UploadFile=File(...)):
    image=await decode_image(file)
    try:
        result=pipeline.process(image)
        # Vercel Functions do not provide durable writable project storage.
        # Keep each analysis response self-contained instead of writing artifacts.
        artifacts={
            "enhanced":image_data_url(result["enhanced"],"enhanced"),
            "mask":image_data_url(result["mask"],"mask"),
            "skeleton":image_data_url(result["skeleton"],"skeleton"),
            "overlay":image_data_url(result["overlay"],"overlay"),
        }
        return {
            "job_id":uuid.uuid4().hex,
            "width":int(image.shape[1]),
            "height":int(image.shape[0]),
            "quality":result["quality"],
            "counts":result["counts"],
            "minutiae":result["minutiae"],
            "template":result["template"],
            "artifacts":artifacts,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500,f"EXTRACTION_FAILED: {type(exc).__name__}: {exc}") from exc
