from pathlib import Path
import json,uuid
import cv2,numpy as np
from fastapi import FastAPI,File,HTTPException,UploadFile
from fastapi.responses import FileResponse,JSONResponse,StreamingResponse
from .pipeline import FingerprintPipeline

BASE=Path(__file__).resolve().parents[1]; OUTPUTS=BASE/"outputs"; OUTPUTS.mkdir(parents=True,exist_ok=True)
app=FastAPI(title="Starfish Fingerprint Engine",version="0.4.0"); pipeline=FingerprintPipeline()

@app.get("/health")
def health(): return {"status":"ok","service":"starfish","version":"0.4.0"}

@app.post("/api/v1/extract")
async def extract(file:UploadFile=File(...)):
    if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(415,"Upload an image file.")
    raw=await file.read()
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
    if image is None: raise HTTPException(400,"Could not decode image.")
    result=pipeline.process(image); job=uuid.uuid4().hex
    for name in ("enhanced","mask","skeleton","overlay"): cv2.imwrite(str(OUTPUTS/f"{job}_{name}.png"),result[name])
    (OUTPUTS/f"{job}_template.json").write_text(json.dumps(result["template"],indent=2))
    return {"job_id":job,"width":int(image.shape[1]),"height":int(image.shape[0]),"quality":result["quality"],"counts":result["counts"],"minutiae":result["minutiae"],"template":result["template"],"artifacts":{k:f"/api/v1/artifacts/{job}/{k}" for k in ("enhanced","mask","skeleton","overlay","template")}}

@app.get("/api/v1/artifacts/{job_id}/{kind}")
def artifact(job_id:str,kind:str):
    if kind=="template":
        p=OUTPUTS/f"{job_id}_template.json"
        if not p.exists(): raise HTTPException(404)
        return JSONResponse(json.loads(p.read_text()))
    if kind not in {"enhanced","mask","skeleton","overlay"}: raise HTTPException(404)
    p=OUTPUTS/f"{job_id}_{kind}.png"
    if not p.exists(): raise HTTPException(404)
    return StreamingResponse(p.open("rb"),media_type="image/png")

@app.get("/api/v1/download/{job_id}/{kind}")
def download(job_id:str,kind:str):
    if kind=="template":
        p=OUTPUTS/f"{job_id}_template.json"; media="application/json"; name=f"starfish_{job_id}_template.json"
    elif kind in {"enhanced","mask","skeleton","overlay"}:
        p=OUTPUTS/f"{job_id}_{kind}.png"; media="image/png"; name=f"starfish_{job_id}_{kind}.png"
    else: raise HTTPException(404)
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,media_type=media,filename=name)
