import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

def test_extract_endpoint_is_self_contained():
    img=np.zeros((256,256),np.uint8)
    for y in range(20,236,14):
        cv2.line(img,(30,y),(225,y),255,4)
    ok,encoded=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,85])
    assert ok
    response=TestClient(app).post("/api/v1/extract",files={"file":("fingerprint.jpg",encoded.tobytes(),"image/jpeg")})
    assert response.status_code==200, response.text
    body=response.json()
    assert body["job_id"]
    assert body["artifacts"]["enhanced"].startswith("data:image/jpeg;base64,")
    assert body["artifacts"]["mask"].startswith("data:image/png;base64,")
    assert isinstance(body["singular_points"],list)
    assert body["template"]["format"]=="starfish-minutiae"
    assert body["template"]["version"]==2

def test_health():
    response=TestClient(app).get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="ok"

def test_invalid_content_type_is_rejected():
    response=TestClient(app).post("/api/v1/extract",files={"file":("not.txt",b"hello","text/plain")})
    assert response.status_code==415

def test_oversized_upload_is_rejected():
    response=TestClient(app).post("/api/v1/extract",files={"file":("big.jpg",b"x"*(4_000_001),"image/jpeg")})
    assert response.status_code==413
