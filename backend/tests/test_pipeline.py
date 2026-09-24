import cv2
import numpy as np
from app.pipeline import FingerprintPipeline

def test_pipeline_integrity():
    img = np.zeros((256,256), np.uint8)
    for y in range(20,236,14):
        cv2.line(img, (30,y), (225,y), 255, 4)
    result = FingerprintPipeline().process(img)
    assert result["enhanced"].shape == img.shape
    assert "counts" in result
    assert isinstance(result["minutiae"], list)
