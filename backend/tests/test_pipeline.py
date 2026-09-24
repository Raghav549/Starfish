import cv2, numpy as np
from app.pipeline import FingerprintPipeline
def test_pipeline_integrity():
 img=np.zeros((256,256),np.uint8)
 for y in range(20,236,14): cv2.line(img,(30,y),(225,y),255,4)
 r=FingerprintPipeline().process(img)
 assert r["enhanced"].shape==img.shape
 assert set(r["counts"])=={"total","endings","bifurcations"}
 assert r["template"]["format"]=="starfish-minutiae"
 assert isinstance(r["quality"]["foreground_ratio"],float)
