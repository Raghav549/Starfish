import cv2,numpy as np
from app.pipeline import FingerprintPipeline

def test_pipeline_integrity():
    img=np.zeros((256,256),np.uint8)
    for y in range(20,236,14): cv2.line(img,(30,y),(225,y),255,4)
    r=FingerprintPipeline().process(img)
    assert r["enhanced"].shape==img.shape
    assert r["reconstruction"].shape==img.shape
    assert r["mask"].shape==img.shape
    assert set(r["counts"])=={"total","endings","bifurcations","singular_points"}
    assert r["template"]["format"]=="starfish-minutiae"
    assert r["template"]["version"]==2
    assert 0.0<=r["quality"]["foreground_ratio"]<=1.0
    assert 0.0<=r["quality"]["frequency_coverage"]<=1.0
    assert "ridge_frequency_median" in r["quality"]
    assert "orientation_concentration" in r["quality"]
    assert "local_clarity_mean" in r["quality"]
    assert "orientation_certainty_mean" in r["quality"]
    assert "orientation_flow_mean" in r["quality"]
    assert "ridge_valley_uniformity" in r["quality"]
    assert isinstance(r["singular_points"],list)

def test_uniform_image_is_safe():
    img=np.full((128,128),128,np.uint8)
    r=FingerprintPipeline().process(img)
    assert r["enhanced"].shape==img.shape
    assert isinstance(r["minutiae"],list)
    assert r["quality"]["status"]=="review"
