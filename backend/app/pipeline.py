import math
from dataclasses import dataclass
import cv2
import numpy as np
from skimage.morphology import skeletonize

@dataclass
class FingerprintPipeline:
    gabor_sigma: float = 4.0
    gabor_lambda: float = 10.0
    min_distance: int = 10

    def _normalize(self, img):
        x = img.astype(np.float32)
        lo, hi = np.percentile(x, (1, 99))
        return np.clip((x - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)

    def _foreground(self, img):
        blur = cv2.GaussianBlur(img, (0, 0), 5)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        energy = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 7)
        mask = (energy > np.percentile(energy, 35)).astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)

    def _orientation(self, img):
        x = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.2)
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.GaussianBlur(gx * gx, (0, 0), 5)
        gyy = cv2.GaussianBlur(gy * gy, (0, 0), 5)
        gxy = cv2.GaussianBlur(gx * gy, (0, 0), 5)
        return 0.5 * np.arctan2(2 * gxy, gxx - gyy + 1e-6) + math.pi / 2

    def _enhance(self, img, mask, orientation):
        imgf = img.astype(np.float32) / 255.0
        angles = np.linspace(0, math.pi, 8, endpoint=False)
        responses = []
        for theta in angles:
            kernel = cv2.getGaborKernel((21, 21), self.gabor_sigma, theta,
                                        self.gabor_lambda, 0.5, 0, cv2.CV_32F)
            responses.append(np.abs(cv2.filter2D(imgf, cv2.CV_32F, kernel)))
        stack = np.stack(responses, axis=-1)
        delta = np.angle(np.exp(1j * (orientation[..., None] - angles)))
        idx = np.argmin(np.abs(delta), axis=-1)
        out = np.take_along_axis(stack, idx[..., None], axis=-1)[..., 0]
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        out[mask == 0] = 0
        return out.astype(np.uint8)

    def _thin(self, enhanced, mask):
        _, bw = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw[mask == 0] = 0
        return (skeletonize(bw > 0).astype(np.uint8) * 255)

    def _crossing_number(self, skel, y, x):
        p = (skel[y-1:y+2, x-1:x+2] > 0).astype(np.uint8)
        ring = [p[0,0],p[0,1],p[0,2],p[1,2],p[2,2],p[2,1],p[2,0],p[1,0]]
        return int(sum(ring[i] != ring[(i+1) % 8] for i in range(8)) / 2)

    def _extract_minutiae(self, skel, orientation, mask):
        ys, xs = np.where(skel > 0)
        out = []
        for y, x in zip(ys, xs):
            if min(x, y) <= 1 or x >= skel.shape[1]-2 or y >= skel.shape[0]-2 or mask[y, x] == 0:
                continue
            cn = self._crossing_number(skel, y, x)
            kind = "ending" if cn == 1 else "bifurcation" if cn == 3 else None
            if kind is None:
                continue
            out.append({"x": int(x), "y": int(y), "type": kind,
                        "angle": float(orientation[y, x]), "quality": 1.0})
        return self._suppress_close(out)

    def _suppress_close(self, points):
        accepted = []
        d2 = self.min_distance ** 2
        for p in points:
            if all((p["x"]-q["x"])**2 + (p["y"]-q["y"])**2 >= d2 for q in accepted):
                accepted.append(p)
        return accepted

    def _overlay(self, base, skeleton, minutiae):
        canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        canvas[skeleton > 0] = (220, 220, 220)
        for m in minutiae:
            c = (0,0,255) if m["type"] == "ending" else (255,0,0)
            x, y = m["x"], m["y"]
            cv2.circle(canvas, (x,y), 4, c, 1, cv2.LINE_AA)
            dx, dy = int(8*math.cos(m["angle"])), int(8*math.sin(m["angle"]))
            cv2.line(canvas, (x,y), (x+dx,y+dy), c, 1, cv2.LINE_AA)
        return canvas

    def process(self, image):
        normalized = self._normalize(image)
        mask = self._foreground(normalized)
        orientation = self._orientation(normalized)
        enhanced = self._enhance(normalized, mask, orientation)
        skeleton = self._thin(enhanced, mask)
        minutiae = self._extract_minutiae(skeleton, orientation, mask)
        ratio = float(np.mean(mask > 0))
        return {
            "enhanced": enhanced,
            "skeleton": skeleton,
            "overlay": self._overlay(enhanced, skeleton, minutiae),
            "minutiae": minutiae,
            "counts": {
                "total": len(minutiae),
                "endings": sum(m["type"] == "ending" for m in minutiae),
                "bifurcations": sum(m["type"] == "bifurcation" for m in minutiae)
            },
            "quality": {
                "foreground_ratio": ratio,
                "status": "usable" if ratio > 0.15 else "low_foreground"
            }
        }
