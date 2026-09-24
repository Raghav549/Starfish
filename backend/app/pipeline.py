from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import threshold_sauvola


@dataclass
class FingerprintPipeline:
    block_size: int = 16
    gabor_sigma: float = 4.0
    min_distance: int = 10
    border_margin: int = 12
    min_component: int = 24
    max_trace: int = 28
    ridge_step_px: float = 1.0

    def _normalize(self, img):
        x = img.astype(np.float32)
        p1, p99 = np.percentile(x, (1, 99))
        return np.clip((x - p1) / max(p99 - p1, 1.0) * 255, 0, 255).astype(np.uint8)

    def _quality(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
        gyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
        gxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)
        den = gxx + gyy + 1e-6
        coh = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / den
        energy = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 5)
        return {
            "ridge_energy": float(np.mean(energy)),
            "coherence": float(np.mean(coh)),
            "coherence_p50": float(np.median(coh)),
        }

    def _segment(self, img):
        local_std = cv2.GaussianBlur(img.astype(np.float32) ** 2, (0, 0), 7)
        mean = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 7)
        var = np.maximum(local_std - mean * mean, 0)
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        energy = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 7)
        score = 0.55 * cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX) + 0.45 * cv2.normalize(
            energy, None, 0, 1, cv2.NORM_MINMAX
        )
        cutoff = max(float(np.percentile(score, 35)), 0.04)
        mask = (score > cutoff).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean = np.zeros_like(mask)
        min_area = max(self.min_component, int(mask.size * 0.001))
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean[labels == i] = 1
        # Keep the dominant connected foreground component(s), avoiding tiny texture islands.
        if np.any(clean):
            comps = []
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    comps.append((area, i))
            comps.sort(reverse=True)
            keep = max(1, min(3, len(comps)))
            out = np.zeros_like(clean)
            for _, i in comps[:keep]:
                out[labels == i] = 1
            clean = out
        return clean * 255

    def _orientation_frequency(self, img, mask):
        h, w = img.shape
        bs = self.block_size
        gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
        gyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
        gxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)
        theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy + 1e-6) + math.pi / 2
        theta = cv2.GaussianBlur(theta.astype(np.float32), (0, 0), 2)
        freq = np.zeros_like(theta, dtype=np.float32)
        coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-6)

        for y in range(0, h - bs + 1, bs):
            for x in range(0, w - bs + 1, bs):
                block_mask = mask[y:y + bs, x:x + bs]
                if np.mean(block_mask > 0) < 0.55:
                    continue
                ang = float(np.median(theta[y:y + bs, x:x + bs]))
                block = img[y:y + bs, x:x + bs].astype(np.float32)
                yy, xx = np.indices(block.shape)
                coord = xx * math.cos(ang) + yy * math.sin(ang)
                bins = np.arange(-bs, bs + 1, dtype=np.float32)
                profile = np.array([
                    block[np.abs(coord - b) < 0.5].mean()
                    if np.any(np.abs(coord - b) < 0.5) else 0.0
                    for b in bins
                ], dtype=np.float32)
                profile -= profile.mean()
                spec = np.abs(np.fft.rfft(profile))
                if len(spec) > 4:
                    k = int(np.argmax(spec[2:]) + 2)
                    f = k / max(len(profile), 1)
                    if 0.03 <= f <= 0.35:
                        freq[y:y + bs, x:x + bs] = f
        freq = cv2.GaussianBlur(freq, (0, 0), 2)
        return theta, freq, coherence

    def _enhance(self, img, mask, ori, freq):
        src = img.astype(np.float32) / 255.0
        valid = freq[(freq > 0) & (mask > 0)]
        ridge_freq = float(np.median(valid)) if valid.size else 0.10
        wavelength = float(np.clip(1.0 / max(ridge_freq, 1e-3), 6.0, 24.0))
        out = np.zeros_like(src)

        # Contextual Gabor bank, weighted continuously by local orientation.
        for theta in np.linspace(0, math.pi, 18, endpoint=False):
            kernel = cv2.getGaborKernel(
                (31, 31), self.gabor_sigma, float(theta), wavelength, 0.5, 0, cv2.CV_32F
            )
            response = cv2.filter2D(src, cv2.CV_32F, kernel)
            delta = np.abs(np.angle(np.exp(1j * (ori - theta))))
            weight = np.exp(-(delta ** 2) / (2 * (math.pi / 20) ** 2))
            out += np.maximum(response, 0) * weight

        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out[mask == 0] = 0

        # Ridge-preserving local contrast enhancement.
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        out = clahe.apply(out)
        den = cv2.GaussianBlur(out, (0, 0), 0.7)
        out = cv2.addWeighted(out, 1.25, den, -0.25, 0)
        out[mask == 0] = 0
        return out

    def _binarize(self, enhanced, mask):
        # Sauvola gives a local threshold better suited to nonuniform fingerprint illumination.
        win = max(15, self.block_size * 2 + 1)
        threshold = threshold_sauvola(enhanced, window_size=win, k=0.18, r=128)
        bw = enhanced > threshold
        bw &= mask > 0
        bw = cv2.morphologyEx(bw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return bw

    def _skeleton(self, enhanced, mask):
        bw = self._binarize(enhanced, mask)
        sk = skeletonize(bw > 0)
        sk = remove_small_objects(sk, min_size=16, connectivity=2)
        # Iterative spur removal using local branch/end geometry.
        sk = self._prune_spurs(sk, iterations=2)
        return sk.astype(np.uint8) * 255

    def _neighbors(self, sk, y, x):
        p = sk[max(0, y - 1):y + 2, max(0, x - 1):x + 2] > 0
        coords = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < sk.shape[0] and 0 <= xx < sk.shape[1] and sk[yy, xx]:
                    coords.append((yy, xx))
        return coords

    def _prune_spurs(self, sk_bool, iterations=2):
        sk = sk_bool.copy()
        for _ in range(iterations):
            ends = []
            ys, xs = np.where(sk)
            for y, x in zip(ys, xs):
                n = self._neighbors(sk, int(y), int(x))
                if len(n) == 1:
                    ends.append((int(y), int(x)))
            to_remove = set()
            for sy, sx in ends:
                path = [(sy, sx)]
                prev = None
                cur = (sy, sx)
                for _ in range(1, self.max_trace):
                    nbrs = [p for p in self._neighbors(sk, *cur) if p != prev]
                    if not nbrs:
                        break
                    nxt = nbrs[0]
                    path.append(nxt)
                    if len(self._neighbors(sk, *nxt)) != 2:
                        break
                    prev, cur = cur, nxt
                if len(path) < 7 and len(self._neighbors(sk, *path[-1])) >= 3:
                    to_remove.update(path[:-1])
            for y, x in to_remove:
                sk[y, x] = False
        return sk

    def _cn(self, sk, y, x):
        ring = []
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
            yy, xx = y + dy, x + dx
            ring.append(1 if 0 <= yy < sk.shape[0] and 0 <= xx < sk.shape[1] and sk[yy, xx] else 0)
        return sum(abs(ring[i] - ring[(i + 1) % 8]) for i in range(8)) / 2.0

    def _trace_direction(self, sk, y, x, orientation, steps=18):
        angle = float(orientation)
        pts = []
        cur = (float(x), float(y))
        for _ in range(steps):
            cx, cy = cur
            ix, iy = int(round(cx)), int(round(cy))
            if ix < 1 or iy < 1 or ix >= sk.shape[1] - 1 or iy >= sk.shape[0] - 1:
                break
            nbrs = self._neighbors(sk > 0, iy, ix)
            if not nbrs:
                break
            candidates = []
            prev_ang = angle
            for ny, nx in nbrs:
                a = math.atan2(ny - cy, nx - cx)
                d = abs(math.atan2(math.sin(a - prev_ang), math.cos(a - prev_ang)))
                candidates.append((d, nx, ny, a))
            candidates.sort(key=lambda z: z[0])
            _, nx, ny, a = candidates[0]
            pts.append((nx, ny))
            angle = a
            cur = (float(nx), float(ny))
        return pts

    def _ridge_support(self, sk, y, x, orientation):
        forward = self._trace_direction(sk, y, x, orientation, self.max_trace)
        backward = self._trace_direction(sk, y, x, orientation + math.pi, self.max_trace)
        return len(forward), len(backward)

    def _local_quality(self, enhanced, ori, coherence, y, x):
        h, w = enhanced.shape
        r = 5
        patch = enhanced[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
        local_std = float(np.std(patch) / 64.0)
        coh = float(np.clip(coherence[y, x], 0, 1))
        q = 0.45 * np.clip(local_std, 0, 1) + 0.55 * coh
        return float(np.clip(q, 0, 1))

    def _extract(self, sk, ori, mask, coherence):
        ys, xs = np.where(sk > 0)
        raw = []
        h, w = sk.shape
        for y, x in zip(ys, xs):
            y, x = int(y), int(x)
            if x < self.border_margin or y < self.border_margin or x >= w-self.border_margin or y >= h-self.border_margin:
                continue
            if mask[y, x] == 0:
                continue
            cn = self._cn(sk, y, x)
            if cn not in (1, 3):
                continue
            typ = "ending" if cn == 1 else "bifurcation"
            a = float(ori[y, x])
            fwd, back = self._ridge_support(sk, y, x, a)
            if fwd < 5 or back < 3:
                continue
            quality = self._local_quality(cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX), ori, coherence, y, x)
            raw.append({
                "x": x, "y": y, "type": typ,
                "angle": a,
                "quality": quality,
                "crossing_number": float(cn),
                "ridge_support_forward": int(fwd),
                "ridge_support_backward": int(back),
            })

        # Non-maximum suppression in coordinate space, retaining best-supported minutia.
        raw.sort(key=lambda p: (p["quality"], p["ridge_support_forward"] + p["ridge_support_backward"]), reverse=True)
        out = []
        for p in raw:
            if all((p["x"] - q["x"]) ** 2 + (p["y"] - q["y"]) ** 2 >= self.min_distance ** 2 for q in out):
                out.append(p)
        return out

    def _singular_points(self, ori, mask):
        h, w = ori.shape
        bs = self.block_size
        # Poincare-index approximation over a circular neighborhood on the orientation field.
        points = []
        for y in range(bs, h-bs, bs):
            for x in range(bs, w-bs, bs):
                if mask[y, x] == 0:
                    continue
                ring = []
                for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                    yy = int(round(y + 10 * math.sin(ang)))
                    xx = int(round(x + 10 * math.cos(ang)))
                    a = float(ori[yy, xx])
                    ring.append(a)
                diffs = []
                for i in range(len(ring)):
                    d = 0.5 * math.atan2(math.sin(2*(ring[(i+1)%len(ring)]-ring[i])),
                                         math.cos(2*(ring[(i+1)%len(ring)]-ring[i])))
                    diffs.append(d)
                idx = float(sum(diffs) / math.pi)
                if abs(idx) >= 0.5:
                    typ = "core_candidate" if idx > 0 else "delta_candidate"
                    points.append({"x": x, "y": y, "type": typ, "poincare_index": idx})
        return points

    def _orientation_stats(self, ori, coherence, mask):
        valid = mask > 0
        angles = ori[valid]
        coh = coherence[valid]
        if angles.size == 0:
            return {
                "mean_orientation": 0.0,
                "orientation_std": 0.0,
                "coherence_mean": 0.0,
                "coherence_p10": 0.0,
                "coherence_p50": 0.0,
                "coherence_p90": 0.0,
            }
        # Double-angle circular statistics because ridge orientation is pi-periodic.
        z = np.exp(1j * 2 * angles)
        mean = 0.5 * math.atan2(float(np.mean(z).imag), float(np.mean(z).real))
        concentration = float(abs(np.mean(z)))
        return {
            "mean_orientation": float(mean),
            "orientation_concentration": concentration,
            "orientation_std": float(np.sqrt(max(0.0, -2.0 * math.log(max(concentration, 1e-6)))) / 2.0),
            "coherence_mean": float(np.mean(coh)),
            "coherence_p10": float(np.percentile(coh, 10)),
            "coherence_p50": float(np.percentile(coh, 50)),
            "coherence_p90": float(np.percentile(coh, 90)),
        }

    def process(self, image):
        n = self._normalize(image)
        q = self._quality(n)
        mask = self._segment(n)
        ori, freq, coherence = self._orientation_frequency(n, mask)
        enh = self._enhance(n, mask, ori, freq)
        sk = self._skeleton(enh, mask)
        pts = self._extract(sk, ori, mask, coherence)
        singular = self._singular_points(ori, mask)
        ostats = self._orientation_stats(ori, coherence, mask)

        foreground_ratio = float(np.mean(mask > 0))
        freq_valid = freq[(freq > 0) & (mask > 0)]
        q.update({
            "foreground_ratio": foreground_ratio,
            "frequency_coverage": float(np.mean((freq > 0) & (mask > 0))),
            "ridge_frequency_median": float(np.median(freq_valid)) if freq_valid.size else 0.0,
            "ridge_frequency_p10": float(np.percentile(freq_valid, 10)) if freq_valid.size else 0.0,
            "ridge_frequency_p90": float(np.percentile(freq_valid, 90)) if freq_valid.size else 0.0,
            "minutiae_density": float(len(pts) / max(int(np.sum(mask > 0)), 1) * 10000),
            "singular_points": len(singular),
            **ostats,
        })
        q["status"] = "usable" if foreground_ratio > 0.12 and q["coherence_p50"] > 0.15 and len(pts) >= 4 else "review"

        return {
            "enhanced": enh,
            "mask": mask,
            "skeleton": sk,
            "overlay": self._overlay(enh, sk, pts, singular),
            "minutiae": pts,
            "singular_points": singular,
            "orientation_field": ori,
            "ridge_frequency": freq,
            "counts": {
                "total": len(pts),
                "endings": sum(m["type"] == "ending" for m in pts),
                "bifurcations": sum(m["type"] == "bifurcation" for m in pts),
                "singular_points": len(singular),
            },
            "quality": q,
            "template": {
                "format": "starfish-minutiae",
                "version": 2,
                "image": {"width": int(n.shape[1]), "height": int(n.shape[0])},
                "quality": q,
                "minutiae": pts,
                "singular_points": singular,
                "ridge_frequency_summary": {
                    "median": q["ridge_frequency_median"],
                    "p10": q["ridge_frequency_p10"],
                    "p90": q["ridge_frequency_p90"],
                },
            },
        }

    def _overlay(self, base, sk, pts, singular):
        c = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        c[sk > 0] = (210, 210, 210)
        for m in pts:
            col = (0, 0, 255) if m["type"] == "ending" else (255, 0, 0)
            x, y, a = m["x"], m["y"], m["angle"]
            cv2.circle(c, (x, y), 4, col, 1, cv2.LINE_AA)
            cv2.line(c, (x, y), (x + int(10*math.cos(a)), y + int(10*math.sin(a))), col, 1, cv2.LINE_AA)
        for p in singular:
            col = (0, 215, 255) if p["type"] == "core_candidate" else (255, 0, 255)
            cv2.drawMarker(c, (p["x"], p["y"]), col, cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        return c
