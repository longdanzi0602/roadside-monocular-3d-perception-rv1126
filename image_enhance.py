#!/usr/bin/env python3
"""考核①: 图像预处理 — CLAHE暗光增强 + 白平衡 + 画质评分."""
import cv2, numpy as np
from dataclasses import dataclass

@dataclass
class ImageQuality:
    brightness: float; contrast: float; sharpness: float
    saturation: float; snr: float; score: float
    def to_dict(self):
        return {k:round(v,2) for k,v in self.__dict__.items()}

class ImageEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    def _measure_quality(self, img_bgr):
        h, w = img_bgr.shape[:2]
        # Downsample to width 480 for 16x faster quality calculation on ARM CPU
        scale_q = 480.0 / w
        img_small = cv2.resize(img_bgr, (480, int(h * scale_q)))
        
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        brightness = gray.mean()
        contrast = gray.std()
        sharpness = cv2.Laplacian(gray, cv2.CV_16S).var() * 6.0
        saturation = hsv[:,:,1].mean()
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        noise = (gray.astype(np.float32) - blurred.astype(np.float32)).std()
        snr = brightness / max(noise, 1.0)
        b_score = max(0, 100 - abs(brightness - 130) * 0.5) / 100
        c_score = min(1.0, contrast / 50)
        s_score = min(1.0, sharpness / 200)
        score = (b_score * 0.2 + c_score * 0.3 + s_score * 0.5) * 100
        return ImageQuality(brightness, contrast, sharpness, saturation, snr, score)

    def auto_white_balance(self, img_bgr, p=0.5):
        b, g, r = cv2.split(img_bgr.astype(np.float32))
        for ch in [b, g, r]:
            lo, hi = np.percentile(ch, [p, 100-p])
            ch[:] = np.clip((ch - lo) / max(hi - lo, 1.0) * 255, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    def enhance(self, img_bgr):
        q_before = self._measure_quality(img_bgr)
        balanced = self.auto_white_balance(img_bgr)
        lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l_eq, a, b_ch]), cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced, (0,0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.2, blurred, -0.2, 0)
        q_after = self._measure_quality(enhanced)
        return enhanced, q_before, q_after

    def is_too_dark(self, img_bgr, threshold=80):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean() < threshold
