from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Candidate2D:
    u: int
    v: int
    radius_px: float
    score: float


@dataclass
class PerceptionConfig:
    roi_width_ratio: float = 0.75
    min_radius_px: float = 10.0
    max_radius_px: float = 20.0
    min_contour_area: float = 120.0
    max_contour_area: float = 2200.0
    min_circularity: float = 0.58
    max_candidates: int = 4
    min_center_distance_px: float = 18.0
    adaptive_block_size: int = 31
    adaptive_c: int = 5
    morph_kernel_size: int = 3
    gauss_kernel_size: int = 5
    clahe_clip_limit: float = 2.0
    hough_fallback: bool = True
    hough_dp: float = 1.6
    hough_min_dist: float = 20.0
    hough_param1: float = 162.0
    hough_param2: float = 30.0


class CupholderPerception:
    def __init__(self, cfg: PerceptionConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _circle_contrast(gray: np.ndarray, u: int, v: int, r: float) -> float:
        r_i = max(2, int(round(r * 0.50)))
        r_o = max(r_i + 2, int(round(r * 1.10)))

        h, w = gray.shape[:2]
        y0 = max(0, v - r_o)
        y1 = min(h - 1, v + r_o)
        x0 = max(0, u - r_o)
        x1 = min(w - 1, u + r_o)

        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        dx = xx - u
        dy = yy - v
        d2 = dx * dx + dy * dy

        inner_mask = d2 <= (r_i * r_i)
        ring_mask = (d2 <= (r_o * r_o)) & (~inner_mask)

        inner_vals = gray[y0 : y1 + 1, x0 : x1 + 1][inner_mask]
        ring_vals = gray[y0 : y1 + 1, x0 : x1 + 1][ring_mask]

        if inner_vals.size < 10 or ring_vals.size < 20:
            return 0.0

        inner_mean = float(np.mean(inner_vals))
        ring_mean = float(np.mean(ring_vals))
        contrast = (ring_mean - inner_mean) / 255.0
        return float(np.clip(contrast, 0.0, 1.0))

    def _preprocess(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        roi_w = int(gray.shape[1] * self.cfg.roi_width_ratio)
        proc = gray[:, :roi_w]

        clahe = cv2.createCLAHE(clipLimit=self.cfg.clahe_clip_limit, tileGridSize=(8, 8))
        proc = clahe.apply(proc)

        gk = self._odd(max(1, int(self.cfg.gauss_kernel_size)))
        proc_blur = cv2.GaussianBlur(proc, (gk, gk), 0) if gk > 1 else proc

        block = self._odd(max(3, int(self.cfg.adaptive_block_size)))
        binary = cv2.adaptiveThreshold(
            proc_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            int(self.cfg.adaptive_c),
        )

        k = max(1, int(self.cfg.morph_kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return proc_blur, binary

    def _contour_candidates(self, proc_gray: np.ndarray, binary: np.ndarray) -> List[Candidate2D]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Candidate2D] = []

        # print(f"[DEBUG] Total contours found: {len(contours)}") # Debug for cupholder detection

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_contour_area or area > self.cfg.max_contour_area:
                if area > 50:  # only log non-tiny contours
                    (u, v), radius = cv2.minEnclosingCircle(cnt)
                    # print(f"[DEBUG] REJECTED area: u={int(u)} v={int(v)} area={area:.0f} bounds=[{self.cfg.min_contour_area}, {self.cfg.max_contour_area}]") # Debug for cupholder detection
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue

            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.cfg.min_circularity:
                (u_dbg, v_dbg), _ = cv2.minEnclosingCircle(cnt)
                # print(f"[DEBUG] REJECTED circularity: u={int(u_dbg)} v={int(v_dbg)} area={area:.0f} circ={circularity:.3f} min={self.cfg.min_circularity}") # Debug for cupholder detection
                # Ellipse fallback: accept elongated cupholders seen at an angle
                if len(cnt) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(cnt)
                        (eu, ev), (minor_ax, major_ax), _angle = ellipse
                        if minor_ax > 1e-3:
                            aspect = major_ax / minor_ax
                            equiv_r = (minor_ax + major_ax) / 4.0
                            # print(f"[DEBUG] ELLIPSE: u={int(round(eu))} v={int(round(ev))} aspect={aspect:.2f} equiv_r={equiv_r:.1f} pass={aspect < 2.5 and equiv_r >= self.cfg.min_radius_px * 0.5}") # Debug for cupholder detection
                            if aspect < 2.5 and equiv_r >= self.cfg.min_radius_px * 0.5:
                                contrast = self._circle_contrast(
                                    proc_gray,
                                    int(round(eu)),
                                    int(round(ev)),
                                    equiv_r,
                                )
                                shape_q = 1.0 / aspect
                                _score = 0.50 * shape_q + 0.30 * contrast
                                candidates.append(
                                    Candidate2D(
                                        u=int(round(eu)),
                                        v=int(round(ev)),
                                        radius_px=float(equiv_r),
                                        score=float(np.clip(_score, 0.0, 1.0)),
                                    )
                                )
                    except cv2.error:
                        pass
                continue

            (u, v), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.cfg.min_radius_px or radius > self.cfg.max_radius_px:
                # print(f"[DEBUG] REJECTED radius: u={int(u)} v={int(v)} r={radius:.1f} bounds=[{self.cfg.min_radius_px}, {self.cfg.max_radius_px}]") # Debug for cupholder detection
                continue

            contrast = self._circle_contrast(proc_gray, int(round(u)), int(round(v)), radius)
            score = 0.70 * circularity + 0.30 * contrast

            candidates.append(
                Candidate2D(
                    u=int(round(u)),
                    v=int(round(v)),
                    radius_px=float(radius),
                    score=float(np.clip(score, 0.0, 1.0)),
                )
            )
            # print(f"[DEBUG] ACCEPTED: u={int(round(u))} v={int(round(v))} r={radius:.1f} area={area:.0f} circ={circularity:.3f} score={score:.3f}") # Debug for cupholder detection

        return candidates

    def _hough_candidates(self, proc_gray: np.ndarray) -> List[Candidate2D]:
        circles = cv2.HoughCircles(
            proc_gray,
            cv2.HOUGH_GRADIENT,
            dp=self.cfg.hough_dp,
            minDist=self.cfg.hough_min_dist,
            param1=self.cfg.hough_param1,
            param2=self.cfg.hough_param2,
            minRadius=int(self.cfg.min_radius_px),
            maxRadius=int(self.cfg.max_radius_px),
        )
        if circles is None:
            return []

        out: List[Candidate2D] = []
        for c in np.around(circles[0]).astype(float):
            u, v, radius = int(round(c[0])), int(round(c[1])), float(c[2])
            contrast = self._circle_contrast(proc_gray, u, v, radius)
            score = 0.55 + 0.45 * contrast
            out.append(Candidate2D(u=u, v=v, radius_px=radius, score=float(np.clip(score, 0.0, 1.0))))
        return out

    def _nms(self, candidates: List[Candidate2D]) -> List[Candidate2D]:
        selected: List[Candidate2D] = []
        for cand in sorted(candidates, key=lambda x: x.score, reverse=True):
            keep = True
            for prev in selected:
                if np.hypot(cand.u - prev.u, cand.v - prev.v) < self.cfg.min_center_distance_px:
                    keep = False
                    break
            if keep:
                selected.append(cand)
            if len(selected) >= self.cfg.max_candidates:
                break
        return selected

    def detect(self, bgr_image: np.ndarray) -> Tuple[List[Candidate2D], np.ndarray]:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        proc_gray, binary = self._preprocess(gray)

        candidates = self._contour_candidates(proc_gray, binary)
        if self.cfg.hough_fallback:
            candidates.extend(self._hough_candidates(proc_gray))

        candidates = self._nms(candidates)
        return candidates, binary
