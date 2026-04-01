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
    red_guided: bool = True
    red_hue_max_1: int = 12
    red_hue_min_2: int = 165
    red_sat_min: int = 90
    red_val_min: int = 60
    red_min_contour_area: float = 120.0
    red_max_contour_area: float = 2600.0
    red_morph_kernel_size: int = 5
    white_circumference_refine: bool = True
    outer_radius_min_scale: float = 1.08
    outer_radius_max_scale: float = 1.65
    outer_radius_step_px: float = 0.5
    outer_radius_samples: int = 72
    outer_radius_min_valid_ratio: float = 0.70
    outer_radius_min_score: float = 4.0
    outer_center_search_px: int = 6
    outer_center_step_px: float = 1.0
    outer_center_move_penalty: float = 0.08


class CupholderPerception:
    def __init__(self, cfg: PerceptionConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    def _clip_radius(self, radius: float) -> float:
        return float(
            np.clip(
                float(radius),
                float(self.cfg.min_radius_px),
                float(self.cfg.max_radius_px) * 1.35,
            )
        )

    @staticmethod
    def _build_edge_map(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        return cv2.GaussianBlur(grad, (3, 3), 0)

    def _ring_score(
        self,
        edge_map: np.ndarray,
        u: float,
        v: float,
        r: float,
        cos_t: np.ndarray,
        sin_t: np.ndarray,
    ) -> float:
        h, w = edge_map.shape[:2]
        xs = np.rint(u + r * cos_t).astype(np.int32)
        ys = np.rint(v + r * sin_t).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        valid_count = int(np.count_nonzero(valid))
        if valid_count <= 0:
            return -1.0

        valid_ratio = float(valid_count) / float(xs.size)
        if valid_ratio < float(self.cfg.outer_radius_min_valid_ratio):
            return -1.0

        vals = edge_map[ys[valid], xs[valid]]
        return float(np.mean(vals))

    def _refine_circle_from_white_circumference(
        self, edge_map: np.ndarray, seed_u: int, seed_v: int, seed_radius: float
    ) -> tuple[int, int, float]:
        """
        Refine both center and radius by maximizing radial edge response on the
        outer white circumference.
        """
        fallback_r = self._clip_radius(seed_radius)
        if not self.cfg.white_circumference_refine:
            return int(seed_u), int(seed_v), fallback_r

        r_min = float(seed_radius) * float(self.cfg.outer_radius_min_scale)
        r_max = float(seed_radius) * float(self.cfg.outer_radius_max_scale)
        if r_max <= r_min + 0.5:
            return int(seed_u), int(seed_v), fallback_r

        h, w = edge_map.shape[:2]
        if seed_u < 0 or seed_v < 0 or seed_u >= w or seed_v >= h:
            return int(seed_u), int(seed_v), fallback_r

        angle_count = max(24, int(self.cfg.outer_radius_samples))
        angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
        cos_t = np.cos(angles)
        sin_t = np.sin(angles)

        step_r = max(0.25, float(self.cfg.outer_radius_step_px))
        step_c = max(1.0, float(self.cfg.outer_center_step_px))
        max_shift = max(0, int(self.cfg.outer_center_search_px))
        offsets = np.arange(-max_shift, max_shift + 1e-6, step_c)

        best_adj_score = -1.0
        best_raw_score = -1.0
        best_u = float(seed_u)
        best_v = float(seed_v)
        best_r = fallback_r

        for du in offsets:
            u = float(seed_u) + float(du)
            if u < 0.0 or u >= float(w):
                continue
            for dv in offsets:
                v = float(seed_v) + float(dv)
                if v < 0.0 or v >= float(h):
                    continue

                shift_penalty = float(self.cfg.outer_center_move_penalty) * float(
                    np.hypot(float(du), float(dv))
                )

                for r in np.arange(r_min, r_max + step_r, step_r):
                    raw_score = self._ring_score(edge_map, u, v, float(r), cos_t, sin_t)
                    if raw_score < 0.0:
                        continue
                    adj_score = raw_score - shift_penalty
                    if adj_score > best_adj_score:
                        best_adj_score = adj_score
                        best_raw_score = raw_score
                        best_u = u
                        best_v = v
                        best_r = float(r)

        if best_raw_score < float(self.cfg.outer_radius_min_score):
            return int(seed_u), int(seed_v), fallback_r

        clipped_r = self._clip_radius(best_r)
        return int(round(best_u)), int(round(best_v)), clipped_r

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

    @staticmethod
    def _ring_ratio(binary: np.ndarray, u: int, v: int, r: float) -> float:
        r_i = max(2, int(round(r * 0.68)))
        r_o = max(r_i + 2, int(round(r * 1.15)))

        h, w = binary.shape[:2]
        y0 = max(0, v - r_o)
        y1 = min(h - 1, v + r_o)
        x0 = max(0, u - r_o)
        x1 = min(w - 1, u + r_o)

        yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        dx = xx - u
        dy = yy - v
        d2 = dx * dx + dy * dy
        ring_mask = (d2 <= (r_o * r_o)) & (d2 >= (r_i * r_i))

        ring_vals = binary[y0 : y1 + 1, x0 : x1 + 1][ring_mask]
        if ring_vals.size < 30:
            return 0.0

        ratio = float(np.count_nonzero(ring_vals)) / float(ring_vals.size)
        return float(np.clip(ratio, 0.0, 1.0))

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

    def _contour_candidates(
        self, proc_gray: np.ndarray, binary: np.ndarray, edge_map: np.ndarray
    ) -> List[Candidate2D]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Candidate2D] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_contour_area or area > self.cfg.max_contour_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue

            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.cfg.min_circularity:
                continue

            (u, v), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.cfg.min_radius_px or radius > self.cfg.max_radius_px:
                continue

            u_i = int(round(u))
            v_i = int(round(v))
            u_ref, v_ref, r_ref = self._refine_circle_from_white_circumference(
                edge_map, u_i, v_i, radius
            )
            contrast = self._circle_contrast(proc_gray, u_ref, v_ref, r_ref)
            score = 0.70 * circularity + 0.30 * contrast

            candidates.append(
                Candidate2D(
                    u=u_ref,
                    v=v_ref,
                    radius_px=float(r_ref),
                    score=float(np.clip(score, 0.0, 1.0)),
                )
            )

        return candidates

    def _red_candidates(
        self, bgr_image: np.ndarray, proc_gray: np.ndarray, edge_map: np.ndarray
    ) -> List[Candidate2D]:
        roi_w = int(bgr_image.shape[1] * self.cfg.roi_width_ratio)
        bgr_roi = bgr_image[:, :roi_w]
        hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(
            hsv,
            (0, int(self.cfg.red_sat_min), int(self.cfg.red_val_min)),
            (int(self.cfg.red_hue_max_1), 255, 255),
        )
        mask2 = cv2.inRange(
            hsv,
            (int(self.cfg.red_hue_min_2), int(self.cfg.red_sat_min), int(self.cfg.red_val_min)),
            (179, 255, 255),
        )
        red_mask = cv2.bitwise_or(mask1, mask2)

        k = self._odd(max(1, int(self.cfg.red_morph_kernel_size)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Candidate2D] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.red_min_contour_area or area > self.cfg.red_max_contour_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 1e-6:
                continue

            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < (0.60 * self.cfg.min_circularity):
                continue

            (u, v), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.cfg.min_radius_px or radius > self.cfg.max_radius_px:
                continue

            u_i = int(round(u))
            v_i = int(round(v))
            ring_ratio = self._ring_ratio(red_mask, u_i, v_i, radius)
            u_ref, v_ref, r_ref = self._refine_circle_from_white_circumference(
                edge_map, u_i, v_i, radius
            )
            contrast = self._circle_contrast(proc_gray, u_ref, v_ref, r_ref)
            score = (
                0.45 * float(np.clip(circularity, 0.0, 1.0))
                + 0.40 * ring_ratio
                + 0.15 * contrast
            )

            candidates.append(
                Candidate2D(
                    u=u_ref,
                    v=v_ref,
                    radius_px=float(r_ref),
                    score=float(np.clip(score, 0.0, 1.0)),
                )
            )

        return candidates

    def _hough_candidates(self, proc_gray: np.ndarray, edge_map: np.ndarray) -> List[Candidate2D]:
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
            u_ref, v_ref, r_ref = self._refine_circle_from_white_circumference(
                edge_map, u, v, radius
            )
            contrast = self._circle_contrast(proc_gray, u_ref, v_ref, r_ref)
            score = 0.55 + 0.45 * contrast
            out.append(
                Candidate2D(
                    u=u_ref,
                    v=v_ref,
                    radius_px=r_ref,
                    score=float(np.clip(score, 0.0, 1.0)),
                )
            )
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
        edge_map = self._build_edge_map(proc_gray)

        candidates: List[Candidate2D] = []
        if self.cfg.red_guided:
            candidates.extend(self._red_candidates(bgr_image, proc_gray, edge_map))
        candidates.extend(self._contour_candidates(proc_gray, binary, edge_map))
        if self.cfg.hough_fallback:
            candidates.extend(self._hough_candidates(proc_gray, edge_map))

        candidates = self._nms(candidates)
        return candidates, binary
