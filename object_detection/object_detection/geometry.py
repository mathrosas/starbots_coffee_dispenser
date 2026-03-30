from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

def estimate_depth_for_circle(
    depth_m: np.ndarray,
    u: int,
    v: int,
    radius_px: float,
    min_depth_m: float,
    max_depth_m: float,
    min_valid_samples: int = 20,
) -> Optional[Tuple[float, float]]:
    h, w = depth_m.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None

    outer = max(3, int(round(radius_px * 0.95)))
    inner = max(2, int(round(radius_px * 0.45)))

    y0 = max(0, v - outer)
    y1 = min(h - 1, v + outer)
    x0 = max(0, u - outer)
    x1 = min(w - 1, u + outer)

    yy, xx = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    dx = xx - u
    dy = yy - v
    d2 = dx * dx + dy * dy

    ring_mask = (d2 <= (outer * outer)) & (d2 >= (inner * inner))
    inner_mask = d2 < (inner * inner)

    patch = depth_m[y0 : y1 + 1, x0 : x1 + 1]
    ring_vals = patch[ring_mask]
    inner_vals = patch[inner_mask]

    ring_vals = ring_vals[np.isfinite(ring_vals)]
    inner_vals = inner_vals[np.isfinite(inner_vals)]

    ring_vals = ring_vals[(ring_vals >= min_depth_m) & (ring_vals <= max_depth_m)]
    inner_vals = inner_vals[(inner_vals >= min_depth_m) & (inner_vals <= max_depth_m)]

    if ring_vals.size < min_valid_samples and inner_vals.size < min_valid_samples:
        return None

    if ring_vals.size >= min_valid_samples:
        z = float(np.median(ring_vals))
        valid_ratio = float(ring_vals.size / max(1, int(np.sum(ring_mask))))
    else:
        z = float(np.median(inner_vals))
        valid_ratio = float(inner_vals.size / max(1, int(np.sum(inner_mask))))

    depth_conf = float(np.clip(valid_ratio * 1.5, 0.0, 1.0))
    return z, depth_conf


def project_pixel_to_3d(u: int, v: int, z: float, k: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if z <= 0.0:
        return None

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if fx == 0.0 or fy == 0.0:
        return None

    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return x, y, z
