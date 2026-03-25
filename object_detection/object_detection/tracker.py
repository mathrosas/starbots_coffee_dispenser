from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class Detection3D:
    position: np.ndarray
    radius_m: float
    score: float


@dataclass
class TrackState:
    track_id: int
    position: np.ndarray
    radius_m: float
    score: float
    stable_hits: int
    missed: int


@dataclass
class TrackOutput:
    track_id: int
    position: np.ndarray
    radius_m: float
    score: float
    confirmed: bool


class StableTracker:
    def __init__(
        self,
        max_ids: int = 4,
        match_distance_m: float = 0.06,
        ema_alpha: float = 0.45,
        min_confirm_frames: int = 3,
        max_missed_frames: int = 6,
    ) -> None:
        self.max_ids = max_ids
        self.match_distance_m = match_distance_m
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.min_confirm_frames = min_confirm_frames
        self.max_missed_frames = max_missed_frames
        self._tracks: Dict[int, TrackState] = {}

    def _next_free_track_id(self) -> int:
        for idx in range(self.max_ids):
            if idx not in self._tracks:
                return idx
        return -1

    def _purge_stale(self) -> None:
        stale = [tid for tid, tr in self._tracks.items() if tr.missed > self.max_missed_frames]
        for tid in stale:
            del self._tracks[tid]

    def update(self, detections: List[Detection3D]) -> List[TrackOutput]:
        outputs: List[TrackOutput] = []

        if not detections:
            for tr in self._tracks.values():
                tr.missed += 1
            self._purge_stale()
            return outputs

        track_ids = list(self._tracks.keys())
        unmatched_track_ids = set(track_ids)
        unmatched_det_ids = set(range(len(detections)))
        assignments: List[tuple[int, int, float]] = []

        # Greedy nearest-neighbor assignment.
        for det_idx, det in enumerate(detections):
            best_tid = -1
            best_dist = float("inf")
            for tid in list(unmatched_track_ids):
                tr = self._tracks[tid]
                dist = float(np.linalg.norm(det.position - tr.position))
                if dist < self.match_distance_m and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid >= 0:
                assignments.append((det_idx, best_tid, best_dist))
                unmatched_track_ids.remove(best_tid)
                unmatched_det_ids.remove(det_idx)

        # Update matched tracks with EMA.
        for det_idx, tid, _ in assignments:
            det = detections[det_idx]
            tr = self._tracks[tid]
            tr.position = self.ema_alpha * det.position + (1.0 - self.ema_alpha) * tr.position
            tr.radius_m = self.ema_alpha * det.radius_m + (1.0 - self.ema_alpha) * tr.radius_m
            tr.score = self.ema_alpha * det.score + (1.0 - self.ema_alpha) * tr.score
            tr.stable_hits += 1
            tr.missed = 0

            outputs.append(
                TrackOutput(
                    track_id=tid,
                    position=tr.position.copy(),
                    radius_m=float(tr.radius_m),
                    score=float(tr.score),
                    confirmed=tr.stable_hits >= self.min_confirm_frames,
                )
            )

        # Create tracks for unmatched detections.
        for det_idx in sorted(unmatched_det_ids):
            tid = self._next_free_track_id()
            if tid < 0:
                continue
            det = detections[det_idx]
            self._tracks[tid] = TrackState(
                track_id=tid,
                position=det.position.copy(),
                radius_m=float(det.radius_m),
                score=float(det.score),
                stable_hits=1,
                missed=0,
            )
            outputs.append(
                TrackOutput(
                    track_id=tid,
                    position=det.position.copy(),
                    radius_m=float(det.radius_m),
                    score=float(det.score),
                    confirmed=False,
                )
            )

        # Mark unmatched tracks as missed.
        for tid in unmatched_track_ids:
            self._tracks[tid].missed += 1

        self._purge_stale()
        return outputs

    def confirmed_tracks(self) -> List[TrackOutput]:
        out: List[TrackOutput] = []
        for tid in sorted(self._tracks.keys()):
            tr = self._tracks[tid]
            if tr.stable_hits >= self.min_confirm_frames and tr.missed == 0:
                out.append(
                    TrackOutput(
                        track_id=tid,
                        position=tr.position.copy(),
                        radius_m=float(tr.radius_m),
                        score=float(tr.score),
                        confirmed=True,
                    )
                )
        return out
