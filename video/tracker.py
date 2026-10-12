"""Greedy IoU multi-object tracker (no external tracking library)."""
from __future__ import annotations

from dataclasses import dataclass, field


def iou(a, b) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    box: list
    last_frame: int


@dataclass
class IouTracker:
    iou_threshold: float = 0.3
    max_gap: int = 3  # sampled frames a track may go unseen before it is dropped
    _tracks: list = field(default_factory=list)
    _next_id: int = 1

    def update(self, frame_idx: int, boxes: list) -> list[int]:
        """Assign a track id to each box (same order). Highest-IoU pairs match first."""
        self._tracks = [t for t in self._tracks if frame_idx - t.last_frame <= self.max_gap]
        pairs = sorted(
            ((iou(t.box, b), ti, bi) for ti, t in enumerate(self._tracks) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        ids: list = [None] * len(boxes)
        used_t: set = set()
        for score, ti, bi in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or ids[bi] is not None:
                continue
            used_t.add(ti)
            t = self._tracks[ti]
            t.box, t.last_frame = list(boxes[bi]), frame_idx
            ids[bi] = t.track_id
        for bi, b in enumerate(boxes):
            if ids[bi] is None:
                self._tracks.append(_Track(self._next_id, list(b), frame_idx))
                ids[bi] = self._next_id
                self._next_id += 1
        return ids
