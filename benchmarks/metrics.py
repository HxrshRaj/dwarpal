"""Metric implementations used by every benchmark script. Kept dependency-light
(no ML framework required) so metrics can be unit tested in isolation."""
from typing import List, Sequence, Tuple


def iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def average_precision_at_iou(
    predictions: List[Tuple[List[float], float]],  # [(bbox_xywh, score), ...]
    ground_truths: List[List[float]],  # [bbox_xywh, ...]
    iou_threshold: float = 0.5,
) -> float:
    """Single-image, single-class AP at a fixed IoU threshold via
    greedy score-sorted matching (standard PASCAL VOC-style AP)."""
    if not ground_truths:
        return 1.0 if not predictions else 0.0
    if not predictions:
        return 0.0

    preds_sorted = sorted(predictions, key=lambda p: -p[1])
    matched_gt = set()
    tp, fp = 0, 0
    precisions = []
    for bbox, _score in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
            iou = iou_xywh(bbox, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))

    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def mean_average_precision(per_image_ap: List[float]) -> float:
    if not per_image_ap:
        return 0.0
    return sum(per_image_ap) / len(per_image_ap)


def exact_match(pred_text: str, gt_text: str) -> bool:
    return pred_text.strip().upper() == gt_text.strip().upper()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def character_error_rate(pred_text: str, gt_text: str) -> float:
    gt = gt_text.strip().upper()
    pred = pred_text.strip().upper()
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)
