"""Detection stage: locate ROIs for truck, plate, USDOT, trailer_id, seal.

Two detectors, both permissively licensed (see docs/data.md "Detector
license choice" for why we did not use ultralytics/YOLO, which is AGPL-3.0):

  1. VehicleDetector — torchvision's pretrained Faster R-CNN (COCO weights,
     BSD-3-Clause license). Used to localize the truck/vehicle itself.
     This is a genuine off-the-shelf pretrained baseline, not fine-tuned.

  2. TextRegionProposer — classical OpenCV (Apache-2.0) MSER-based region
     proposal. Used to propose candidate plate/USDOT/trailer_id regions
     inside (or without) a vehicle box. This is deliberately simple and
     its real recall/precision on our data is reported honestly in
     docs/benchmarks.md — it is NOT a trained detector, and its accuracy
     is expected to be well below a fine-tuned model. That gap is itself
     part of the feasibility answer (docs/feasibility.md).
"""
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
]
TRUCK_CATEGORY_ID = 8  # torchvision COCO category index for "truck"
BUS_CATEGORY_ID = 6
CAR_CATEGORY_ID = 3


@dataclass
class BoxDetection:
    label: str
    bbox_xywh: List[float]  # [x, y, w, h] in pixels
    score: float


class VehicleDetector:
    """torchvision Faster R-CNN MobileNetV3 (COCO-pretrained, BSD-3-Clause)."""

    def __init__(self, score_threshold: float = 0.5, device: str = "cpu"):
        import torch
        from torchvision.models.detection import (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            fasterrcnn_mobilenet_v3_large_320_fpn,
        )

        self.device = device
        self.score_threshold = score_threshold
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        self.model.eval().to(device)
        self.transforms = weights.transforms()
        self.torch = torch

    def detect(self, image_rgb: np.ndarray) -> List[BoxDetection]:
        tensor = self.transforms(self.torch.from_numpy(image_rgb).permute(2, 0, 1))
        with self.torch.no_grad():
            output = self.model([tensor.to(self.device)])[0]

        results = []
        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if score.item() < self.score_threshold:
                continue
            if label.item() not in (TRUCK_CATEGORY_ID, BUS_CATEGORY_ID, CAR_CATEGORY_ID):
                continue
            x1, y1, x2, y2 = box.tolist()
            name = {TRUCK_CATEGORY_ID: "truck", BUS_CATEGORY_ID: "bus", CAR_CATEGORY_ID: "car"}[label.item()]
            results.append(BoxDetection(label=name, bbox_xywh=[x1, y1, x2 - x1, y2 - y1], score=score.item()))
        return results


class TextRegionProposer:
    """Classical MSER-based candidate text-region proposer (OpenCV, Apache-2.0).

    Not a trained detector. Proposes rectangular regions likely to contain
    dense small text/characters (plates, USDOT stencils, trailer IDs) based
    on MSER blob stability and aspect ratio filtering. Every proposal is
    unlabeled as to which field class it is — that's decided downstream by
    OCR + validators finding which regions produce a format-valid string.
    """

    def __init__(self, min_area: int = 200, max_area_frac: float = 0.15):
        self.mser = cv2.MSER_create()
        self.min_area = min_area
        self.max_area_frac = max_area_frac

    def propose(self, image_bgr: np.ndarray) -> List[BoxDetection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        regions, _ = self.mser.detectRegions(gray)

        char_boxes = []
        for region in regions:
            x, y, bw, bh = cv2.boundingRect(region.reshape(-1, 1, 2))
            area = bw * bh
            if area < self.min_area or area > self.max_area_frac * h * w:
                continue
            aspect = bw / max(bh, 1)
            if aspect < 0.15 or aspect > 6:
                continue
            char_boxes.append([x, y, x + bw, y + bh])

        if not char_boxes:
            return []

        merged = self._merge_nearby(char_boxes, w, h)
        return [BoxDetection(label="text_region", bbox_xywh=[x, y, bw, bh], score=1.0) for (x, y, bw, bh) in merged]

    @staticmethod
    def _merge_nearby(boxes, img_w, img_h, x_gap_frac=0.03, y_gap_frac=0.02):
        """Cluster nearby character-sized boxes into line-level region proposals."""
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        x_gap = img_w * x_gap_frac
        y_gap = img_h * y_gap_frac
        clusters = []
        for box in boxes:
            placed = False
            for cluster in clusters:
                cx1, cy1, cx2, cy2 = cluster
                if box[0] <= cx2 + x_gap and box[1] <= cy2 + y_gap and box[3] >= cy1 - y_gap:
                    cluster[0] = min(cx1, box[0])
                    cluster[1] = min(cy1, box[1])
                    cluster[2] = max(cx2, box[2])
                    cluster[3] = max(cy2, box[3])
                    placed = True
                    break
            if not placed:
                clusters.append(list(box))
        return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in clusters]
