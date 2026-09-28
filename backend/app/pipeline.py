"""End-to-end gate pipeline: image -> vehicle box -> candidate text regions
-> OCR (two engines) -> validation/normalization -> confidence gating.

This is the single code path used by both the FastAPI endpoint (Phase 5)
and the benchmark scripts (Phase 3), so a reported metric always reflects
exactly what the API does — no separate "demo" implementation.
"""
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.validators import DEFAULT_CONFIDENCE_THRESHOLDS, needs_human_review, validate_field

from .detection import TextRegionProposer, VehicleDetector
from .ocr import EasyOcrEngine, TesseractEngine, run_both_engines


@dataclass
class FieldResult:
    field_class: str
    bbox_xywh: list
    raw_text: str
    normalized_text: str
    ocr_engine: str
    ocr_candidates: list  # [{engine, text, confidence}, ...] for comparison
    confidence: float
    is_valid_format: bool
    format_note: str
    needs_review: bool
    id: Optional[str] = None  # populated by the API after DB insert; empty in benchmarks


@dataclass
class PipelineResult:
    vehicle_boxes: list
    fields: List[FieldResult]


class GatePipeline:
    def __init__(self, vehicle_score_threshold: float = 0.5, confidence_thresholds: Optional[dict] = None):
        self.vehicle_detector = VehicleDetector(score_threshold=vehicle_score_threshold)
        self.text_proposer = TextRegionProposer()
        self.easy_ocr = EasyOcrEngine()
        self.tesseract = TesseractEngine()
        self.confidence_thresholds = confidence_thresholds or DEFAULT_CONFIDENCE_THRESHOLDS

    def classify_field(self, text: str) -> str:
        """Heuristic field-class assignment from OCR text shape: try each
        validator and keep the first that reports a valid format; falls back
        to 'plate' (the most permissive shape) if none validate."""
        upper = text.strip().upper()
        if "USDOT" in upper or validate_field("usdot", upper).is_valid_format:
            return "usdot"
        if validate_field("trailer_id", upper).is_valid_format:
            return "trailer_id"
        return "plate"

    def run(self, image_bgr: np.ndarray) -> PipelineResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        vehicle_boxes = self.vehicle_detector.detect(image_rgb)
        text_regions = self.text_proposer.propose(image_bgr)

        fields = []
        h, w = image_bgr.shape[:2]
        for region in text_regions:
            x, y, bw, bh = [int(v) for v in region.bbox_xywh]
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            if x2 <= x or y2 <= y:
                continue
            crop = image_bgr[y:y2, x:x2]
            if crop.size == 0:
                continue

            ocr_results = run_both_engines(crop, self.easy_ocr, self.tesseract)
            candidates = [asdict(r) for r in ocr_results]
            best = max(ocr_results, key=lambda r: r.confidence)
            if not best.text.strip():
                continue

            field_class = self.classify_field(best.text)
            validation = validate_field(field_class, best.text)
            review = needs_human_review(field_class, best.confidence, self.confidence_thresholds)

            fields.append(
                FieldResult(
                    field_class=field_class,
                    bbox_xywh=[x, y, x2 - x, y2 - y],
                    raw_text=best.text,
                    normalized_text=validation.normalized_text,
                    ocr_engine=best.engine,
                    ocr_candidates=candidates,
                    confidence=best.confidence,
                    is_valid_format=validation.is_valid_format,
                    format_note=validation.format_note,
                    needs_review=review,
                )
            )

        return PipelineResult(
            vehicle_boxes=[asdict(b) for b in vehicle_boxes],
            fields=fields,
        )
