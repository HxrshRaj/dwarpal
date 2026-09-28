import datetime

from pydantic import BaseModel, ConfigDict


class OcrCandidate(BaseModel):
    engine: str
    text: str
    confidence: float


class FieldOut(BaseModel):
    id: str | None = None
    field_class: str
    bbox_xywh: list[float]
    raw_text: str
    normalized_text: str
    ocr_engine: str
    ocr_candidates: list[OcrCandidate]
    confidence: float
    is_valid_format: bool
    format_note: str
    needs_review: bool


class VehicleBoxOut(BaseModel):
    label: str
    bbox_xywh: list[float]
    score: float


class VisitOut(BaseModel):
    id: str
    created_at: datetime.datetime
    image_path: str
    status: str
    vehicle_boxes: list[VehicleBoxOut]
    fields: list[FieldOut]

    model_config = ConfigDict(from_attributes=True)


class CorrectionIn(BaseModel):
    detection_id: str | None = None
    field_class: str
    corrected_text: str
    corrected_by: str = "operator"


class CorrectionOut(BaseModel):
    id: str
    visit_id: str
    field_class: str
    original_text: str | None
    corrected_text: str
    corrected_by: str
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class ReviewQueueItem(BaseModel):
    visit_id: str
    created_at: datetime.datetime
    image_path: str
    field: FieldOut


class AuditEventOut(BaseModel):
    id: str
    visit_id: str
    created_at: datetime.datetime
    actor: str
    action: str
    detail: dict | None

    model_config = ConfigDict(from_attributes=True)
