"""FastAPI gate workflow service.

Endpoints:
  POST /visits            submit an image -> detected fields, confidences,
                           validation status, review flag. Creates a Visit,
                           persists Detections, writes an AuditEvent.
  GET  /visits             list visits
  GET  /visits/{id}        get one visit with its fields
  GET  /review-queue       fields across all visits still needing review
  POST /visits/{id}/corrections   operator submits a correction; persisted
                                   + audit-logged; becomes retraining data
  GET  /visits/{id}/audit  full audit trail for one visit
  GET  /export/corrections export all corrections as a labeled dataset (JSON)
"""
import os
import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models, schemas
from .db import Base, SessionLocal, engine, get_db

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", Path(__file__).resolve().parents[1] / "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Dwarpal Gate Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from .pipeline import GatePipeline

        _pipeline = GatePipeline()
    return _pipeline


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/visits", response_model=schemas.VisitOut)
async def create_visit(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(400, "could not decode image")

    fname = f"{uuid.uuid4()}_{file.filename}"
    dest = UPLOAD_DIR / fname
    dest.write_bytes(contents)

    pipeline = get_pipeline()
    result = pipeline.run(image_bgr)

    visit = models.Visit(image_path=str(dest), status="pending_review")
    db.add(visit)
    db.flush()

    any_review = False
    for f in result.fields:
        if f.needs_review:
            any_review = True
        det = models.Detection(
            visit_id=visit.id,
            field_class=f.field_class,
            bbox_xywh=f.bbox_xywh,
            raw_text=f.raw_text,
            normalized_text=f.normalized_text,
            ocr_engine=f.ocr_engine,
            confidence=f.confidence,
            is_valid_format=f.is_valid_format,
            format_note=f.format_note,
            needs_review=f.needs_review,
        )
        db.add(det)

    visit.status = "pending_review" if any_review else "auto_accepted"

    db.add(
        models.AuditEvent(
            visit_id=visit.id,
            actor="system",
            action="auto_ocr",
            detail={"vehicle_boxes": result.vehicle_boxes, "n_fields": len(result.fields), "status": visit.status},
        )
    )
    db.commit()
    db.refresh(visit)

    return _visit_to_out(visit, result.vehicle_boxes, result.fields)


def _visit_to_out(visit: models.Visit, vehicle_boxes=None, fields_override=None) -> schemas.VisitOut:
    if fields_override is not None:
        fields = [
            schemas.FieldOut(
                field_class=f.field_class,
                bbox_xywh=f.bbox_xywh,
                raw_text=f.raw_text,
                normalized_text=f.normalized_text,
                ocr_engine=f.ocr_engine,
                ocr_candidates=f.ocr_candidates,
                confidence=f.confidence,
                is_valid_format=f.is_valid_format,
                format_note=f.format_note,
                needs_review=f.needs_review,
            )
            for f in fields_override
        ]
    else:
        fields = [
            schemas.FieldOut(
                id=d.id,
                field_class=d.field_class,
                bbox_xywh=d.bbox_xywh,
                raw_text=d.raw_text or "",
                normalized_text=d.normalized_text or "",
                ocr_engine=d.ocr_engine or "",
                ocr_candidates=[],
                confidence=d.confidence,
                is_valid_format=d.is_valid_format,
                format_note=d.format_note or "",
                needs_review=d.needs_review,
            )
            for d in visit.detections
        ]
    return schemas.VisitOut(
        id=visit.id,
        created_at=visit.created_at,
        image_path=visit.image_path,
        status=visit.status,
        vehicle_boxes=vehicle_boxes or [],
        fields=fields,
    )


@app.get("/visits", response_model=List[schemas.VisitOut])
def list_visits(db: Session = Depends(get_db)):
    visits = db.scalars(select(models.Visit).order_by(models.Visit.created_at.desc())).all()
    return [_visit_to_out(v) for v in visits]


@app.get("/visits/{visit_id}", response_model=schemas.VisitOut)
def get_visit(visit_id: str, db: Session = Depends(get_db)):
    visit = db.get(models.Visit, visit_id)
    if visit is None:
        raise HTTPException(404, "visit not found")
    return _visit_to_out(visit)


@app.get("/review-queue", response_model=List[schemas.ReviewQueueItem])
def review_queue(db: Session = Depends(get_db)):
    dets = db.scalars(select(models.Detection).where(models.Detection.needs_review == True)).all()  # noqa: E712
    items = []
    for d in dets:
        visit = db.get(models.Visit, d.visit_id)
        items.append(
            schemas.ReviewQueueItem(
                visit_id=d.visit_id,
                created_at=visit.created_at,
                image_path=visit.image_path,
                field=schemas.FieldOut(
                    id=d.id,
                    field_class=d.field_class,
                    bbox_xywh=d.bbox_xywh,
                    raw_text=d.raw_text or "",
                    normalized_text=d.normalized_text or "",
                    ocr_engine=d.ocr_engine or "",
                    ocr_candidates=[],
                    confidence=d.confidence,
                    is_valid_format=d.is_valid_format,
                    format_note=d.format_note or "",
                    needs_review=d.needs_review,
                ),
            )
        )
    return items


@app.post("/visits/{visit_id}/corrections", response_model=schemas.CorrectionOut)
def submit_correction(visit_id: str, correction: schemas.CorrectionIn, db: Session = Depends(get_db)):
    visit = db.get(models.Visit, visit_id)
    if visit is None:
        raise HTTPException(404, "visit not found")

    original_text = None
    det = None
    if correction.detection_id:
        det = db.get(models.Detection, correction.detection_id)
        if det:
            original_text = det.normalized_text
            det.normalized_text = correction.corrected_text
            det.needs_review = False

    corr = models.Correction(
        visit_id=visit_id,
        detection_id=correction.detection_id,
        field_class=correction.field_class,
        original_text=original_text,
        corrected_text=correction.corrected_text,
        corrected_by=correction.corrected_by,
    )
    db.add(corr)

    still_needs_review = any(d.needs_review for d in visit.detections if d.id != (det.id if det else None))
    visit.status = "pending_review" if still_needs_review else "reviewed"

    db.add(
        models.AuditEvent(
            visit_id=visit_id,
            actor=f"operator:{correction.corrected_by}",
            action="field_corrected",
            detail={"field_class": correction.field_class, "original": original_text, "corrected": correction.corrected_text},
        )
    )
    db.commit()
    db.refresh(corr)
    return corr


@app.get("/visits/{visit_id}/audit", response_model=List[schemas.AuditEventOut])
def get_audit_trail(visit_id: str, db: Session = Depends(get_db)):
    events = db.scalars(
        select(models.AuditEvent).where(models.AuditEvent.visit_id == visit_id).order_by(models.AuditEvent.created_at)
    ).all()
    return events


@app.get("/export/corrections")
def export_corrections(db: Session = Depends(get_db)):
    corrections = db.scalars(select(models.Correction)).all()
    return [
        {
            "visit_id": c.visit_id,
            "field_class": c.field_class,
            "original_text": c.original_text,
            "corrected_text": c.corrected_text,
            "corrected_by": c.corrected_by,
            "created_at": c.created_at.isoformat(),
        }
        for c in corrections
    ]
