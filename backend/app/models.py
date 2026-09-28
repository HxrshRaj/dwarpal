import datetime
import uuid

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Visit(Base):
    """A simulated gate visit: one truck arrival."""

    __tablename__ = "visits"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    image_path: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending_review")  # pending_review | auto_accepted | reviewed

    detections: Mapped[list["Detection"]] = relationship(back_populates="visit", cascade="all, delete-orphan")
    corrections: Mapped[list["Correction"]] = relationship(back_populates="visit", cascade="all, delete-orphan")
    audit_events: Mapped[list["AuditEvent"]] = relationship(back_populates="visit", cascade="all, delete-orphan")


class Detection(Base):
    """One detected+OCR'd field on a visit (plate, usdot, trailer_id, seal)."""

    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    visit_id: Mapped[str] = mapped_column(ForeignKey("visits.id"))
    field_class: Mapped[str] = mapped_column(String)  # plate | usdot | trailer_id | seal
    bbox_xywh: Mapped[list] = mapped_column(JSON)
    raw_text: Mapped[str] = mapped_column(Text, nullable=True)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=True)
    ocr_engine: Mapped[str] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    is_valid_format: Mapped[bool] = mapped_column(Boolean, default=False)
    format_note: Mapped[str] = mapped_column(Text, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)

    visit: Mapped["Visit"] = relationship(back_populates="detections")


class Correction(Base):
    """An operator's correction to a detection — becomes a feedback dataset."""

    __tablename__ = "corrections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    visit_id: Mapped[str] = mapped_column(ForeignKey("visits.id"))
    detection_id: Mapped[str] = mapped_column(String, nullable=True)
    field_class: Mapped[str] = mapped_column(String)
    original_text: Mapped[str] = mapped_column(Text, nullable=True)
    corrected_text: Mapped[str] = mapped_column(Text)
    corrected_by: Mapped[str] = mapped_column(String, default="operator")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    visit: Mapped["Visit"] = relationship(back_populates="corrections")


class AuditEvent(Base):
    """Every automated decision and every human correction, logged."""

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    visit_id: Mapped[str] = mapped_column(ForeignKey("visits.id"))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    actor: Mapped[str] = mapped_column(String)  # "system" | "operator:<id>"
    action: Mapped[str] = mapped_column(String)  # e.g. "auto_ocr", "flagged_for_review", "field_corrected"
    detail: Mapped[dict] = mapped_column(JSON, nullable=True)

    visit: Mapped["Visit"] = relationship(back_populates="audit_events")
