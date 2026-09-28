"""API contract tests. These exercise the FastAPI endpoints against a real
(sqlite) database but stub out the heavy CV pipeline (torch/easyocr model
loading) so the tests run fast and offline — the pipeline itself is
exercised separately in benchmarks/ against real images."""
import io

from PIL import Image


def _fake_image_bytes():
    img = Image.new("RGB", (64, 64), (120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def _install_fake_pipeline(monkeypatch, fields=None):
    import app.main as main_module
    from app.pipeline import FieldResult, PipelineResult

    fields = fields if fields is not None else [
        FieldResult(
            field_class="usdot",
            bbox_xywh=[1, 2, 3, 4],
            raw_text="USDOT 1234567",
            normalized_text="1234567",
            ocr_engine="fake",
            ocr_candidates=[],
            confidence=0.3,  # below default threshold -> needs review
            is_valid_format=True,
            format_note="ok",
            needs_review=True,
        )
    ]

    class FakePipeline:
        def run(self, image_bgr):
            return PipelineResult(vehicle_boxes=[], fields=fields)

    monkeypatch.setattr(main_module, "_pipeline", FakePipeline())
    monkeypatch.setattr(main_module, "get_pipeline", lambda: main_module._pipeline)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_visit_flags_low_confidence_for_review(client, monkeypatch):
    _install_fake_pipeline(monkeypatch)
    resp = client.post("/visits", files={"file": ("truck.png", _fake_image_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending_review"
    assert len(body["fields"]) == 1
    assert body["fields"][0]["needs_review"] is True
    assert body["fields"][0]["field_class"] == "usdot"


def test_create_visit_auto_accepts_high_confidence(client, monkeypatch):
    from app.pipeline import FieldResult

    high_conf_field = FieldResult(
        field_class="plate",
        bbox_xywh=[0, 0, 1, 1],
        raw_text="ABC123",
        normalized_text="ABC123",
        ocr_engine="fake",
        ocr_candidates=[],
        confidence=0.95,
        is_valid_format=True,
        format_note="ok",
        needs_review=False,
    )
    _install_fake_pipeline(monkeypatch, fields=[high_conf_field])
    resp = client.post("/visits", files={"file": ("truck.png", _fake_image_bytes(), "image/png")})
    assert resp.status_code == 200
    assert resp.json()["status"] == "auto_accepted"


def test_review_queue_lists_low_confidence_fields(client, monkeypatch):
    _install_fake_pipeline(monkeypatch)
    client.post("/visits", files={"file": ("truck.png", _fake_image_bytes(), "image/png")})
    resp = client.get("/review-queue")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["field"]["field_class"] == "usdot"


def test_correction_clears_review_flag_and_is_audited(client, monkeypatch):
    _install_fake_pipeline(monkeypatch)
    visit = client.post("/visits", files={"file": ("truck.png", _fake_image_bytes(), "image/png")}).json()
    detection_id = visit["fields"][0]["id"]

    resp = client.post(
        f"/visits/{visit['id']}/corrections",
        json={"detection_id": detection_id, "field_class": "usdot", "corrected_text": "7654321"},
    )
    assert resp.status_code == 200
    corr = resp.json()
    assert corr["corrected_text"] == "7654321"
    assert corr["original_text"] == "1234567"

    updated_visit = client.get(f"/visits/{visit['id']}").json()
    assert updated_visit["status"] == "reviewed"
    assert updated_visit["fields"][0]["needs_review"] is False

    audit = client.get(f"/visits/{visit['id']}/audit").json()
    actions = [e["action"] for e in audit]
    assert "auto_ocr" in actions
    assert "field_corrected" in actions


def test_export_corrections_returns_labeled_dataset(client, monkeypatch):
    _install_fake_pipeline(monkeypatch)
    visit = client.post("/visits", files={"file": ("truck.png", _fake_image_bytes(), "image/png")}).json()
    detection_id = visit["fields"][0]["id"]
    client.post(
        f"/visits/{visit['id']}/corrections",
        json={"detection_id": detection_id, "field_class": "usdot", "corrected_text": "7654321"},
    )
    resp = client.get("/export/corrections")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["corrected_text"] == "7654321"


def test_visit_not_found_returns_404(client):
    resp = client.get("/visits/does-not-exist")
    assert resp.status_code == 404
