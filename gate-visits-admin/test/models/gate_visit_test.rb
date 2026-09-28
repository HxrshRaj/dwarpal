require "test_helper"

class FakeApiClient
  def initialize(visits: [], visit: nil)
    @visits = visits
    @visit = visit
  end

  def list_visits
    @visits
  end

  def get_visit(_id)
    @visit
  end
end

class GateVisitTest < ActiveSupport::TestCase
  SAMPLE_VISIT = {
    "id" => "abc123",
    "created_at" => "2026-09-28T12:00:00Z",
    "image_path" => "/uploads/abc123_truck.jpg",
    "status" => "pending_review",
    "vehicle_boxes" => [ { "label" => "truck", "bbox_xywh" => [ 1, 2, 3, 4 ], "score" => 0.9 } ],
    "fields" => [
      {
        "id" => "det-1",
        "field_class" => "usdot",
        "bbox_xywh" => [ 5, 6, 7, 8 ],
        "raw_text" => "USDOT 1234567",
        "normalized_text" => "1234567",
        "ocr_engine" => "easyocr",
        "confidence" => 0.42,
        "is_valid_format" => true,
        "format_note" => "ok",
        "needs_review" => true
      }
    ]
  }.freeze

  test "all wraps each visit from the API client" do
    client = FakeApiClient.new(visits: [ SAMPLE_VISIT ])
    visits = GateVisit.all(client: client)

    assert_equal 1, visits.size
    assert_equal "abc123", visits.first.id
    assert_equal "pending_review", visits.first.status
  end

  test "find wraps a single visit and its fields" do
    client = FakeApiClient.new(visit: SAMPLE_VISIT)
    visit = GateVisit.find("abc123", client: client)

    assert_equal "abc123", visit.id
    assert_equal 1, visit.fields.size

    field = visit.fields.first
    assert_equal "usdot", field.field_class
    assert_equal "1234567", field.normalized_text
    assert_equal true, field.needs_review
  end

  test "needs_review? reflects the visit status" do
    pending = GateVisit.new(SAMPLE_VISIT)
    assert pending.needs_review?

    reviewed = GateVisit.new(SAMPLE_VISIT.merge("status" => "reviewed"))
    assert_not reviewed.needs_review?
  end

  test "handles a visit with no fields" do
    visit = GateVisit.new(SAMPLE_VISIT.merge("fields" => []))
    assert_empty visit.fields
  end
end
