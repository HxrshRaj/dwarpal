require "test_helper"
require "minitest/mock"

class FakeDwarpalApiClient
  SAMPLE_VISIT = {
    "id" => "abc123",
    "created_at" => "2026-09-28T12:00:00Z",
    "image_path" => "/uploads/abc123_truck.jpg",
    "status" => "pending_review",
    "vehicle_boxes" => [],
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

  attr_reader :last_correction

  def base_url
    "http://fake-api.test"
  end

  def list_visits
    [ SAMPLE_VISIT ]
  end

  def get_visit(_id)
    SAMPLE_VISIT
  end

  def audit_trail(_visit_id)
    [ { "created_at" => "2026-09-28T12:00:01Z", "actor" => "system", "action" => "auto_ocr", "detail" => {} } ]
  end

  def submit_correction(visit_id, detection_id:, field_class:, corrected_text:, corrected_by: "rails-admin")
    @last_correction = { visit_id: visit_id, detection_id: detection_id, field_class: field_class, corrected_text: corrected_text }
  end
end

class GateVisitsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @fake_client = FakeDwarpalApiClient.new
  end

  test "index lists visits from the API" do
    DwarpalApiClient.stub :new, @fake_client do
      get gate_visits_path
    end

    assert_response :success
    assert_match "abc123", response.body
  end

  test "show renders a visit's fields and audit trail" do
    DwarpalApiClient.stub :new, @fake_client do
      get gate_visit_path("abc123")
    end

    assert_response :success
    assert_match "usdot", response.body
    assert_match "auto_ocr", response.body
  end

  test "create_correction submits to the API and redirects" do
    DwarpalApiClient.stub :new, @fake_client do
      post corrections_gate_visit_path("abc123"), params: { detection_id: "det-1", field_class: "usdot", corrected_text: "7654321" }
    end

    assert_redirected_to gate_visit_path("abc123")
    assert_equal "7654321", @fake_client.last_correction[:corrected_text]
  end

  test "renders a friendly error when the API is unreachable" do
    failing_client = Object.new
    def failing_client.list_visits
      raise DwarpalApiClient::ApiError, "connection refused"
    end

    DwarpalApiClient.stub :new, failing_client do
      get gate_visits_path
    end

    assert_response :bad_gateway
    assert_match "Dwarpal API unavailable", response.body
  end
end
