# A plain Ruby model (no ActiveRecord / no database of its own -- this app
# was generated with --skip-active-record) wrapping a Visit fetched from the
# FastAPI gate service. All persistence lives in the Python backend's
# PostgreSQL database; this app is a read/correct client over HTTP.
class GateVisit
  attr_reader :id, :created_at, :image_path, :status, :vehicle_boxes, :fields

  def initialize(attrs)
    @id = attrs.fetch("id")
    @created_at = attrs["created_at"]
    @image_path = attrs["image_path"]
    @status = attrs["status"]
    @vehicle_boxes = attrs.fetch("vehicle_boxes", [])
    @fields = attrs.fetch("fields", []).map { |f| Field.from_json(f) }
  end

  def self.all(client: DwarpalApiClient.new)
    client.list_visits.map { |attrs| new(attrs) }
  end

  def self.find(id, client: DwarpalApiClient.new)
    new(client.get_visit(id))
  end

  def needs_review?
    status == "pending_review"
  end

  Field = Struct.new(:id, :field_class, :bbox_xywh, :raw_text, :normalized_text,
                      :ocr_engine, :confidence, :is_valid_format, :format_note,
                      :needs_review, keyword_init: true)

  class Field
    def self.from_json(attrs)
      new(
        id: attrs["id"],
        field_class: attrs["field_class"],
        bbox_xywh: attrs["bbox_xywh"],
        raw_text: attrs["raw_text"],
        normalized_text: attrs["normalized_text"],
        ocr_engine: attrs["ocr_engine"],
        confidence: attrs["confidence"],
        is_valid_format: attrs["is_valid_format"],
        format_note: attrs["format_note"],
        needs_review: attrs["needs_review"]
      )
    end
  end
end
