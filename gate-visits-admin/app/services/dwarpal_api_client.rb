# Thin HTTP client for the FastAPI gate service (backend/app in the main
# repo). Deliberately dependency-free (stdlib Net::HTTP + JSON only) so this
# admin app needs no extra gems beyond what `rails new` already generated.
require "net/http"
require "json"

class DwarpalApiClient
  class ApiError < StandardError; end

  attr_reader :base_url

  def initialize(base_url: ENV.fetch("DWARPAL_API_URL", "http://localhost:8000"))
    @base_url = base_url
  end

  def list_visits
    get("/visits")
  end

  def get_visit(id)
    get("/visits/#{id}")
  end

  def review_queue
    get("/review-queue")
  end

  def audit_trail(visit_id)
    get("/visits/#{visit_id}/audit")
  end

  def submit_correction(visit_id, detection_id:, field_class:, corrected_text:, corrected_by: "rails-admin")
    post(
      "/visits/#{visit_id}/corrections",
      detection_id: detection_id,
      field_class: field_class,
      corrected_text: corrected_text,
      corrected_by: corrected_by
    )
  end

  private

  def get(path)
    uri = URI("#{@base_url}#{path}")
    response = Net::HTTP.get_response(uri)
    handle(response)
  end

  def post(path, body)
    uri = URI("#{@base_url}#{path}")
    http = Net::HTTP.new(uri.host, uri.port)
    request = Net::HTTP::Post.new(uri, "Content-Type" => "application/json")
    request.body = body.to_json
    handle(http.request(request))
  end

  def handle(response)
    unless response.is_a?(Net::HTTPSuccess)
      raise ApiError, "Dwarpal API returned #{response.code}: #{response.body}"
    end

    JSON.parse(response.body)
  rescue JSON::ParserError => e
    raise ApiError, "Dwarpal API returned invalid JSON: #{e.message}"
  end
end
