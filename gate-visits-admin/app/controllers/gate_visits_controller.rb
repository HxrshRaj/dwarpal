class GateVisitsController < ApplicationController
  rescue_from DwarpalApiClient::ApiError, with: :api_unavailable

  def index
    @visits = GateVisit.all(client: api_client)
    @api_base_url = api_client.base_url
  end

  def show
    @visit = GateVisit.find(params[:id], client: api_client)
    @audit_events = api_client.audit_trail(@visit.id)
  end

  def create_correction
    api_client.submit_correction(
      params[:id],
      detection_id: params[:detection_id],
      field_class: params[:field_class],
      corrected_text: params[:corrected_text]
    )
    redirect_to gate_visit_path(params[:id]), notice: "Correction saved."
  end

  private

  def api_client
    @api_client ||= DwarpalApiClient.new
  end

  def api_unavailable(error)
    render plain: "Dwarpal API unavailable: #{error.message}\n\n" \
                  "Start it with `docker compose up backend` or `uvicorn app.main:app` " \
                  "from the backend/ directory in the main repo, then set DWARPAL_API_URL " \
                  "if it is not on http://localhost:8000.",
           status: :bad_gateway
  end
end
