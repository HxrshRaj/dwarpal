const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type OcrCandidate = { engine: string; text: string; confidence: number };

export type FieldOut = {
  id?: string;
  field_class: string;
  bbox_xywh: number[];
  raw_text: string;
  normalized_text: string;
  ocr_engine: string;
  ocr_candidates: OcrCandidate[];
  confidence: number;
  is_valid_format: boolean;
  format_note: string;
  needs_review: boolean;
};

export type VehicleBoxOut = { label: string; bbox_xywh: number[]; score: number };

export type VisitOut = {
  id: string;
  created_at: string;
  image_path: string;
  status: string;
  vehicle_boxes: VehicleBoxOut[];
  fields: FieldOut[];
};

export type ReviewQueueItem = {
  visit_id: string;
  created_at: string;
  image_path: string;
  field: FieldOut;
};

export type AuditEventOut = {
  id: string;
  visit_id: string;
  created_at: string;
  actor: string;
  action: string;
  detail: Record<string, unknown> | null;
};

export async function uploadVisit(file: File): Promise<VisitOut> {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_URL}/visits`, { method: "POST", body: form });
  if (!resp.ok) throw new Error(`upload failed: ${resp.status}`);
  return resp.json();
}

export async function listVisits(): Promise<VisitOut[]> {
  const resp = await fetch(`${API_URL}/visits`, { cache: "no-store" });
  if (!resp.ok) throw new Error(`list visits failed: ${resp.status}`);
  return resp.json();
}

export async function getReviewQueue(): Promise<ReviewQueueItem[]> {
  const resp = await fetch(`${API_URL}/review-queue`, { cache: "no-store" });
  if (!resp.ok) throw new Error(`review queue failed: ${resp.status}`);
  return resp.json();
}

export async function submitCorrection(
  visitId: string,
  body: { detection_id?: string; field_class: string; corrected_text: string; corrected_by?: string }
) {
  const resp = await fetch(`${API_URL}/visits/${visitId}/corrections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`correction failed: ${resp.status}`);
  return resp.json();
}

export async function getAuditTrail(visitId: string): Promise<AuditEventOut[]> {
  const resp = await fetch(`${API_URL}/visits/${visitId}/audit`, { cache: "no-store" });
  if (!resp.ok) throw new Error(`audit failed: ${resp.status}`);
  return resp.json();
}

export { API_URL };
