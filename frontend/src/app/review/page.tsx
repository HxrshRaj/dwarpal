"use client";

import { useEffect, useState } from "react";
import { API_URL, ReviewQueueItem, getReviewQueue, submitCorrection } from "@/lib/api";

export default function ReviewQueuePage() {
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [edits, setEdits] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      setItems(await getReviewQueue());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function approve(item: ReviewQueueItem) {
    const key = item.field.id ?? "";
    setBusy(key);
    try {
      const text = edits[key] ?? item.field.normalized_text;
      await submitCorrection(item.visit_id, {
        detection_id: item.field.id,
        field_class: item.field.field_class,
        corrected_text: text,
        corrected_by: "operator",
      });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-1">Review queue</h1>
      <p className="text-sm text-neutral-500 mb-6">
        Fields below fell under the confidence threshold and were auto-flagged instead of auto-accepted.
        Correcting them here writes to the audit log and becomes a labeled dataset (<code>/export/corrections</code>).
      </p>

      {error && <p className="text-red-600 text-sm mb-4">Error: {error} (is the backend running at {API_URL}?)</p>}
      {items.length === 0 && !error && <p className="text-neutral-500">Queue is empty.</p>}

      <div className="space-y-4">
        {items.map((item) => {
          const key = item.field.id ?? `${item.visit_id}-${item.field.field_class}`;
          return (
            <div key={key} className="border border-neutral-200 rounded-lg p-4 flex items-center gap-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`${API_URL}/visits/${item.visit_id}/image`}
                alt="frame"
                className="w-28 h-20 object-cover rounded border border-neutral-200"
              />
              <div className="flex-1">
                <div className="text-xs text-neutral-500 font-mono">{item.field.field_class} · confidence {(item.field.confidence * 100).toFixed(0)}%</div>
                <div className="text-xs text-neutral-400 mb-2">{item.field.format_note}</div>
                <input
                  className="border border-neutral-300 rounded px-2 py-1 text-sm font-mono w-full max-w-xs"
                  defaultValue={item.field.normalized_text}
                  onChange={(e) => setEdits((prev) => ({ ...prev, [key]: e.target.value }))}
                />
              </div>
              <button
                onClick={() => approve(item)}
                disabled={busy === key}
                className="px-3 py-1.5 text-sm rounded bg-neutral-900 text-white disabled:opacity-50"
              >
                {busy === key ? "Saving…" : "Approve"}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
