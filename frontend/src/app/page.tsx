"use client";

import { useState } from "react";
import { API_URL, VisitOut, uploadVisit } from "@/lib/api";
import { BoundingBoxOverlay } from "@/components/BoundingBoxOverlay";

export default function Home() {
  const [visit, setVisit] = useState<VisitOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number }>({ w: 1, h: 1 });

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    try {
      const result = await uploadVisit(file);
      setVisit(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen p-8 max-w-5xl mx-auto">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold">Dwarpal — Gate Capture Prototype</h1>
        <p className="text-sm text-neutral-500 mt-1">
          Upload a frame to run detection + OCR + validation. This is a research prototype — see the
          {" "}<a href="/feasibility" className="underline">feasibility page</a> for what is real vs. estimated.
        </p>
      </header>

      <div className="border-2 border-dashed border-neutral-300 rounded-lg p-8 text-center mb-6">
        <input
          type="file"
          accept="image/*"
          disabled={loading}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
        {loading && <p className="mt-3 text-sm text-neutral-500">Running detection + OCR on CPU — this can take a few seconds…</p>}
        {error && <p className="mt-3 text-sm text-red-600">Error: {error} (is the backend running at {API_URL}?)</p>}
      </div>

      {visit && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h2 className="font-medium mb-2">
              Frame — status: <span className="font-mono text-sm px-2 py-0.5 rounded bg-neutral-100">{visit.status}</span>
            </h2>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${API_URL}/visits/${visit.id}/image`}
              alt="uploaded frame"
              className="hidden"
              onLoad={(e) => {
                const img = e.currentTarget;
                setDims({ w: img.naturalWidth, h: img.naturalHeight });
              }}
            />
            <BoundingBoxOverlay
              imageUrl={`${API_URL}/visits/${visit.id}/image`}
              naturalWidth={dims.w}
              naturalHeight={dims.h}
              vehicleBoxes={visit.vehicle_boxes}
              fields={visit.fields}
            />
          </div>

          <div>
            <h2 className="font-medium mb-2">Extracted fields</h2>
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="text-left border-b border-neutral-200">
                  <th className="py-1 pr-2">Field</th>
                  <th className="py-1 pr-2">Text</th>
                  <th className="py-1 pr-2">Confidence</th>
                  <th className="py-1 pr-2">Valid</th>
                  <th className="py-1">Review</th>
                </tr>
              </thead>
              <tbody>
                {visit.fields.length === 0 && (
                  <tr>
                    <td colSpan={5} className="py-3 text-neutral-500">
                      No fields detected — this classical/off-the-shelf pipeline has limited recall (see docs/benchmarks.md).
                    </td>
                  </tr>
                )}
                {visit.fields.map((f, i) => (
                  <tr key={i} className="border-b border-neutral-100">
                    <td className="py-1 pr-2 font-mono">{f.field_class}</td>
                    <td className="py-1 pr-2 font-mono">{f.normalized_text || "—"}</td>
                    <td className="py-1 pr-2">{(f.confidence * 100).toFixed(0)}%</td>
                    <td className="py-1 pr-2">{f.is_valid_format ? "✓" : "✗"}</td>
                    <td className="py-1">{f.needs_review ? <span className="text-amber-600">flagged</span> : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-neutral-500 mt-3">
              Fields flagged for review appear in the <a href="/review" className="underline">review queue</a>.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
