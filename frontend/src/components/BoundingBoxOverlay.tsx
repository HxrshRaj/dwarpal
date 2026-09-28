"use client";

import { FieldOut, VehicleBoxOut } from "@/lib/api";

const FIELD_COLORS: Record<string, string> = {
  plate: "#22c55e",
  usdot: "#3b82f6",
  trailer_id: "#f59e0b",
  seal: "#ef4444",
  truck: "#a855f7",
  bus: "#a855f7",
  car: "#a855f7",
};

export function BoundingBoxOverlay({
  imageUrl,
  naturalWidth,
  naturalHeight,
  vehicleBoxes,
  fields,
}: {
  imageUrl: string;
  naturalWidth: number;
  naturalHeight: number;
  vehicleBoxes: VehicleBoxOut[];
  fields: FieldOut[];
}) {
  return (
    <div className="relative inline-block w-full max-w-2xl">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={imageUrl} alt="visit frame" className="w-full h-auto rounded border border-neutral-300" />
      <svg
        viewBox={`0 0 ${naturalWidth} ${naturalHeight}`}
        className="absolute inset-0 w-full h-full"
        preserveAspectRatio="none"
      >
        {vehicleBoxes.map((b, i) => {
          const [x, y, w, h] = b.bbox_xywh;
          return (
            <g key={`veh-${i}`}>
              <rect x={x} y={y} width={w} height={h} fill="none" stroke={FIELD_COLORS[b.label] ?? "#a855f7"} strokeWidth={3} />
              <text x={x} y={Math.max(12, y - 4)} fill={FIELD_COLORS[b.label] ?? "#a855f7"} fontSize={16}>
                {b.label} {(b.score * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
        {fields.map((f, i) => {
          const [x, y, w, h] = f.bbox_xywh;
          const color = FIELD_COLORS[f.field_class] ?? "#eab308";
          return (
            <g key={`field-${i}`}>
              <rect x={x} y={y} width={w} height={h} fill="none" stroke={color} strokeWidth={2} strokeDasharray={f.needs_review ? "6 4" : undefined} />
              <text x={x} y={Math.max(12, y - 4)} fill={color} fontSize={14}>
                {f.field_class}: {f.normalized_text || "?"} ({(f.confidence * 100).toFixed(0)}%)
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
