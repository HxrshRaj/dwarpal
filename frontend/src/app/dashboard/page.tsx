"use client";

import { useEffect, useState } from "react";

type Row = {
  field: string;
  source: string;
  n: number;
  detection_map50?: number;
  ocr_exact_match?: number;
  cer?: number;
};

type BenchmarkData = {
  generated: boolean;
  note: string;
  real: Row[];
  synthetic: Row[];
  review_rate_over_time: { date: string; review_rate: number }[];
};

function ResultsTable({ title, rows, badgeClass, badgeText }: { title: string; rows: Row[]; badgeClass: string; badgeText: string }) {
  return (
    <div className="mb-8">
      <h3 className="font-medium mb-2 flex items-center gap-2">
        {title}
        <span className={`text-xs px-2 py-0.5 rounded ${badgeClass}`}>{badgeText}</span>
      </h3>
      {rows.length === 0 ? (
        <p className="text-sm text-neutral-500">No results yet — run benchmarks/run_benchmarks.py.</p>
      ) : (
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="text-left border-b border-neutral-200">
              <th className="py-1 pr-4">Field</th>
              <th className="py-1 pr-4">Source</th>
              <th className="py-1 pr-4">N</th>
              <th className="py-1 pr-4">Detection mAP@0.5</th>
              <th className="py-1 pr-4">OCR exact match</th>
              <th className="py-1">CER</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-neutral-100">
                <td className="py-1 pr-4 font-mono">{r.field}</td>
                <td className="py-1 pr-4">{r.source}</td>
                <td className="py-1 pr-4">{r.n}</td>
                <td className="py-1 pr-4">{r.detection_map50 !== undefined ? r.detection_map50.toFixed(3) : "—"}</td>
                <td className="py-1 pr-4">{r.ocr_exact_match !== undefined ? r.ocr_exact_match.toFixed(3) : "—"}</td>
                <td className="py-1">{r.cer !== undefined ? r.cer.toFixed(3) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<BenchmarkData | null>(null);

  useEffect(() => {
    fetch("/data/benchmark_results.json")
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null));
  }, []);

  return (
    <div className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-1">Benchmark dashboard</h1>
      <p className="text-sm text-neutral-500 mb-6">
        Real and synthetic results are always reported separately — never averaged together. See{" "}
        <a href="/feasibility" className="underline">the feasibility page</a> for the full writeup.
      </p>

      {!data?.generated && (
        <div className="border border-amber-300 bg-amber-50 text-amber-800 text-sm rounded p-3 mb-6">
          Benchmarks have not been generated yet in this deployment. {data?.note}
        </div>
      )}

      <ResultsTable title="Real data" rows={data?.real ?? []} badgeClass="bg-green-100 text-green-800" badgeText="REAL" />
      <ResultsTable title="Synthetic data" rows={data?.synthetic ?? []} badgeClass="bg-purple-100 text-purple-800" badgeText="SYNTHETIC" />
    </div>
  );
}
