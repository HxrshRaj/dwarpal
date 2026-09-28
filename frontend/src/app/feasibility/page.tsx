export default function Feasibility() {
  const rows = [
    { field: "License plate (US)", data: "Real (OpenALPR benchmark, AGPL — local eval only) + synthetic", verdict: "See docs/feasibility.md" },
    { field: "USDOT number", data: "Synthetic only — no public dataset found", verdict: "See docs/feasibility.md" },
    { field: "Cab number", data: "None found (real or synthetic)", verdict: "See docs/feasibility.md" },
    { field: "Trailer ID", data: "Synthetic only — no public dataset found", verdict: "See docs/feasibility.md" },
    { field: "Seal presence", data: "Synthetic only, explicitly experimental", verdict: "See docs/feasibility.md" },
  ];

  return (
    <div className="min-h-screen p-8 max-w-3xl mx-auto">
      <h1 className="text-2xl font-semibold mb-1">Feasibility summary</h1>
      <p className="text-sm text-neutral-500 mb-6">
        This page mirrors <code>docs/feasibility.md</code>, the primary deliverable of this project. Read that file
        for full numbers, commands, and reasoning — this page is a navigable summary, not a replacement.
      </p>

      <table className="w-full text-sm border-collapse mb-8">
        <thead>
          <tr className="text-left border-b border-neutral-200">
            <th className="py-1 pr-4">Field</th>
            <th className="py-1 pr-4">Data availability</th>
            <th className="py-1">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.field} className="border-b border-neutral-100">
              <td className="py-2 pr-4 font-mono">{r.field}</td>
              <td className="py-2 pr-4">{r.data}</td>
              <td className="py-2">{r.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="space-y-4 text-sm">
        <div className="border border-neutral-200 rounded p-3">
          <div className="font-medium mb-1">What is real</div>
          <p className="text-neutral-600">
            Detection/OCR run on real, permissively-sourced images (COCO trucks; OpenALPR benchmark plates,
            evaluated locally only). Numbers on this data reflect actual model runs on this machine — see
            docs/benchmarks.md for exact commands and hardware.
          </p>
        </div>
        <div className="border border-neutral-200 rounded p-3">
          <div className="font-medium mb-1">What is synthetic or estimated</div>
          <p className="text-neutral-600">
            USDOT/trailer ID/seal numbers, and anything touching Jetson-class hardware, GPU fine-tuning, or gate
            deployment accuracy — all clearly labeled where they appear.
          </p>
        </div>
        <div className="border border-neutral-200 rounded p-3">
          <div className="font-medium mb-1">What is pending</div>
          <p className="text-neutral-600">
            GPU fine-tuning runs (delivered as notebooks), Jetson measurements, and cloud deployment — see the
            top-level README&apos;s final report for the current, authoritative list.
          </p>
        </div>
      </div>
    </div>
  );
}
