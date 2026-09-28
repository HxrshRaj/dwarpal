# Feasibility: automated gate capture, field by field

This is the primary deliverable of this project. It is deliberately blunt.
Every number cited here links back to `docs/benchmarks.md` or
`docs/edge-feasibility.md` — nothing here is invented for the sake of a
clean-looking table.

**Status: verdicts below are structural (based on real data-availability
findings, which do not change once benchmarks run) — the accuracy columns
are filled in from `docs/benchmarks.md` in the same commit as the first
full benchmark run.** See that file for current status.

## Summary table

| Field | Data availability | Best measured accuracy (real) | Best measured accuracy (synthetic) | Main failure modes | Verdict |
|---|---|---|---|---|---|
| License plate (US) | Small real, labeled set (OpenALPR benchmark, 444 images, **AGPL-3.0 — local eval only**, see `docs/data.md`); no permissively-licensed real dataset found without an API key we don't have | _pending benchmarks.md_ | _pending benchmarks.md_ | Classical MSER proposer has limited recall on small/angled/low-contrast plates (expected — it is not a trained detector, see `docs/data.md` "Detector license choice"); OCR degrades under our synthetic blur/glare augmentations | **Needs a trained detector + a licensed dataset before in-house is credible.** Off-the-shelf classical CV alone is not sufficient (see measured recall); a fine-tuned model (YOLO-class, budget for Ultralytics commercial license or an Apache-licensed reimplementation) on a properly licensed plate dataset is a realistic near-term investment, not a research problem. |
| USDOT number | **No real public dataset found.** Format is well-specified (49 CFR 390.21), but no labeled images of USDOT markings on real vehicles are publicly available at any license. | Not applicable — no real data | _pending benchmarks.md_ | Stencil-style fonts, panel glare, and the "USDOT" prefix text competing with the digits for the OCR region are the synthetic-data failure modes observed; real-world failure modes (mud, dents, non-standard mounting) are unknown | **Needs real data collection before any in-house accuracy claim is credible.** The validator/normalizer (regex + digit-confusion fixes) is solid and format-verified against the actual regulation — that part is achievable now. Detection+OCR accuracy on real trucks is genuinely unknown until real images exist. |
| Cab number | **No real or synthetic data built for this field in this pass** — out of scope for this iteration (see README final report). Cab numbers have no regulatory format reference the way USDOT does (carrier-assigned, arbitrary). | Not applicable | Not applicable | N/A | **Needs significant investment**: no format standard to validate against, no dataset, and unlike USDOT there is no CFR citation to constrain the format search space — this is the least-specified field in the whole project. |
| Trailer ID | **No real public dataset found** for domestic trailers. ISO 6346 (intermodal chassis/container ID, with checksum) is well-specified and testable; domestic dry-van IDs have no public standard at all. | Not applicable — no real data | _pending benchmarks.md_ | Two different ID conventions competing for the same detector class is itself a real, structural difficulty — a single model has to learn both shapes, or the system needs carrier-specific config | **Needs real data collection.** The ISO 6346 checksum validator is a real, useful piece of achievable-now infrastructure (catches OCR errors deterministically without any image). Detection+OCR accuracy is unknown without real trailer photos. |
| Seal presence | **No real or synthetic-with-realistic-physics data.** Our synthetic seal is a simple procedural icon (red ellipse + tab), not a rendering of an actual security seal. | Not applicable | _pending benchmarks.md_ (and explicitly caveated — see below) | A 2D procedural icon cannot represent real seal material, color variety, partial occlusion, or the many real seal hardware types (bolt seals, cable seals, plastic strap seals) | **Needs significant investment, starting with what "seal" even means for this yard.** This module is explicitly experimental (see `backend/app/detection.py` and the project brief). Treat any accuracy number here as a sanity check on the *code path*, not a signal about real-world seal detection difficulty. |

## What would firm up each verdict

1. **License plate**: license a real, permissively-licensed (or explicitly
   commercially-licensed) plate dataset with enough volume to fine-tune a
   detector (thousands, not hundreds, of images); budget for either an
   Ultralytics commercial license or adopting an Apache-licensed detector
   architecture.
2. **USDOT / trailer ID / cab number**: this is the actual gap a yard
   operator is uniquely positioned to close — even a few hundred real gate
   photos with these fields boxed and transcribed would move these fields
   from "no public data" to "small real dataset," the same category plates
   are already in.
3. **Seal presence**: define which seal types this yard actually uses
   first (bolt/cable/plastic strap all look different), then collect real
   labeled examples — this is a data-collection problem before it is a
   modeling problem.
4. **All fields**: real gate camera geometry (angle, distance, lighting
   rig) is unlike any of the proxy datasets used here. Even a small
   (~50-100 image) real-gate calibration set, without full labeling, would
   let us measure the domain gap between these benchmarks and reality —
   right now that gap is completely unknown and should not be assumed
   small.

## Edge deployment feasibility

See `docs/edge-feasibility.md` in full. Summary: CPU latency/size numbers
are real and measured on the build machine; Jetson feasibility is an
estimate with explicit assumptions, or "unknown" where no real number or
citable source supports even an estimate.
