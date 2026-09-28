# Limitations

This document is intentionally blunt. It exists so nobody — including future-me —
overstates what this prototype proves.

## No real gate imagery

**No real yard-gate camera imagery was used anywhere in this project.** No yard
management company or trucking operation supplied data. Every image is one of:

1. A real photograph from a public dataset, taken in a context that is *not* a
   yard gate (e.g. street-level license plate photos, general traffic/CCTV
   photos containing trucks). These are useful proxies for plate/vehicle
   detection but do not capture gate-specific conditions: fixed camera
   geometry, gate lighting rigs, close-range cab/trailer framing, weather at a
   specific site, dirt/mud patterns from a specific yard.
2. Synthetically rendered (text composited onto real or synthetic backgrounds
   with augmentation), and always labeled as such.

Any accuracy number in this repo reflects performance on these proxies, not on
gate imagery. Anywhere gate deployment accuracy is discussed, it is stated as
an **estimate with assumptions**, never as measured fact.

## No GPU in the build environment

The prototype was built on a CPU-only Windows machine (see `docs/benchmarks.md`
and `docs/edge-feasibility.md` for the exact hardware). Consequences:

- All accuracy numbers in this repo come from **pretrained models run in
  inference mode on CPU**, or from notebooks executed by hand — never from
  fine-tuning runs that did not actually happen.
- Any fine-tuning is delivered as a Colab/Kaggle notebook
  (`training/*.ipynb`), clearly marked **"pending user run — needs a GPU
  runtime"**. No numbers are fabricated for these.
- No Jetson-class device was available. `docs/edge-feasibility.md` reports
  ONNX/CPU measurements plus a labeled *estimate* for Jetson, never a
  measurement.

## Dataset scarcity is real, not a shortcut taken

Public, permissively-licensed, labeled datasets exist for license plates and
general vehicle imagery. They do **not** meaningfully exist for USDOT numbers,
cab numbers, or trailer ID markings, and they do not exist at all for seal
presence/absence at a gate. This is documented per-field in `docs/data.md`
and `docs/feasibility.md` — it is a finding of this project, not a gap I
papered over with synthetic data pretending to be real.

## Scope not completed

See the final report in the top-level `README.md` ("What is real / synthetic
/ estimated / pending") for the current, authoritative list of what shipped,
what is a notebook pending a GPU run, and what needs credentials or hardware
I do not have (cloud deployment, Jetson).
