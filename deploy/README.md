# Deployment

**Status: pending.** No cloud account credentials (Render, Railway, Fly.io,
AWS, GCP, etc.) are available in the build environment this project was
created in. Nothing here has been deployed. This directory contains
ready-to-run configuration so deployment is a credentials problem, not a
missing-code problem.

## What's here

- `render.yaml` — a Render.com Blueprint that deploys `backend/Dockerfile`
  and `frontend/Dockerfile` plus a managed free-tier Postgres database.
  Free-tier CPU only (consistent with `docs/edge-feasibility.md` — no GPU
  numbers are claimed for a hosted deployment either).

## To actually deploy (needs a Render account)

1. Push this repo to GitHub (done — see the repo root).
2. In the Render dashboard: New -> Blueprint -> connect this repo -> Render
   reads `deploy/render.yaml` automatically.
3. Set `NEXT_PUBLIC_API_URL` if Render's automatic service-to-service URL
   wiring needs a manual override (rare, but check after first deploy).
4. Free-tier services spin down after inactivity — expect a cold-start
   delay (typically tens of seconds) on the first request after idle. This
   is a known free-tier characteristic, not something this project
   measured — do not repeat it as a project benchmark.

## Honest expectation setting

CPU-only inference on a free-tier instance will be slower than the numbers
in `docs/edge-feasibility.md` (which were measured on the build machine, a
[see docs/edge-feasibility.md for the exact spec] — likely a more capable
CPU than a shared free-tier instance gets). If this is deployed, re-run
`benchmarks/measure_edge_latency.py` against the deployed instance and
update the docs rather than assuming the build-machine numbers transfer.
