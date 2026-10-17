# metal.wheels

A long-horizon (1900 → 3025) material-flow and energy model asking whether the degradation of metal supply — falling ore grades, greater mining depth, dispersal, and contamination "ratchets" — compounds across industries into a growing **maintenance energy** burden: the energy needed just to keep providing today's level of material service.

The build proceeds in four phases (see [`PLAN.md`](PLAN.md), the source of truth):

1. **Copper core** — sourced stock-flow model with collection and processing efficiencies, lognormal grade distribution, depth layers (Harmsen 2013 and Kesler & Wilkinson 2008 shown side by side), energy and tailings.
2. **Steel coupling** — copper as a tramp element in steel scrap, forcing primary-iron dilution and an energy penalty.
3. **Cascade** — a small input–output matrix; tracks the spectral radius $\rho(\mathbf{A})$ and maintenance energy.
4. **Ratchets and design levers** — a ratchet catalogue, Metal Wheel carrier-compatibility matrix, and the "carbon-cup" scenario.

Every parameter carries provenance (source, locator, value, unit, range). Unsourced values are explicitly marked as assumptions.

## Quick start

```bash
uv sync
uv run pytest                                  # full test suite
uv run metalwheels scenarios                   # list scenarios
uv run metalwheels run baseline                # 1900-3025 run -> runs/baseline/
uv run metalwheels copper history              # historical stock-flow table
uv run metalwheels copper project --collection 0.95 --processing 0.95
uv run metalwheels params list --domain geology
uv run metalwheels export web                   # interactive page -> web/dist/index.html (open in a browser)
uv run metalwheels mc baseline                  # Monte Carlo (capped workers) -> runs/baseline/mc/
uv run metalwheels report phase1                # validation report -> reports/phase1_validation.md
uv run metalwheels data steel-production        # world crude steel 1900-2025 -> data/processed/
uv run metalwheels report phase23               # steel coupling + cascade -> reports/phase2_3.md
uv run metalwheels cascade sensitivity -y 2100  # elasticities of rho(A) and maintenance energy
uv run metalwheels cascade gaps                 # cascade cells resting on assumptions
uv run metalwheels mc baseline --no-steel       # faster Monte Carlo without Phases 2-3
uv run metalwheels ratchets list                # ratchet catalogue: tolerances, energy steps
uv run metalwheels ratchets wheel               # Metal Wheel element x carrier-route matrix
uv run metalwheels ratchets eligibility         # Ellingham carbon-reducibility by element
uv run metalwheels ratchets carbon-cup          # carbon-cup scenario vs baseline
uv run metalwheels report synthesis             # cross-phase report -> reports/synthesis.md
```

The browser tests need Chromium once: `uv run playwright install chromium`.

Outputs per run: `runs/<scenario>/timeseries.parquet` (schema in `metalwheels.export`), `summary.csv`, `meta.json` (run ID, git commit, registry hash, calibration values). The synthesis report cites runs by run ID and writes the registry it used to `reports/synthesis_snapshot.json`.

## Data note

USGS Data Series 140 (historical mineral statistics) was degraded during the 2026 US federal appropriations lapse. The copper series used here comes from an independent mirror of the DS140 S3 bucket, cross-checked against the Wayback Machine; the raw files actually used are committed under `data/raw/` with SHA-256 hashes.

## Licences

- Code: MIT ([`LICENSE`](LICENSE))
- Documentation, reports and registry text: CC BY 4.0 ([`LICENSE-docs`](LICENSE-docs))
