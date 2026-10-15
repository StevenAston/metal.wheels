# Project: metal.wheels

*Planning artefact — status: **CONFIRMED** by Steve, 2026-10-15. This document is the source of truth for the build; any divergence is flagged here and re-confirmed.*

## Overview

**metal.wheels** is a long-horizon (1900 → 3025) material-flow and energy model for asking whether quality degradation of metal supply — falling ore grades, increasing depth, dispersal, and contamination ratchets — compounds across industries into a growing "maintenance energy" burden. It starts with a fully sourced copper stock-flow model, couples copper to the steel scrap system (copper as a tramp element forcing primary-iron dilution), then generalises to a small input–output cascade whose spectral radius $\rho(\mathbf{A})$ tracks how close the system comes to spending its output reproducing its own inputs. A final phase catalogues ratchets (via Metal Wheel carrier-compatibility logic) and tests design levers, including the "carbon-cup" scenario: carbon-reducible metals housed in burnable sub-matrices recovered from concentrated ash, with the CO₂ stream captured.

Every parameter carries provenance (source, table/page, value, unit, range). Numbers without a source are flagged as assumptions, never silently embedded.

---

## Architecture

```
            ┌─────────────────────── parameter registry (YAML + provenance) ───────────────────────┐
            │                                                                                       │
 historical production ─►  COPPER CORE (Phase 1)                                                  │
 population scenarios  ─►  ├─ dynamic stock model (product groups, lifetimes, fabrication loop)   │
                           ├─ end-of-life fate partition (c, p, landfill, other scrap, dissipation)│
                           ├─ geology (lognormal grade × depth layers; endowment bracket)          │
                           ├─ energy (grade hyperbola + depth term; optional Koppelaar)            │
                           └─ mass outputs (ore, waste rock, tailings volume)                      │
                                         │ copper → "other scrap markets"                          │
                                         ▼                                                          │
                           STEEL COUPLING (Phase 2): scrap-pool Cu concentration, tolerances,      │
                           dilution share, primary-iron energy penalty                             │
                                         │ state (grade, depth, contamination)                     │
                                         ▼                                                          │
                           CASCADE (Phase 3): A(t), Leontief inverse, ρ(A), maintenance energy     │
                                         ▼                                                          │
                           RATCHETS & LEVERS (Phase 4): catalogue, Metal Wheel matrix, carbon-cup ◄┘
                                         ▼
          scenario engine + Monte Carlo (capped workers) ─► outputs (Parquet/CSV) ─► reports + web artifact
```

Data flow is one-directional per annual time step within a run: stocks → flows → fates → primary demand → geology/energy → (steel) → (cascade). Phases 2–4 read Phase 1 state; they never mutate it except through declared scenario levers.

Key relations (documented in code docstrings and reports):

- Per-cycle loss: $L = 1-(1-f)(1-u)\,c\,p$
- Dilution share: $x = \dfrac{c_s - c_t}{c_s - c_p}$
- Energy per kg refined Cu: $E(g,D) = a + \dfrac{b}{g} + k_D\,(D - D_0)$
- Cascade: $\mathbf{e} = \boldsymbol{\varepsilon}\,(\mathbf{I}-\mathbf{A})^{-1}$, tracked with $\rho(\mathbf{A})$

---

## File / Module Structure

```
metal.wheels/
├── PLAN.md                     # this artefact (source of truth)
├── README.md                   # usage, reproduction, citation
├── LICENSE                     # MIT (code)
├── LICENSE-docs                # CC-BY-4.0 (documentation, reports, parameter registry text)
├── pyproject.toml              # package metadata, deps, ruff/pytest config
├── .gitignore
├── references/
│   └── references.json         # CSL-JSON bibliography; every param.source must resolve here
├── params/                     # parameter registry (YAML, one file per domain)
│   ├── copper_flows.yaml       # Glöser-Chahoud 2017 / Helbig 2020 / Charpentier Poncelet 2022
│   ├── copper_lifetimes.yaml   # Glöser-Chahoud Table 5.1 product groups
│   ├── geology.yaml            # Gerst 2008, Harmsen 2013, Kesler & Wilkinson 2008, USGS
│   ├── energy.yaml             # Norgate & Jahanshahi, Koppelaar 2016, Harmsen 2013
│   ├── steel.yaml              # Daehn et al. 2017/2019, steel route energies
│   ├── cascade.yaml            # inter-sector coefficients (Phase 3)
│   ├── ratchets.yaml           # ratchet catalogue (Phase 4)
│   └── demand.yaml             # saturation stock per capita (Task 1.4)
├── data/
│   ├── raw/                    # source files actually used, with SHA-256 manifest
│   └── processed/              # cleaned series (historical production, population)
├── scenarios/                  # scenario YAMLs (baseline, high-collection, endowment brackets, carbon-cup…)
├── src/metalwheels/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI with Rich progress output
│   ├── logging_setup.py
│   ├── registry.py             # load/validate parameters with provenance (pydantic)
│   ├── copper/
│   │   ├── production.py       # Task 1.1 world mine production ingest
│   │   ├── stockflow.py        # dynamic stock model
│   │   ├── fates.py            # EoL partition + production losses
│   │   ├── demand.py           # stock-driven future demand (saturation × population)
│   │   ├── geology.py          # lognormal grade, depth layers, extraction ordering
│   │   ├── energy.py           # energy functions with range guards
│   │   └── mass.py             # ore, waste rock, tailings
│   ├── steel/
│   │   ├── scrap_pool.py
│   │   └── dilution.py
│   ├── cascade/
│   │   ├── matrix.py           # A(t) construction from model state
│   │   └── leontief.py         # inverse, ρ(A), maintenance energy
│   ├── ratchets/
│   │   ├── catalogue.py
│   │   ├── metal_wheel.py      # element × carrier compatibility
│   │   └── carbon_cup.py       # Ellingham eligibility + scenario lever
│   ├── scenarios.py            # scenario loading/overrides
│   ├── montecarlo.py           # sampling + capped process pool
│   ├── export.py               # Parquet/CSV + JSON for web artifact
│   └── report.py               # validation & synthesis reports
├── web/                        # interactive artifact source (Phase 1 front end onward)
└── tests/                      # pytest; one test module per task done-condition
```

---

## Dependencies

- **Python ≥ 3.12** (development environment runs 3.13), managed with `uv`
- Core: `numpy`, `scipy`, `pandas`, `pyarrow`, `pydantic` (v2), `pyyaml`
- CLI / UX: `typer`, `rich` (progress bars, tables)
- Plots: `matplotlib`
- Dev: `pytest`, `pytest-cov`, `ruff`
- Spreadsheet ingest (Task 1.1): `xlrd` for legacy `.xls`, `openpyxl` for `.xlsx`
- Web artifact (Task 1.11): self-contained HTML/JS page fed by exported JSON; Playwright for end-to-end checks
- Parallelism: `concurrent.futures.ProcessPoolExecutor`, default workers $= \max(1, n_{\text{cpu}} - 2)$, overridable in config/CLI

---

## Resolved Decisions (formerly Open Questions)

1. **Historical production series.** USGS Data Series 140 is degraded during the 2026 US federal appropriations lapse (Cloudflare errors; shutdown banner on archived pages). Steve mirrored the DS140 S3 bucket (live S3 cross-checked against Wayback) into `sources/usgs.140/` on his machine, with `manifest.csv` and `run.log`. Task 1.1 ingests the copper files from that mirror; the specific raw files used are committed to `data/raw/` with SHA-256 hashes and the mirror's provenance recorded. The full mirror is kept outside git.
2. **Execution environment.** Development and tests run in Claude's cloud workspace; GitHub is the hub; Steve's folder `C:\Users\steve\Development\metal.wheels` becomes a clone so he can run and tinker locally.
3. **Remote.** GitHub. Licences: MIT (code), CC-BY-4.0 (docs, reports, registry text).
4. **Population driver.** UN World Population Prospects 2024, medium variant to 2100, then held flat; low/high variants as scenarios.
5. **Sector scheme.** Glöser-Chahoud's 16 product groups (Tabelle 5.1) as the core; Charpentier Poncelet sector collection yields mapped where finer data is missing.
6. **Endowment.** Harmsen 2013 and Kesler & Wilkinson 2008 always shown **side by side**; neither is "default".
7. **Steel scope.** Minimal Phase 2 module with exogenous steel demand and scrap supply by product class (Daehn/Pauliuk).

---

## Tasks

### Phase 0 — Foundations

#### Task 0.1: Repository scaffold and tooling

| Field | Detail |
|---|---|
| Description | Initialise git, `pyproject.toml`, src layout, ruff/pytest config, logging setup, Typer CLI stub, `.gitignore`, licences. |
| Input | This plan. |
| Output / Effect | Installable package `metalwheels`; initial commit containing PLAN.md and scaffold. |
| Done condition | `pytest` passes (≥1 smoke test); `ruff check` and `ruff format --check` exit 0; `metalwheels --version` prints the package version; `git log` shows the initial commit `chore: initial scaffold and planning artefact`. |

#### Task 0.2: Parameter registry with provenance

| Field | Detail |
|---|---|
| Description | Pydantic schema for parameters: `value`, `unit` (SI), `low`, `high`, `distribution`, `source` (bibliography key), `locator` (table/page/figure), `note`, `status` (sourced / assumption / derived). Loader for all YAML files. |
| Input | `params/*.yaml`. |
| Output / Effect | `registry.load()` returns validated parameter objects; CLI `metalwheels params list`. |
| Done condition | Tests: a parameter missing `unit` or `source` (unless `status: assumption`) raises a validation error; `low ≤ value ≤ high` enforced; `metalwheels params list` prints a table with every Phase 1 parameter and exits 0. |

#### Task 0.3: Bibliography

| Field | Detail |
|---|---|
| Description | CSL-JSON bibliography of every source used to date (Glöser 2013, Glöser-Chahoud 2017, Helbig 2020, Charpentier Poncelet 2022, Lifset 2012, Gerst 2008, Mudd & Jowitt 2018, Northey 2014, Koppelaar 2016, Norgate & Jahanshahi 2010, Harmsen 2013, Kesler & Wilkinson 2008, USGS FS 2014-3004, USGS MCS 2026, Daehn 2017/2019, Ciacci 2015, Reuter et al., Valero & Valero, CEEC 2021, IEA). |
| Input | References gathered in chat. |
| Output / Effect | `references/references.json`. |
| Done condition | Test asserts every `source` key in every params YAML resolves to a bibliography entry, and every entry has a DOI or URL. |

### Phase 1 — Copper core

#### Task 1.1: Historical production ingest

| Field | Detail |
|---|---|
| Description | Clean world mine copper production 1900–2025 into a single annual series. |
| Input | Copper files from Steve's DS140 mirror (`sources/usgs.140/`) + MCS 2026. |
| Output / Effect | `data/processed/cu_mine_production.csv` (year, Mt Cu, source); raw files + SHA-256 in `data/raw/`. |
| Done condition | 126 annual rows, no gaps; cumulative 1900–2015 within ±10% of USGS's ~600 Mt; 2025 value 23 ± 0.5 Mt; test verifies all three. |

#### Task 1.2: Dynamic in-use stock model (historical, inflow-driven)

| Field | Detail |
|---|---|
| Description | Product-group cohort model with normal lifetime distributions (σ = 15% of mean), fabrication efficiencies, and new-scrap loop. Supply-driven: primary input = Task 1.1 mine production; scrap flows endogenous via $c$, $p$ (decided by Steve 2026-10-15). ICSG refined usage/production 1960–2019 used as an independent check. |
| Input | Task 1.1 series; `copper_lifetimes.yaml`. |
| Output / Effect | Annual in-use stock and end-of-life outflow by product group. |
| Done condition | In-use stock in 2010 within 320–380 Mt and EoL outflow in 2011 within 11.1 ± 1.5 Mt (Glöser-Chahoud Fig. 5.8/5.12); mass balance (inflow − outflow − Δstock) < 10⁻⁶ relative every year. |

#### Task 1.3: End-of-life fate and production-loss partition

| Field | Detail |
|---|---|
| Description | Partition EoL copper via collection $c$ and processing $p$ into old scrap, other scrap markets, not-recycled (landfill/incineration); in-use dissipation; production losses (tailings 18% of Cu in ore, smelting losses). |
| Input | Task 1.2 outputs; `copper_flows.yaml`. |
| Output / Effect | Annual flows to each receiving medium; cumulative landfill, tailings Cu, dissipated, cross-contaminating. |
| Done condition | 2011 flows each within ±15% of Helbig Fig. S11 (tailings 3,550 kt; old scrap 5,200; other scrap markets 2,300; not recycled 3,600; dissipation 450); cumulative landfilled Cu in 2010 within 105–170 Mt; whole-system mass balance closes to < 0.1% every year. |

#### Task 1.4: Future demand driver (stock-driven)

| Field | Detail |
|---|---|
| Description | From 2026, demand derived from per-capita in-use stock approaching saturation (140–300 kg/person, central 220; UNEP 2010 Table 1 more-developed-country range) on a logistic path fitted to 1995–2025 history, × population, plus replacement of retirements. |
| Input | Task 1.2/1.3 state at 2025; UN WPP 2024 population; saturation parameters. |
| Output / Effect | Annual refined demand, primary vs secondary split. |
| Done condition | Refined demand in 2026 within 26–32 Mt; stock converges to within 1% of saturation target once population is flat; per-capita stock never exceeds the configured ceiling (tests). |

#### Task 1.5: Geology — grade and depth

| Field | Detail |
|---|---|
| Description | Lognormal grade-tonnage (Gerst 2008: $\mu_w = -0.6616$, $\sigma_w^2 = 0.1567$; Kesler-extended $-0.7467$, $0.1892$) applied within depth layers (Harmsen Table 1; Kesler & Wilkinson bracket); extraction proceeds by accessibility then grade. |
| Input | Primary extraction series; `geology.yaml`. |
| Output / Effect | Mined grade, mean mining depth, remaining copper by layer, layer exhaustion years — for both endowment brackets side by side. |
| Done condition | Unit tests: distribution median 0.52 ± 0.01% Cu; analytic cumulative-above-cutoff reproduced to 10⁻⁶; replaying historical production yields mined grade in 2010 within 0.5–0.75% Cu (observed 0.62%); layer exhaustion under a constant extraction rate matches the analytic quotient. |

#### Task 1.6: Energy module

| Field | Detail |
|---|---|
| Description | $E(g,D)$ with Norgate coarse ($13 + 59/g$) and fine ($14 + 115/g$) options, depth term 2.9–4.6 MJ/kg per 100 m beyond 497 m, optional Koppelaar log-linear fit with data-range guards. |
| Input | Grade, depth; `energy.yaml`. |
| Output / Effect | Energy per kg refined Cu; annual total energy (EJ). |
| Done condition | Reproduces Norgate 72/131/249 MJ/kg at 1.0/0.5/0.25% within 2%; Koppelaar surface 52/95/255 MJ/kg at 0.5/0.4/0.3% (300 m) within 3%; a WARNING is logged whenever an option is evaluated outside its source data range (test captures the log). |

#### Task 1.7: Mass outputs

| Field | Detail |
|---|---|
| Description | Ore, waste rock (strip-ratio parameter), tailings mass and volume. |
| Input | Mined Cu, grade, recovery. |
| Output / Effect | Annual and cumulative Gt and km³. |
| Done condition | Tests: ore = Cu ÷ grade exactly; tailings volume uses configured dry density (default 1.40 t/m³, Morales et al. 2021); 23 Mt at 0.62% returns 3.7 ± 0.05 Gt ore. |

#### Task 1.8: Scenario engine and CLI run

| Field | Detail |
|---|---|
| Description | YAML scenarios with parameter overrides (baseline; $c,p$ = 0.85/0.95/0.98; endowment Harmsen/Kesler side by side); `metalwheels run <scenario>`. |
| Input | Tasks 1.1–1.7. |
| Output / Effect | Parquet time series + summary CSV per run; Rich progress display. |
| Done condition | `metalwheels run baseline` completes 1900–3025 in < 60 s on one core and writes outputs matching the declared schema (schema test); each named scenario runs without error. |

#### Task 1.9: Monte Carlo

| Field | Detail |
|---|---|
| Description | Sample registry ranges; run in a process pool capped at $\max(1, n_{\text{cpu}}-2)$ by default. |
| Input | Scenario + registry distributions; seed. |
| Output / Effect | 5/50/95% bands for key outputs. |
| Done condition | 1,000 draws complete; test confirms worker count ≤ configured cap; identical seed gives bit-identical summary; bands written to output. |

#### Task 1.10: Phase 1 validation report

| Field | Detail |
|---|---|
| Description | Auto-generated report comparing model against Glöser 2011 flows, USGS 2025 production, EoL recycling rate 0.46 ± 0.05, mined grade, Northey 2014 peak framing, Harmsen layer framing. |
| Input | Baseline + Monte Carlo outputs. |
| Output / Effect | `reports/phase1_validation.md` (+ figures). |
| Done condition | Report regenerates from one CLI command; contains a pass/fail table for every calibration target in Tasks 1.1–1.7; all pass, or each deviation is explained and accepted by Steve before Phase 2 begins. |

#### Task 1.11: Interactive web artifact (Phase 1 front end)

| Field | Detail |
|---|---|
| Description | Self-contained page: scenario selector, sliders for $c$, $p$, saturation; endowment brackets shown side by side; charts of primary demand, grade, depth, energy, tailings, layer exhaustion. Fed by exported JSON scenario grid. |
| Input | `export.py` output. |
| Output / Effect | Published artifact. |
| Done condition | Playwright test: page loads, all controls change charts; three spot-check values equal the Python outputs within 0.5%; renders at 375 px width without horizontal scroll. |

### Phase 2 — Steel coupling

#### Task 2.1: Steel scrap pool

| Field | Detail |
|---|---|
| Description | Exogenous steel demand and EoL scrap by product class (flat, long, other); copper concentration tracked through the pool across cycles. |
| Input | `steel.yaml` (Daehn 2017, Pauliuk et al. series). |
| Output / Effect | Annual scrap Cu concentration by class. |
| Done condition | 2008 scrap copper concentrations reproduce Daehn et al. 2017 estimates within their stated ranges (test). |

#### Task 2.2: Copper → steel link

| Field | Detail |
|---|---|
| Description | Route a parameterised share of copper's "other scrap markets" flow into the ferrous pool. |
| Input | Task 1.3 flows; Task 2.1 pool. |
| Output / Effect | Coupled runs. |
| Done condition | Cross-module mass-conservation test: copper leaving the copper module equals copper entering the steel pool plus declared other sinks, every year, to 10⁻⁶ relative. |

#### Task 2.3: Tolerances and dilution

| Field | Detail |
|---|---|
| Description | Product tolerances (0.06–0.4 wt%); dilution share $x = (c_s - c_t)/(c_s - c_p)$; primary-iron requirement; unusable-scrap accounting. |
| Input | Task 2.1–2.2 state. |
| Output / Effect | Primary iron demand attributable to copper; crossover year. |
| Done condition | Unit tests of $x$ against hand-computed cases (0.4→0.1 gives 0.79); with Daehn-equivalent inputs the scrap-copper-exceeds-tolerance crossover falls in 2040–2060. |

#### Task 2.4: Energy penalty accounting

| Field | Detail |
|---|---|
| Description | Route energies (BF-BOF, scrap-EAF, DRI-EAF; sourced in Task 0.3/params) and penalty $x\,(E_{\text{primary}} - E_{\text{scrap}})$ charged back to copper processing. |
| Input | Task 2.3 outputs. |
| Output / Effect | Annual copper-induced steel energy penalty (EJ). |
| Done condition | Penalty computation verified against hand calculation; a scenario with $p = 1$ (no copper to other scrap) yields zero copper-induced penalty (test). |

### Phase 3 — Cascade

#### Task 3.1: Coefficient compilation

| Field | Detail |
|---|---|
| Description | Literature table for a 7-sector matrix: copper, steel, aluminium, cement, ammonia/explosives, water/desalination, energy — e.g. grinding-media steel per tonne ore, explosives per tonne rock, water per tonne ore, desalination and pumping energy. |
| Input | Literature search. |
| Output / Effect | `params/cascade.yaml` with provenance and ranges. |
| Done condition | Every non-zero cell has source + range; every zero cell carries a justification note; registry validation passes. |

#### Task 3.2: IO engine

| Field | Detail |
|---|---|
| Description | Build $\mathbf{A}(t)$ from model state (grade, depth, contamination); compute $(\mathbf{I}-\mathbf{A})^{-1}$ and $\rho(\mathbf{A})$. |
| Input | Task 3.1 coefficients; Phases 1–2 state. |
| Output / Effect | Time series of $\rho(\mathbf{A})$ and multipliers. |
| Done condition | On a test matrix with known inverse, result matches `numpy.linalg.inv` to 10⁻⁹; WARNING logged when $\rho \geq 0.9$, ERROR when $\rho \geq 1$ (tests). |

#### Task 3.3: Maintenance energy metric

| Field | Detail |
|---|---|
| Description | Energy required to sustain the 2025 final-demand vector $\mathbf{y}$ over time: $\boldsymbol{\varepsilon}(t)\,(\mathbf{I}-\mathbf{A}(t))^{-1}\,\mathbf{y}_{2025}$. |
| Input | Task 3.2. |
| Output / Effect | Maintenance-energy time series per scenario. |
| Done condition | For static $\mathbf{A}$ and $\boldsymbol{\varepsilon}$ the series is constant and equals the direct calculation (test); baseline run produces a series and the report documents its definition. |

#### Task 3.4: Cascade sensitivity

| Field | Detail |
|---|---|
| Description | Elasticity of $\rho(\mathbf{A})$ and maintenance energy to each coefficient. |
| Input | Task 3.2–3.3. |
| Output / Effect | Ranked sensitivity table. |
| Done condition | Table generated by CLI; top-5 drivers reported; finite-difference elasticities agree with analytic eigenvalue sensitivities within 1% (test). |

### Phase 4 — Ratchets and design levers

#### Task 4.1: Ratchet catalogue

| Field | Detail |
|---|---|
| Description | Schema and initial entries: mechanism type (tramp contamination, dissipation, grade, depth, secondary→primary route switch), host material, tolerance, energy step, source. Seeds: Cu in steel, Sn in steel, Fe/Si in aluminium, Zn to EAF dust, glass colour mixing, polymer blending, PGM dissipation. |
| Input | Literature. |
| Output / Effect | `params/ratchets.yaml`. |
| Done condition | Catalogue validates against schema; ≥ 10 sourced entries. |

#### Task 4.2: Metal Wheel compatibility matrix

| Field | Detail |
|---|---|
| Description | Element × carrier route (Fe, Al, Cu, Pb/Zn, Ni, Sn) → recovered / dissolved-benign / dissolved-harmful / lost to slag / volatilised. |
| Input | Reuter et al. Metal Wheel publications. |
| Output / Effect | Matrix module + rendered table. |
| Done condition | ≥ 30 elements covered; every cell sourced or explicitly `unknown`; test checks no blank cells. |

#### Task 4.3: Carbon-cup scenario

| Field | Detail |
|---|---|
| Description | Lever shifting a share of carbon-reducible metals into burnable, separable sub-matrices; Ellingham-based eligibility; effects on $p$, steel contamination, cascade, and a CO₂ stream with configurable capture fraction and biogenic share. |
| Input | Tasks 1–4.2. |
| Output / Effect | Scenario outputs: Δ maintenance energy, Δ$\rho(\mathbf{A})$, CO₂ captured/emitted. |
| Done condition | Eligibility test rejects Al, Mg, Ti, rare earths, Li and accepts Cu, Ni, Co, Sn, Pb, Ag, Au, PGMs; scenario runs end to end and reports all three outputs. |

#### Task 4.4: Synthesis report

| Field | Detail |
|---|---|
| Description | Cross-phase report: ratchet interactions, maintenance-energy trajectories, design-lever comparison. |
| Input | All phases. |
| Output / Effect | `reports/synthesis.md` (+ artifact update). |
| Done condition | Regenerates from one CLI command; every number in it traces to a run ID and parameter registry snapshot. |

---

## Task Ordering

[ ] Sequential
[ ] Parallel
[x] **Mixed**

- **Phase 0:** 0.1 → 0.2 → 0.3 (sequential).
- **Phase 1:** 1.1 → 1.2 → 1.3 → 1.4. Tasks 1.5, 1.6, 1.7 depend only on 0.2 and may run in parallel with 1.1–1.4. Task 1.8 requires 1.3–1.7; then 1.9 → 1.10 → 1.11.
- **Phase 2:** begins after 1.10 is accepted; 2.1 → 2.2 → 2.3 → 2.4.
- **Phase 3:** 3.1 (literature) may run in parallel with Phase 2; 3.2 requires Phase 2; then 3.3 → 3.4.
- **Phase 4:** 4.1 and 4.2 (literature) may run in parallel with Phase 3; 4.3 requires 3.3 and 4.2; 4.4 last.

---

## Change Log

- **2026-10-15** — Plan confirmed. Open Questions resolved (see above). Python requirement relaxed to ≥ 3.12 (dev env is 3.13). Task 1.1 input changed to Steve's DS140 mirror owing to the USGS appropriations lapse. Licence files added to structure.
- **2026-10-15** — Task 0.2: Glöser-Chahoud Tabelle 5.1 has 16 product groups, not 15 (corrected above). Added `params/demand.yaml` (saturation stock, Task 1.4) to the structure; mass parameters (tailings density) live in `geology.yaml`. Registry convention: grades are mass fractions, never percent (Gerst's ln-wt% parameters stored as published and converted in code). Added optional `range_source` field so bounds can cite a different source from the central value.
- **2026-10-15** — Task 0.3: bibliography built from doi.org/Crossref/DataCite records (22 entries), adding USGS DS140/DS896 (data sources for Task 1.1) and Morales et al. 2021 (tailings density, supplied by Steve). Tailings dry density default changed 1.6 → 1.40 t/m³ (in-situ range 1.30–1.48). CEEC 2021 and IEA entries deferred until the specific documents are identified.
- **2026-10-15** — Bibliography: added CEEC 2021 (Allen/engeco), IEA 2020 steel roadmap, IEA 2021 heavy-industry article, Michaux 2021 (GTK 16/2021) and Rötzer & Schmidt 2020. Noted for later tasks (no done conditions changed): (a) IEA sources disagree on scrap-vs-ore steel energy (1/8 vs 60–70 % less) — reconcile on a common primary-energy basis in Task 2.4; (b) Rötzer & Schmidt 2020 CED (~89 GJ/t at 0.5 % Cu) as a cross-check against Norgate (131 MJ/kg at 0.5 %) in Task 1.6; (c) ore texture (grind size) and ore hardness as candidate ratchets for Task 4.1, after Michaux 2021 §9–10.
- **2026-10-15** — Task 1.1: DS896 (global) only covers 1990 onward, so the series uses the DS140 copper workbook's 'World production' column for 1900–2020 and MCS 2022–2026 for 2021–2025 (splice year 2020 agrees exactly at 20.6 Mt). MCS PDFs were reachable live and are committed with hashes. Cumulative 1900–2015 = 647.9 Mt (+8 % vs USGS's rounded 600 Mt past production). Added `geo.usgs.identified_unextracted` (1.5 Bt) after MCS 2026 clarified that the 2.1 Bt identified figure includes past production.
- **2026-10-15** — Task 1.2 decisions (Steve): supply-driven historical driver; 16 Glöser-Chahoud groups with ICA 2019 end-use shares (ICSG Factbook 2020 p. 47) split equally within each of the five categories. Further assumptions, flagged for the Task 1.10 report: shares and $c$, $p$ held constant over 1900–2025; zero in-use stock in 1900 (pre-1900 copper ignored); in-use dissipation applied at entry; ICA 'Equipment' mapped to Konsum & Elektronik. Added `params/copper_enduse.yaml` and Helbig-S11-derived fractions (semis loss, new-scrap capture, scrap-to-smelter share). Result with sourced values and no tuning: 2010 stock 338 Mt, 2011 EoL 11.5 Mt; model refined copper sits 7–9 % below ICSG in 1960–1980 and ~5 % above in 2000.
- **2026-10-15** — Task 1.3: receiving media follow the S11 nodes (ENV.1 tailings + slag, WMR.2 other scrap markets incl. semis/manufacturing losses, ENV.2 landfill, ENV.0 dissipation). Tailings share held at 18 % for all years (assumption: historical concentrator recovery was likely lower early in the century). 2010 cumulative landfill 112 Mt sits near the low end of 105–170 Mt, consistent with ignoring pre-1900 copper and holding $c$ constant.
- **2026-10-15** — Task 1.4: saturation range changed from the unsourced 215–300 kg/person to UNEP 2010's sourced 140–300 kg/person (more-developed countries, 2000–2006), central value 220. Per-capita stock follows a logistic path from the model's own 2025 value (61.5 kg/person) with growth rate fitted to 1995–2025 (g = 0.0211/yr at k_sat = 220). Cross-check: model global per-capita stock 2000–2006 is 40.8–45.3 kg, inside UNEP's 35–55 kg. Baseline result at today's $c$, $p$: 2026 refined 28.2 Mt; stock saturates ~2.24 Bt by ~2300; steady mine output ~55 Mt/yr; cumulative mine output 2026–3025 ≈ 53 Bt (cf. Harmsen 24.9 Bt to 5 km; Kesler 89 Bt above 3.3 km). Mass-balance residual now normalised by throughput (mine + EoL) so years with zero mine output are handled.
- **2026-10-15** — Task 1.5 decisions (Steve): paced discovery and a calibrated depth-order multiplier. Implementation: 100 m depth slices; discovered copper = Harmsen 'analysed' (85 % of first-km discovered copper above 500 m, after Long 2011); only discovered copper is minable through 2010; afterwards a uniform fraction of undiscovered copper is released to keep discovered-unmined copper ≥ the 2010 ratio to annual extraction (72 yr Harmsen, 62 yr Kesler). Extraction ordered by marginal cost b/g + m·k_D·max(0, D − 497 m); m calibrated (1.42 Harmsen, 1.62 Kesler) so 77.1 % of production to 2010 comes from the top 500 m (Harmsen), which also reproduces ~11 % from 1–2 km (Harmsen 9.3 %). Kesler bracket = Harmsen depth profile truncated at 3.3 km and scaled to 89 Bt, with the Kesler-extended grade distribution. Results with baseline demand: 2010 mined grade 0.64 % (Harmsen) / 0.61 % (Kesler) vs 0.62 % observed; Harmsen endowment exhausted 2416; Kesler not exhausted by 3025 (grade 0.53 %, depth plateau ~1.9 km).
- **2026-10-15** — Task 1.6: energy options implemented as specified. Koppelaar Eq. (4) covers mining + concentrating only (not like-for-like with Norgate); range warnings are logged per evaluation call with counts and extremes. Annual energy currently covers primary copper only; a sourced secondary (scrap) energy intensity is still needed. Baseline preview (Norgate coarse): 109 MJ/kg and 1.7 EJ in 2010; ~157 MJ/kg and ~6.7 EJ by 2100 (Harmsen) / 143 MJ/kg, 6.1 EJ (Kesler).
- **2026-10-15** — Task 1.7: strip ratio 1.5 (range 1.0–2.0) derived from Nassar et al. 2022 (copper's attributable ore + waste rock 9.4 Bt in 2018, minus the model's 2018 ore 3.8 Bt; revenue-allocated, so a lower bound; 2 is the conventional TMR assumption). Concentrate grade 28.5 % (Harmsen 27–30 %). Baseline previews: ore 4.5 Gt and rock moved 11.3 Gt in 2025; ~8.4 / 21 Gt by 2100; cumulative tailings ~100 km³ by 2025 and ~450 km³ by 2100.
- **2026-10-15** — Task 1.8: recycling scenarios are *policies* applied from 2026 (instant switch, optional linear ramp) so the 1900–2025 replay is unchanged; registry `with_overrides` lets scenario values sit outside sourced ranges. Baseline 1900–3025 runs in ~1 s. Cumulative mine output 2026–3025: 53.0 Bt (baseline), 30.3 (c = p = 0.85), 14.5 (0.95), 9.5 (0.98); Harmsen endowment exhausted 2416 / 2674 / not by 3025 / not by 3025. Energy currently primary copper only; once an endowment is exhausted the stock-flow model still assumes demand is met (shortfall reported, no feedback in Phase 1).
