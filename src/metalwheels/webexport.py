"""Task 1.11: scenario grid for the interactive web page.

Runs the Phase 1 chain over a grid of end-of-life collection rate c, processing rate p (both as
policies from 2026) and per-capita saturation stock, and writes the results as compact JSON that
``web/index.template.html`` embeds into a single self-contained page (``web/dist/index.html``).
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from pathlib import Path

import numpy as np

from metalwheels import __version__, registry, scenarios

WEB = Path(__file__).resolve().parents[2] / "web"
TEMPLATE = WEB / "index.template.html"
PLACEHOLDER = "/*__GRID_DATA__*/null"

#: Default grid (baseline values first, so index 0 of each axis is today's system).
GRID_FULL = {
    "collection": [0.67, 0.75, 0.85, 0.95, 0.98],
    "processing": [0.69, 0.75, 0.85, 0.95, 0.98],
    "saturation": [140.0, 180.0, 220.0, 260.0, 300.0],
}
GRID_SMALL = {"collection": [0.67, 0.95], "processing": [0.69, 0.95], "saturation": [220.0]}

#: Series sent to the page (every ``STEP`` years).
SERIES = (
    "mine",
    "stock_per_capita",
    "harmsen_grade_pct",
    "kesler_grade_pct",
    "harmsen_depth_m",
    "kesler_depth_m",
    "harmsen_energy_intensity_mj_per_kg",
    "kesler_energy_intensity_mj_per_kg",
    "harmsen_primary_energy_ej",
    "kesler_primary_energy_ej",
    "harmsen_cum_tailings_volume_km3",
    "kesler_cum_tailings_volume_km3",
    "harmsen_remaining_total",
    "kesler_remaining_total",
)
LAYER_SERIES = {
    "harmsen": ["0_1km", "1_2km", "2_3km", "3_4km", "4_5km"],
    "kesler": ["0_1km", "1_2km", "2_3km", "3_4km"],
}
#: Scalars shown as readouts (exact Python values; spot-checked by the Playwright test).
SCALARS = (
    "cum_mine_output_future",
    "mine_2050",
    "mine_2100",
    "refined_2026",
    "stock_2100",
    "harmsen_first_shortfall_year",
    "kesler_first_shortfall_year",
    "harmsen_grade_2100",
    "kesler_grade_2100",
    "harmsen_depth_2100",
    "kesler_depth_2100",
    "harmsen_energy_intensity_2100",
    "kesler_energy_intensity_2100",
    "harmsen_primary_energy_2100",
    "kesler_primary_energy_2100",
    "harmsen_cum_primary_energy",
    "kesler_cum_primary_energy",
    "harmsen_cum_tailings_volume_end",
    "kesler_cum_tailings_volume_end",
)
STEP = 5


def _round(v: float, sig: int = 5) -> float | None:
    if v is None or not math.isfinite(v):
        return None
    if v == 0:
        return 0.0
    return float(f"{v:.{sig}g}")


def grid_scenario(c: float, p: float, sat: float, end_year: int = 3025) -> scenarios.Scenario:
    policy = None
    if not (math.isclose(c, 0.67) and math.isclose(p, 0.69)):
        policy = {"start_year": 2026, "ramp_years": 0, "collection": c, "processing": p}
    return scenarios.Scenario(
        name=f"grid_c{c:.2f}_p{p:.2f}_s{sat:.0f}",
        overrides={"dem.saturation_stock_per_capita": sat},
        policy=policy,
        end_year=end_year,
    )


def build_grid(grid: dict[str, list[float]] = GRID_FULL, progress=None) -> dict:
    logging.getLogger("metalwheels").setLevel(logging.ERROR)
    reg = registry.load()
    runs = []
    years = None
    for c, p, sat in itertools.product(grid["collection"], grid["processing"], grid["saturation"]):
        out = scenarios.run_scenario(grid_scenario(c, p, sat), reg)
        ts = out.timeseries
        if years is None:
            years = ts.index.to_numpy()[::STEP]
        series = {k: [_round(v) for v in ts[k].to_numpy()[::STEP]] for k in SERIES}
        for br, layers in LAYER_SERIES.items():
            for layer in layers:
                col = f"{br}_remaining_{layer}"
                series[col] = [_round(v) for v in ts[col].to_numpy()[::STEP]]
        scalars = {k: _round(out.summary[k][0], 8) for k in SCALARS}
        runs.append({"c": c, "p": p, "s": sat, "scalars": scalars, "series": series})
        if progress:
            progress()
    return {
        "version": __version__,
        "grid": grid,
        "step": STEP,
        "years": [int(y) for y in np.asarray(years)],
        "layers": LAYER_SERIES,
        "runs": runs,
    }


_SKELETON = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "[hidden]{display:none!important}img{max-width:100%}</style>\n</head>\n<body>\n{body}\n</body>\n</html>\n"
)


def render_page(data: dict, template: Path = TEMPLATE, standalone: bool = False) -> str:
    """Page body with the grid embedded. ``standalone`` wraps it in a full HTML document for
    opening from disk; the artifact publisher adds its own skeleton, so the default omits it."""
    html = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in html:
        raise ValueError(f"template has no {PLACEHOLDER!r} placeholder")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    body = html.replace(PLACEHOLDER, payload)
    return _SKELETON.replace("{body}", body) if standalone else body


def write_pages(data: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ``index.html`` (standalone, open locally) and ``artifact.html`` (for publishing)."""
    d = out_dir or WEB / "dist"
    d.mkdir(parents=True, exist_ok=True)
    local, art = d / "index.html", d / "artifact.html"
    local.write_text(render_page(data, standalone=True), encoding="utf-8")
    art.write_text(render_page(data), encoding="utf-8")
    return local, art
