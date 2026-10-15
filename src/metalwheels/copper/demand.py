"""Task 1.4: stock-driven future copper demand.

From 2026 the in-use stock follows a target

    S*(t) = k(t) * P(t)

where P(t) is UN WPP 2024 world population (medium variant to 2100, held flat afterwards) and
k(t) is per-capita stock on a logistic path towards the saturation level k_sat:

    k(t) = k_sat / (1 + ((k_sat - k0) / k0) * exp(-g (t - t0)))

k0 is the model's own per-capita stock in t0 = 2025, so the projection starts exactly where the
historical run ends. The growth rate g is fitted to the historical run: the mean log growth of
per-capita stock over the calibration window, divided by the logistic slowing term
(1 - k / k_sat) at the window's midpoint level. The stock-flow engine then solves for the mine
output needed each year to reach S*(t), after end-of-life returns and the scrap loop.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metalwheels.copper import production
from metalwheels.copper import stockflow as sf
from metalwheels.registry import Registry

POPULATION_CSV = production.RAW_DIR / "un" / "wpp2024_world_population_medium.csv"
LAST_HISTORICAL_YEAR = 2025


def load_population(path: Path = POPULATION_CSV) -> dict[int, float]:
    """World population (persons) by year from the WPP extract."""
    with path.open(newline="", encoding="utf-8") as fh:
        return {int(r["year"]): float(r["population_thousands"]) * 1e3 for r in csv.DictReader(fh)}


def population_path(years: np.ndarray, pop: dict[int, float], hold_year: int) -> np.ndarray:
    """Population for ``years``; values after ``hold_year`` (or after the data end) held flat.

    Years before the data start (WPP begins in 1950) are NaN; they are only used for reporting.
    """
    last = min(hold_year, max(pop))
    out = []
    for y in years:
        y = int(y)
        if y in pop and y <= last:
            out.append(pop[y])
        elif y > last:
            out.append(pop[last])
        else:
            out.append(np.nan)
    return np.array(out)


def logistic_path(k0: float, k_sat: float, g: float, dt: np.ndarray) -> np.ndarray:
    """Per-capita stock after ``dt`` years starting from ``k0``."""
    if not 0 < k0 < k_sat:
        raise ValueError(f"need 0 < k0 ({k0:.1f}) < k_sat ({k_sat:.1f})")
    return k_sat / (1 + ((k_sat - k0) / k0) * np.exp(-g * dt))


def fit_growth_rate(per_capita: dict[int, float], k_sat: float, window: int, end: int) -> float:
    """Logistic rate g from mean log growth over [end - window, end]."""
    start = end - window
    k_start, k_end = per_capita[start], per_capita[end]
    mean_growth = math.log(k_end / k_start) / window
    k_mid = math.sqrt(k_start * k_end)
    return mean_growth / (1 - k_mid / k_sat)


@dataclass
class Projection:
    result: sf.StockFlowResult
    population: np.ndarray  # persons, aligned with result.years
    per_capita_target: np.ndarray  # kg/person (NaN for historical years)
    k0: float
    k_sat: float
    growth_rate: float

    @property
    def per_capita_stock(self) -> np.ndarray:
        return self.result.flows["stock"].to_numpy() * 1e9 / self.population


def project(
    end_year: int = 3025,
    reg: Registry | None = None,
    k_sat: float | None = None,
    growth_rate: float | None = None,
    **overrides: float,
) -> Projection:
    """Historical run 1900-2025 (supply-driven) then stock-driven demand to ``end_year``."""
    from metalwheels import registry as _registry

    reg = reg or _registry.load()
    prm = sf.CopperParams.from_registry(reg, **overrides)
    hist = production.load()
    years = np.arange(min(hist), end_year + 1)
    pop = population_path(years, load_population(), int(reg.value("dem.population_hold_year")))
    k_sat = reg.value("dem.saturation_stock_per_capita") if k_sat is None else k_sat

    # Historical run gives the starting point and the growth-rate calibration.
    hist_years = years[years <= LAST_HISTORICAL_YEAR]
    hist_run = sf.run(hist_years, np.array([hist[y] for y in hist_years]), prm)
    pc_hist = {
        int(y): hist_run.flows.loc[y, "stock"] * 1e9 / pop[i] for i, y in enumerate(hist_years)
    }
    k0 = pc_hist[LAST_HISTORICAL_YEAR]
    if growth_rate is None:
        window = int(reg.value("dem.growth_calibration_years"))
        growth_rate = fit_growth_rate(pc_hist, k_sat, window, LAST_HISTORICAL_YEAR)

    future = years > LAST_HISTORICAL_YEAR
    k_target = np.full(len(years), np.nan)
    k_target[future] = logistic_path(
        k0, k_sat, growth_rate, (years[future] - LAST_HISTORICAL_YEAR).astype(float)
    )
    target = k_target * pop / 1e9  # Mt
    mine = np.array([hist[y] if y <= LAST_HISTORICAL_YEAR else np.nan for y in years])
    result = sf.run(years, mine, prm, target_stock=np.where(future, target, np.nan))
    return Projection(result, pop, k_target, k0, k_sat, growth_rate)
