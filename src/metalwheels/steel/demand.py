"""Steel demand: historical product inflow to 2025, then stock-driven to the end year.

Same structure as the copper driver (``metalwheels.copper.demand``): from 2026 the total in-use
steel stock follows S*(t) = k(t) P(t), with k(t) on a logistic path from the model's own 2025
per-capita stock towards ``st.saturation_stock_per_capita``. The growth rate is fitted to the
historical run over ``dem.growth_calibration_years``. Population is UN WPP 2024 (medium), held
flat after ``dem.population_hold_year``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metalwheels.copper import demand as cu_demand
from metalwheels.registry import Registry
from metalwheels.steel import stockflow as ssf

LAST_HISTORICAL_YEAR = cu_demand.LAST_HISTORICAL_YEAR


@dataclass
class SteelProjection:
    result: ssf.SteelResult
    population: np.ndarray
    per_capita_target: np.ndarray  # kg/person, NaN for historical years
    k0: float
    k_sat: float
    growth_rate: float

    @property
    def per_capita_stock(self) -> np.ndarray:
        return self.result.flows["stock"].to_numpy() * 1e9 / self.population


def project(
    years: np.ndarray,
    reg: Registry,
    attached_cu: np.ndarray | None = None,
    k_sat: float | None = None,
    growth_rate: float | None = None,
) -> SteelProjection:
    """Steel run over ``years`` (must start at the first historical year)."""
    years = np.asarray(years)
    prm = ssf.SteelParams.from_registry(reg)
    pop = cu_demand.population_path(
        years, cu_demand.load_population(), int(reg.value("dem.population_hold_year"))
    )
    k_sat = reg.value("st.saturation_stock_per_capita") if k_sat is None else k_sat
    hist = years <= LAST_HISTORICAL_YEAR
    products = np.full(len(years), np.nan)
    products[hist] = ssf.historical_products(reg, years[hist])

    attached = None if attached_cu is None else np.asarray(attached_cu, dtype=float)
    hist_run = ssf.run(
        years[hist], products[hist], prm, None if attached is None else attached[hist]
    )
    pc = {
        int(y): hist_run.flows.loc[y, "stock"] * 1e9 / pop[i]
        for i, y in enumerate(years[hist])
        if np.isfinite(pop[i])
    }
    k0 = pc[LAST_HISTORICAL_YEAR]
    if growth_rate is None:
        window = int(reg.value("dem.growth_calibration_years"))
        growth_rate = cu_demand.fit_growth_rate(pc, k_sat, window, LAST_HISTORICAL_YEAR)
    future = ~hist
    k_target = np.full(len(years), np.nan)
    k_target[future] = cu_demand.logistic_path(
        k0, k_sat, growth_rate, (years[future] - LAST_HISTORICAL_YEAR).astype(float)
    )
    target = np.where(future, k_target * pop / 1e9, np.nan)
    result = ssf.run(years, products, prm, attached, target_stock=target)
    return SteelProjection(result, pop, k_target, k0, k_sat, growth_rate)
