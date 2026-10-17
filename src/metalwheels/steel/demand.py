"""Steel demand: historical product inflow to 2025, then stock-driven to the end year.

Same structure as the copper driver (``metalwheels.copper.demand``), but by sector, following
Pauliuk et al. (2013a): from 2026 each sector's in-use stock follows S*_g(t) = k_g(t) P(t), with
k_g on a logistic path from the model's own 2025 per-capita sector stock towards
``st.saturation_stock_per_capita.<sector>`` (Pauliuk et al. 2013b Table 6). Each sector's growth
rate is fitted to its historical run over ``dem.growth_calibration_years`` (see
:func:`logistic_path` for sectors that start above saturation). Population is UN WPP
2024 (medium), held flat after ``dem.population_hold_year``.

Pauliuk et al. fit a generalised logistic per region with an imposed saturation time (global
saturation around 2150). Here one global logistic per sector is fitted to recent growth instead,
so the timing of saturation is an output rather than an input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metalwheels.copper import demand as cu_demand
from metalwheels.registry import Registry
from metalwheels.steel import stockflow as ssf
from metalwheels.steel.link import SECTORS

LAST_HISTORICAL_YEAR = cu_demand.LAST_HISTORICAL_YEAR


@dataclass
class SteelProjection:
    result: ssf.SteelResult
    population: np.ndarray
    per_capita_target: np.ndarray  # (years, sectors) kg/person, NaN for historical years
    k0: np.ndarray  # (sectors,) kg/person in 2025
    k_sat: np.ndarray  # (sectors,)
    growth_rate: np.ndarray  # (sectors,)

    @property
    def per_capita_stock(self) -> np.ndarray:
        return self.result.flows["stock"].to_numpy() * 1e9 / self.population


def logistic_path(k0: float, k_sat: float, g: float, dt: np.ndarray) -> np.ndarray:
    """Logistic per-capita path from ``k0`` to ``k_sat``. Unlike the copper version, ``k0`` may
    exceed ``k_sat`` (a Monte Carlo draw with a low saturation level); the stock then declines
    smoothly to saturation at the same rate. Sectors starting above saturation, or whose growth
    rate cannot be fitted, use the mean fitted rate of the other sectors."""
    if k0 <= 0 or k_sat <= 0:
        raise ValueError("per-capita stocks must be positive")
    return k_sat / (1 + ((k_sat - k0) / k0) * np.exp(-g * dt))


def saturation_levels(reg: Registry) -> np.ndarray:
    return np.array([reg.value(f"st.saturation_stock_per_capita.{s}") for s in SECTORS])


def project(
    years: np.ndarray,
    reg: Registry,
    attached_cu: np.ndarray | None = None,
    k_sat: np.ndarray | None = None,
    growth_rate: np.ndarray | None = None,
) -> SteelProjection:
    """Steel run over ``years`` (must start at the first historical year)."""
    years = np.asarray(years)
    prm = ssf.SteelParams.from_registry(reg)
    pop = cu_demand.population_path(
        years, cu_demand.load_population(), int(reg.value("dem.population_hold_year"))
    )
    k_sat = saturation_levels(reg) if k_sat is None else np.asarray(k_sat, dtype=float)
    hist = years <= LAST_HISTORICAL_YEAR
    products = np.full(len(years), np.nan)
    products[hist] = ssf.historical_products(reg, years[hist])

    attached = None if attached_cu is None else np.asarray(attached_cu, dtype=float)
    hist_run = ssf.run(
        years[hist], products[hist], prm, None if attached is None else attached[hist]
    )
    stock_hist = hist_run.stock_by_sector
    ok = [i for i in range(int(hist.sum())) if np.isfinite(pop[i])]
    k0 = stock_hist.loc[LAST_HISTORICAL_YEAR].to_numpy() * 1e9 / pop[hist][-1]
    window = int(reg.value("dem.growth_calibration_years"))
    rates = []
    for j, s in enumerate(SECTORS):
        pc = {int(years[i]): stock_hist[s].iloc[i] * 1e9 / pop[i] for i in ok}
        start = pc[LAST_HISTORICAL_YEAR - window]
        if np.sqrt(start * k0[j]) < k_sat[j]:
            rates.append(cu_demand.fit_growth_rate(pc, k_sat[j], window, LAST_HISTORICAL_YEAR))
        else:
            rates.append(np.nan)  # already at or above saturation (possible in Monte Carlo)
    g = np.array(rates) if growth_rate is None else np.asarray(growth_rate, dtype=float)
    if np.all(np.isnan(g)):
        raise ValueError("every steel sector is already above its saturation level")
    g = np.where(np.isnan(g), np.nanmean(g), g)

    future = ~hist
    dt = (years[future] - LAST_HISTORICAL_YEAR).astype(float)
    k_target = np.full((len(years), len(SECTORS)), np.nan)
    for j in range(len(SECTORS)):
        k_target[future, j] = logistic_path(k0[j], k_sat[j], g[j], dt)
    target = k_target * pop[:, None] / 1e9
    result = ssf.run(years, products, prm, attached, target_stock=target)
    return SteelProjection(result, pop, k_target, k0, k_sat, g)
