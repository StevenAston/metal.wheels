"""Calibrate the copper -> steel link shares to Daehn et al. (2017) 2008 scrap copper.

The share phi_g of copper 'other scrap markets' that ends up attached to sector g's steel scrap
is not observed directly. It is chosen so that the model's 2008 scrap copper concentration by
sector equals Daehn's expected values (``st.target.scrap_cu_2008.<sector>``):

    c_g(2008) = (rho B_g + phi_g X_g) / (rho E_g)  ->  phi_g = (c*_g rho E_g - rho B_g) / X_g,

where B_g is bulk copper returning in end-of-life steel and X_g the copper 'other scrap markets'
mapped to sector g. B_g depends on earlier attachment through the copper carried by each cohort,
so phi is found by damped fixed-point iteration on the 1900-2025 run. phi is clamped to [0, 1]. A
sector whose bulk copper alone exceeds its target ends at phi = 0, and the miss is reported.
Calibration runs inside every scenario and Monte Carlo draw, so phi follows the sampled
lifetimes, recovery rate and copper parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metalwheels.copper.stockflow import StockFlowResult
from metalwheels.logging_setup import get_logger
from metalwheels.registry import Registry
from metalwheels.steel import link
from metalwheels.steel import stockflow as ssf

log = get_logger(__name__)

CALIBRATION_YEAR = 2008


@dataclass(frozen=True)
class LinkCalibration:
    phi: np.ndarray  # (sectors,)
    target: np.ndarray  # scrap Cu concentration targets (mass fraction)
    achieved: np.ndarray  # model scrap Cu concentration in the calibration year
    bulk_share: np.ndarray  # share of 2008 scrap copper that is bulk (returning) copper
    iterations: int

    @property
    def max_miss(self) -> float:
        return float(np.max(np.abs(self.achieved - self.target) / self.target))


def targets(reg: Registry) -> np.ndarray:
    return np.array([reg.value(f"st.target.scrap_cu_2008.{s}") for s in link.SECTORS])


def calibrate_shares(
    cu: StockFlowResult,
    reg: Registry,
    max_iter: int = 50,
    tol: float = 1e-7,
    damping: float = 0.5,
) -> LinkCalibration:
    """phi by sector so 2008 scrap copper matches Daehn et al. (2017)."""
    other = link.other_scrap_by_category(cu)
    years = np.asarray(cu.years)
    hist = years <= CALIBRATION_YEAR
    yrs = years[hist]
    x = np.column_stack(
        [other[list(link.CU_CATEGORY_MAP[s])].sum(axis=1).to_numpy() for s in link.SECTORS]
    )[hist]
    prm = ssf.SteelParams.from_registry(reg)
    products = ssf.historical_products(reg, yrs)
    tgt = targets(reg)
    x_cal = x[-1]
    phi = np.full(len(link.SECTORS), 0.3)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):  # noqa: B007 - n_iter is reported after the loop
        r = ssf.run(yrs, products, prm, x * phi)
        rec = prm.recovery * r.eol_by_sector.loc[CALIBRATION_YEAR].to_numpy()
        conc = r.scrap_cu_by_sector.loc[CALIBRATION_YEAR].to_numpy()
        bulk = conc * rec - phi * x_cal
        want = np.clip(np.divide(tgt * rec - bulk, x_cal, where=x_cal > 0, out=np.zeros(4)), 0, 1)
        if np.max(np.abs(want - phi)) < tol:
            phi = want
            break
        phi = phi + damping * (want - phi)
    r = ssf.run(yrs, products, prm, x * phi)
    rec = prm.recovery * r.eol_by_sector.loc[CALIBRATION_YEAR].to_numpy()
    achieved = r.scrap_cu_by_sector.loc[CALIBRATION_YEAR].to_numpy()
    bulk = achieved * rec - phi * x_cal
    cal = LinkCalibration(phi, tgt, achieved, bulk / (achieved * rec), n_iter)
    if cal.max_miss > 1e-3:
        log.warning(
            "steel link calibration misses 2008 targets by up to %.1f %% (phi %s)",
            100 * cal.max_miss,
            np.round(phi, 4),
        )
    return cal
