"""Phase 2 pipeline: calibrate the copper -> steel link, route copper, run the steel model."""

from __future__ import annotations

from dataclasses import dataclass

from metalwheels.copper.stockflow import StockFlowResult
from metalwheels.registry import Registry
from metalwheels.steel import calibrate, demand, link


@dataclass
class CoupledSteel:
    calibration: calibrate.LinkCalibration
    link: link.LinkResult
    projection: demand.SteelProjection


def run(cu: StockFlowResult, reg: Registry) -> CoupledSteel:
    """Steel run over the copper run's years, with attached copper from the copper module."""
    cal = calibrate.calibrate_shares(cu, reg)
    lk = link.route(cu, cal.phi)
    sp = demand.project(cu.years, reg, lk.to_steel.to_numpy())
    return CoupledSteel(cal, lk, sp)
