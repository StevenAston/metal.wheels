"""Task 2.2: copper -> steel link.

The copper module's "other scrap markets" flow (Helbig 2020 node WMR.2: copper that is collected
at end of life but ends up in the wrong scrap stream) is split by copper end-use category. A
share phi_g of the copper from the categories mapped to steel sector g becomes copper attached
to that sector's steel scrap (wiring, motors and the like co-shredded with steel). The rest goes
to other sinks (mainly aluminium and mixed scrap streams), which are not modelled further here.

Because collection and processing rates are the same for every product group in a given year,
the category split of "other scrap markets" equals the category split of end-of-life copper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metalwheels.copper.stockflow import StockFlowResult

SECTORS = ("transport", "industrial_equipment", "construction", "metal_goods")

#: Copper end-use categories feeding each steel sector.
CU_CATEGORY_MAP: dict[str, tuple[str, ...]] = {
    "transport": ("transport",),
    "industrial_equipment": ("industry",),
    "construction": ("construction", "infrastructure"),
    "metal_goods": ("consumer",),
}


def other_scrap_by_category(cu: StockFlowResult) -> pd.DataFrame:
    """Copper 'other scrap markets' (Mt/yr) by copper end-use category."""
    eol = cu.eol_by_group
    cats = sorted({g.split("_", 1)[0] for g in eol.columns})
    by_cat = pd.DataFrame(
        {c: eol[[g for g in eol.columns if g.split("_", 1)[0] == c]].sum(axis=1) for c in cats}
    )
    total = by_cat.sum(axis=1)
    share = by_cat.div(total.where(total > 0), axis=0).fillna(0.0)
    return share.mul(cu.flows["other_scrap_markets"], axis=0)


@dataclass
class LinkResult:
    other_scrap: pd.DataFrame  # copper other scrap markets by copper category (Mt/yr)
    to_steel: pd.DataFrame  # copper attached to steel scrap by steel sector (Mt/yr)
    to_other_sinks: pd.Series  # remainder (Mt/yr)

    def conservation_residual(self) -> pd.Series:
        """(copper leaving copper module - copper into steel - other sinks) / leaving."""
        out = self.other_scrap.sum(axis=1)
        resid = out - self.to_steel.sum(axis=1) - self.to_other_sinks
        return resid / out.where(out > 0, 1.0)


def route(cu: StockFlowResult, phi: np.ndarray) -> LinkResult:
    """Split copper 'other scrap markets' into steel-attached copper and other sinks.

    ``phi`` (one share per steel sector, in ``SECTORS`` order) normally comes from
    ``metalwheels.steel.calibrate.calibrate_shares``."""
    phi = np.asarray(phi, dtype=float)
    if phi.shape != (len(SECTORS),) or np.any((phi < 0) | (phi > 1)):
        raise ValueError("copper link shares must be one value in [0, 1] per sector")
    other = other_scrap_by_category(cu)
    to_steel = pd.DataFrame(index=other.index)
    for s, f in zip(SECTORS, phi, strict=True):
        cats = [c for c in CU_CATEGORY_MAP[s] if c in other.columns]
        to_steel[s] = f * other[cats].sum(axis=1)
    mapped = sorted({c for cs in CU_CATEGORY_MAP.values() for c in cs})
    unmapped = [c for c in other.columns if c not in mapped]
    if unmapped:
        raise ValueError(f"copper categories without a steel mapping: {unmapped}")
    rest = other.sum(axis=1) - to_steel.sum(axis=1)
    return LinkResult(other, to_steel, rest)
