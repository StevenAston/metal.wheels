"""Task 1.3: end-of-life fate and production-loss partition into receiving media.

Receiving media follow Helbig et al. 2020 Fig. S11 (2011, kt Cu):

* ``tailings_and_slag`` (ENV.1): mining losses to tailings + smelting losses
  (S11: 3,550 + 350 = 3,900). Tailings = M * tau / (1 - tau), where M is recoverable mine output
  and tau the share of Cu in ore lost to tailings (S11: 3,550 of 19,800 in ore).
* ``other_scrap_markets`` (WMR.2): collected-but-unprocessed EoL copper, semis losses and
  manufacturing losses (S11: 2,300 + 150 + 200 = 2,650). This copper is not lost to the
  environment: it travels inside steel, aluminium and other scrap streams, which is the
  cross-contamination route Phase 2 follows into steel.
* ``landfill`` (ENV.2): EoL copper not collected (landfill, incineration; S11: 3,600).
* ``environment`` (ENV.0): in-use dissipation (S11: 450).

Mass balance: Cu in ore = sum of receiving media + change in in-use stock.
"""

from __future__ import annotations

import pandas as pd

from metalwheels.copper.stockflow import StockFlowResult

MEDIA = ("tailings_and_slag", "other_scrap_markets", "landfill", "environment")


def partition(result: StockFlowResult, tailings_share: float) -> pd.DataFrame:
    """Annual flows to each receiving medium plus cumulative totals (Mt Cu).

    Columns: ``ore_cu``, ``tailings``, the four ``MEDIA`` flows, ``cum_<medium>`` for each,
    ``cum_ore_cu`` and ``stock``.
    """
    if not 0 <= tailings_share < 1:
        raise ValueError(f"tailings_share must be in [0, 1), got {tailings_share}")
    f = result.flows
    tau = tailings_share
    out = pd.DataFrame(index=f.index)
    out["ore_cu"] = f["mine"] / (1 - tau)
    out["tailings"] = out["ore_cu"] - f["mine"]
    out["tailings_and_slag"] = out["tailings"] + f["smelting_loss"]
    out["other_scrap_markets"] = (
        f["other_scrap_markets"] + f["semis_loss"] + f["manufacturing_loss"]
    )
    out["landfill"] = f["not_recycled"]
    out["environment"] = f["dissipation"]
    for m in (*MEDIA, "ore_cu"):
        out[f"cum_{m}"] = out[m].cumsum()
    out["stock"] = f["stock"]
    return out


def balance_residual(media: pd.DataFrame) -> pd.Series:
    """(ore Cu - media - Δstock) / ore Cu, per year."""
    d_stock = media["stock"].diff().fillna(media["stock"].iloc[0])
    return (media["ore_cu"] - media[list(MEDIA)].sum(axis=1) - d_stock) / media["ore_cu"]


def shares(media: pd.DataFrame, year: int) -> dict[str, float]:
    """Share of each receiving medium in that year's total losses."""
    row = media.loc[year, list(MEDIA)]
    return (row / row.sum()).to_dict()
