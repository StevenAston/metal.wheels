"""Task 1.2: dynamic in-use stock model for copper (supply-driven, historical).

Per year *t*, with *M(t)* = world mine production (Task 1.1; USGS "recoverable" copper, i.e. Cu in
concentrate plus SX-EW, which corresponds to "concentrated ore" in Helbig 2020 Fig. S11):

1. End-of-life outflow by product group from past inflows and normal lifetime distributions
   (sigma = 15 % of the mean):  E_g(t) = sum_a I_g(t - a) * pmf_g(a),  a >= 1.
2. EoL partition with collection c and processing p:
   old scrap O = c p E; other scrap markets X = c (1 - p) E; not recycled (1 - c) E.
3. Scrap pool W = O + N (old + new scrap). A share w is low quality and is re-smelted with
   concentrate; the rest is melted directly by semis makers.
4. Refined copper R = (M + w W)(1 - s); fabrication input F = R + (1 - w) W;
   semis = F (1 - f); final goods by group G_g = eta_g * share_g * semis;
   new scrap N = r (semis - sum G_g); manufacturing loss = (1 - r)(semis - sum G_g).
5. In-use dissipation d at entry: inflow I_g = G_g (1 - d); stock S_g += I_g - E_g.

Because new scrap N depends on semis, which depends on the scrap pool W that contains N, steps
3-4 are solved in closed form each year:  W = (O + k M (1 - s)) / (1 - k (1 - w s)),
with k = (1 - f) r (1 - eta_bar) and eta_bar = sum_g share_g * eta_g.

Initial condition: zero in-use stock at the start year (pre-1900 copper is ignored; assumption).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from metalwheels.registry import Registry

CATEGORIES = ("construction", "infrastructure", "industry", "transport", "consumer")

#: Flow columns in the annual results (Mt Cu/yr unless noted).
FLOW_COLUMNS = (
    "mine",
    "smelter_input",
    "smelting_loss",
    "refined",
    "scrap_pool",
    "scrap_to_smelter",
    "direct_melt",
    "fabrication_input",
    "semis_loss",
    "semis",
    "final_goods",
    "new_scrap",
    "manufacturing_loss",
    "dissipation",
    "inflow_to_use",
    "eol",
    "old_scrap",
    "other_scrap_markets",
    "not_recycled",
    "stock",
    "primary_refined",
    "secondary_supply",
    "target_stock",
    "demand_driven",
)


@dataclass(frozen=True)
class CopperParams:
    """Numeric parameters for the stock-flow engine (built from the registry)."""

    groups: tuple[str, ...]
    category: tuple[str, ...]
    share: np.ndarray  # semis share per product group, sums to 1
    lifetime: np.ndarray  # mean lifetime per group (a)
    fab_eff: np.ndarray  # fabrication efficiency per group
    sigma_frac: float
    collection: float
    processing: float
    smelting_loss: float
    semis_loss: float
    new_scrap_capture: float
    scrap_to_smelter: float
    dissipation: float

    @classmethod
    def from_registry(cls, reg: Registry, **overrides: float) -> CopperParams:
        groups = tuple(
            sorted(k.removeprefix("cu.lifetime.") for k in reg if k.startswith("cu.lifetime."))
        )
        category = tuple(g.split("_", 1)[0] for g in groups)
        unknown = set(category) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"product groups with unknown category prefix: {sorted(unknown)}")
        counts = {c: category.count(c) for c in CATEGORIES}
        share = np.array([reg.value(f"cu.share.{c}") / counts[c] for c in category])
        if not np.isclose(share.sum(), 1.0, atol=1e-9):
            raise ValueError(f"end-use shares sum to {share.sum():.6f}, expected 1")
        vals = {
            "sigma_frac": reg.value("cu.lifetime_sigma_fraction"),
            "collection": reg.value("cu.eol_collection_rate"),
            "processing": reg.value("cu.eol_processing_rate"),
            "smelting_loss": reg.value("cu.smelting_loss_fraction"),
            "semis_loss": reg.value("cu.semis_loss_fraction"),
            "new_scrap_capture": reg.value("cu.new_scrap_capture"),
            "scrap_to_smelter": reg.value("cu.scrap_to_smelter_share"),
            "dissipation": reg.value("cu.in_use_dissipation_fraction"),
        }
        bad = set(overrides) - set(vals)
        if bad:
            raise ValueError(f"unknown overrides: {sorted(bad)}")
        vals.update(overrides)
        return cls(
            groups=groups,
            category=category,
            share=share,
            lifetime=np.array([reg.value(f"cu.lifetime.{g}") for g in groups]),
            fab_eff=np.array([reg.value(f"cu.fab_efficiency.{g}") for g in groups]),
            **vals,
        )

    @property
    def eta_bar(self) -> float:
        return float(self.share @ self.fab_eff)


def lifetime_pmf(mean: np.ndarray, sigma_frac: float, max_age: int) -> np.ndarray:
    """Discrete retirement probabilities, shape (groups, max_age), for ages 1..max_age.

    Normal distribution discretised on [a - 0.5, a + 0.5); mass below age 1.5 is assigned to age 1
    and each row is renormalised to sum to exactly 1 (no copper stays in use forever).
    """
    ages = np.arange(1, max_age + 1)
    sd = sigma_frac * mean[:, None]
    cdf = norm.cdf((ages[None, :] + 0.5 - mean[:, None]) / sd)
    pmf = np.diff(cdf, axis=1, prepend=0.0)
    return pmf / pmf.sum(axis=1, keepdims=True)


@dataclass
class StockFlowResult:
    years: np.ndarray
    flows: pd.DataFrame  # index year, FLOW_COLUMNS (Mt)
    stock_by_group: pd.DataFrame  # index year, columns groups (Mt)
    eol_by_group: pd.DataFrame
    inflow_by_group: pd.DataFrame

    def balance_residual(self) -> pd.Series:
        """Whole-system relative mass-balance residual per year:
        mine - (losses + Δstock), divided by throughput (mine + EoL).

        Throughput rather than mine output is the denominator so the check stays meaningful in
        years when recycling alone supplies demand and mine output is zero."""
        f = self.flows
        d_stock = f["stock"].diff().fillna(f["stock"].iloc[0])
        out = (
            f["smelting_loss"]
            + f["semis_loss"]
            + f["manufacturing_loss"]
            + f["dissipation"]
            + f["other_scrap_markets"]
            + f["not_recycled"]
            + d_stock
        )
        return (f["mine"] - out) / (f["mine"] + f["eol"])

    def stock_residual(self) -> pd.Series:
        """In-use stock balance (inflow - outflow - Δstock) relative to inflow, per year."""
        f = self.flows
        d_stock = f["stock"].diff().fillna(f["stock"].iloc[0])
        return (f["inflow_to_use"] - f["eol"] - d_stock) / f["inflow_to_use"]


def run(
    years: np.ndarray,
    mine: np.ndarray,
    prm: CopperParams,
    max_age: int = 200,
    target_stock: np.ndarray | None = None,
    collection: np.ndarray | None = None,
    processing: np.ndarray | None = None,
) -> StockFlowResult:
    """Run the stock-flow model. ``mine`` in Mt Cu/yr aligned with ``years``.

    Years where ``mine`` is finite are supply-driven. Years where ``mine`` is NaN are
    demand-driven: mine output is solved so that the in-use stock reaches ``target_stock``
    (Mt) that year, given end-of-life returns and the scrap loop. If recycling alone would
    overshoot the target, mine output is set to zero and the stock is allowed to exceed it.

    ``collection`` / ``processing`` optionally give per-year EoL rates (policy scenarios);
    otherwise the scalar values in ``prm`` apply to every year.
    """
    years = np.asarray(years)
    mine = np.asarray(mine, dtype=float)
    if target_stock is not None:
        target_stock = np.asarray(target_stock, dtype=float)
        if target_stock.shape != mine.shape:
            raise ValueError("target_stock must align with mine")
    demand_years = np.isnan(mine)
    if demand_years.any() and target_stock is None:
        raise ValueError("NaN mine values require target_stock")
    T, G = len(years), len(prm.groups)
    pmf = lifetime_pmf(prm.lifetime, prm.sigma_frac, max_age)  # (G, A)
    A = pmf.shape[1]

    inflow = np.zeros((T, G))
    eol = np.zeros((T, G))
    stock = np.zeros((T, G))
    flows = np.zeros((T, len(FLOW_COLUMNS)))
    col = {name: i for i, name in enumerate(FLOW_COLUMNS)}

    T_ = len(years)
    c_t = np.full(T_, prm.collection) if collection is None else np.asarray(collection, float)
    p_t = np.full(T_, prm.processing) if processing is None else np.asarray(processing, float)
    if c_t.shape != (T_,) or p_t.shape != (T_,):
        raise ValueError("collection/processing must align with years")
    if ((c_t < 0) | (c_t > 1) | (p_t < 0) | (p_t > 1)).any():
        raise ValueError("collection/processing rates must lie in [0, 1]")
    s, f = prm.smelting_loss, prm.semis_loss
    r, w, d = prm.new_scrap_capture, prm.scrap_to_smelter, prm.dissipation
    k = (1 - f) * r * (1 - prm.eta_bar)
    prev = np.zeros(G)

    for t in range(T):
        # 1. EoL from past inflows (ages 1..A): inflow[t-1], inflow[t-2], ... against pmf[:, 0..]
        n = min(t, A)
        if n:
            past = inflow[t - n : t][::-1]  # (n, G): age 1 first
            eol[t] = np.einsum("ag,ga->g", past, pmf[:, :n])
        E = eol[t].sum()
        # 2. partition
        c, p = c_t[t], p_t[t]
        old, other, notrec = c * p * E, c * (1 - p) * E, (1 - c) * E
        # 3-4. closed-form scrap pool and fabrication chain
        if demand_years[t]:
            M = _required_mine(target_stock[t] - prev.sum() + E, old, prm)
        else:
            M = mine[t]
        W = (old + k * M * (1 - s)) / (1 - k * (1 - w * s))
        smelter_in = M + w * W
        refined = smelter_in * (1 - s)
        fab_in = refined + (1 - w) * W
        semis = fab_in * (1 - f)
        goods = prm.fab_eff * prm.share * semis
        residue = semis - goods.sum()
        new = r * residue
        # 5. use phase
        inflow[t] = goods * (1 - d)
        stock[t] = prev + inflow[t] - eol[t]
        prev = stock[t]

        row = flows[t]
        row[col["mine"]] = M
        row[col["smelter_input"]] = smelter_in
        row[col["smelting_loss"]] = smelter_in * s
        row[col["refined"]] = refined
        row[col["scrap_pool"]] = W
        row[col["scrap_to_smelter"]] = w * W
        row[col["direct_melt"]] = (1 - w) * W
        row[col["fabrication_input"]] = fab_in
        row[col["semis_loss"]] = fab_in * f
        row[col["semis"]] = semis
        row[col["final_goods"]] = goods.sum()
        row[col["new_scrap"]] = new
        row[col["manufacturing_loss"]] = residue - new
        row[col["dissipation"]] = goods.sum() * d
        row[col["inflow_to_use"]] = inflow[t].sum()
        row[col["eol"]] = E
        row[col["old_scrap"]] = old
        row[col["other_scrap_markets"]] = other
        row[col["not_recycled"]] = notrec
        row[col["stock"]] = stock[t].sum()
        row[col["primary_refined"]] = M * (1 - s)
        row[col["secondary_supply"]] = w * W * (1 - s) + (1 - w) * W
        row[col["target_stock"]] = target_stock[t] if target_stock is not None else np.nan
        row[col["demand_driven"]] = float(demand_years[t])

        if not np.isclose(W, old + new, rtol=1e-12, atol=1e-15):
            raise AssertionError(f"scrap pool closure failed in {years[t]}: {W} vs {old + new}")

    idx = pd.Index(years, name="year")
    return StockFlowResult(
        years=years,
        flows=pd.DataFrame(flows, index=idx, columns=list(FLOW_COLUMNS)),
        stock_by_group=pd.DataFrame(stock, index=idx, columns=list(prm.groups)),
        eol_by_group=pd.DataFrame(eol, index=idx, columns=list(prm.groups)),
        inflow_by_group=pd.DataFrame(inflow, index=idx, columns=list(prm.groups)),
    )


def _required_mine(required_inflow: float, old_scrap: float, prm: CopperParams) -> float:
    """Mine output (Mt) that delivers ``required_inflow`` to use, given this year's old scrap.

    Inverts steps 3-5: goods = I / (1 - d); semis = goods / eta_bar; N = r (semis - goods);
    W = O + N; F = semis / (1 - f); R = F - (1 - w) W; M = R / (1 - s) - w W. Clamped at 0.
    """
    inflow = max(required_inflow, 0.0)
    goods = inflow / (1 - prm.dissipation)
    semis = goods / prm.eta_bar
    W = old_scrap + prm.new_scrap_capture * (semis - goods)
    refined = semis / (1 - prm.semis_loss) - (1 - prm.scrap_to_smelter) * W
    M = refined / (1 - prm.smelting_loss) - prm.scrap_to_smelter * W
    return max(M, 0.0)


def run_historical(reg: Registry | None = None, **overrides: float) -> StockFlowResult:
    """Convenience: registry + Task 1.1 production series, 1900-2025."""
    from metalwheels import registry as _registry
    from metalwheels.copper import production

    reg = reg or _registry.load()
    series = production.load()
    years = np.array(sorted(series))
    return run(
        years, np.array([series[y] for y in years]), CopperParams.from_registry(reg, **overrides)
    )
