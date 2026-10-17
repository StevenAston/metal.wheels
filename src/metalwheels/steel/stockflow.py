"""Task 2.1: steel in-use stock by sector, with copper carried through the scrap cycle.

Sectors (Daehn et al. 2017; Pauliuk et al. 2013): transport, industrial equipment, construction,
metal goods. Per year *t*:

1. Final steel products F(t) (Mt) enter use, split by fixed sector shares. Historically
   F = product_yield x crude steel; in demand-driven years F is solved so the total in-use stock
   reaches a target.
2. End-of-life steel E_g(t) and the copper it carries in bulk B_g(t) follow from past inflows
   and normal lifetimes (sigma a fixed fraction of the mean). Each cohort keeps the copper
   concentration it was made with, so copper put into steel today returns decades later. This
   is the cross-cycle accumulation that Daehn et al. approximate with a fixed 75-year
   correction.
3. A share rho of end-of-life steel is recovered as scrap. Recovered scrap carries its bulk
   copper plus copper attached by co-shredding, A_g(t), supplied by the copper module through
   the Task 2.2 link. Unrecovered steel goes to obsolete stocks or landfill with its bulk copper.
4. Iron sources needed = (1 + lambda) F, split across sectors and tolerance bands. Scrap from
   the four sectors and primary iron are allocated to those sinks by the Task 2.3 LP
   (``dilution.allocate``), run on band totals. The LP gives primary iron, scrap that cannot
   be used, and the copper concentration of every product. Unusable scrap is stockpiled and
   not offered again (assumption).
5. The lambda losses carry a share pi (``st.cu_to_process_losses``) of the melt copper
   concentration. With pi = 0 (default) copper stays with the steel, since it is more noble
   than iron and does not go to slag, so tolerances apply to product mass (Daehn Fig. 2); with
   pi = 1 concentration is preserved through the losses (Daehn Fig. 3).

Copper balance of the steel system, checked every year:
    A + c_p P = Δ(Cu in stock) + Cu to landfill + Cu in stockpiled scrap + Cu in process losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metalwheels.copper.stockflow import lifetime_pmf
from metalwheels.registry import Registry
from metalwheels.steel import dilution
from metalwheels.steel.link import SECTORS

FLOW_COLUMNS = (
    "products",
    "iron_sources",
    "primary_iron",
    "mass_only_primary",
    "copper_induced_primary",
    "eol",
    "scrap_recovered",
    "scrap_used",
    "scrap_unused",
    "surplus_scrap",
    "unusable_scrap",
    "not_recovered",
    "stock",
    "stockpile",
    "cu_attached",
    "cu_eol_bulk",
    "cu_scrap",
    "cu_primary",
    "cu_products",
    "cu_process_loss",
    "cu_landfill",
    "cu_stockpiled",
    "cu_stock",
    "cu_supply_aggregate",
    "cu_tolerable_aggregate",
    "scrap_cu_mean",
    "secondary_cu",
    "product_cu_mean",
    "target_stock",
    "demand_driven",
)


@dataclass(frozen=True)
class SteelParams:
    sectors: tuple[str, ...]
    bands: tuple[str, ...]
    share: np.ndarray  # (S,)
    lifetime: np.ndarray  # (S,) years
    sigma_frac: float
    recovery: float
    product_yield: float
    loss_allowance: float
    cu_to_losses: float  # share of melt copper concentration carried by process losses
    allocation: str  # "mixed" or "pinch" (see dilution.allocate)
    primary_cu: float
    tolerance: np.ndarray  # (B,)
    band_share: np.ndarray  # (S, B), rows sum to 1

    @classmethod
    def from_registry(cls, reg: Registry, **overrides) -> SteelParams:
        bands = tuple(
            sorted(
                (k.removeprefix("st.tolerance.") for k in reg if k.startswith("st.tolerance.")),
                reverse=True,
            )
        )
        tol = np.array([reg.value(f"st.tolerance.{b}") for b in bands])
        band_share = np.array(
            [
                [reg.value(f"st.band_share.{s}.{b}") if f"st.band_share.{s}.{b}" in reg else 0.0
                 for b in bands]
                for s in SECTORS
            ]
        )  # fmt: skip
        vals = {
            "sigma_frac": reg.value("st.lifetime_sigma_fraction"),
            "recovery": reg.value("st.eol_recovery_rate"),
            "product_yield": reg.value("st.product_yield"),
            "loss_allowance": reg.value("st.process_loss_allowance"),
            "cu_to_losses": reg.value("st.cu_to_process_losses"),
            "allocation": "pinch" if reg.value("st.allocation_pinch") >= 0.5 else "mixed",
            "primary_cu": reg.value("st.primary_cu"),
            "share": np.array([reg.value(f"st.share.{s}") for s in SECTORS]),
            "lifetime": np.array([reg.value(f"st.lifetime.{s}") for s in SECTORS]),
            "band_share": band_share,
        }
        bad = set(overrides) - set(vals)
        if bad:
            raise ValueError(f"unknown overrides: {sorted(bad)}")
        vals.update(overrides)
        prm = cls(sectors=SECTORS, bands=bands, tolerance=tol, **vals)
        prm.validate()
        return prm

    def validate(self) -> None:
        if not np.isclose(self.share.sum(), 1.0, atol=1e-6):
            raise ValueError(f"sector shares sum to {self.share.sum():.6f}")
        if not np.allclose(self.band_share.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("band shares must sum to 1 for every sector")
        if np.any(self.tolerance <= self.primary_cu):
            raise ValueError("tolerances must exceed the primary copper concentration")


@dataclass
class SteelResult:
    years: np.ndarray
    flows: pd.DataFrame
    stock_by_sector: pd.DataFrame
    eol_by_sector: pd.DataFrame
    scrap_cu_by_sector: pd.DataFrame  # concentration (mass fraction)
    product_cu_by_sector: pd.DataFrame  # concentration of steel entering use
    unusable_by_sector: pd.DataFrame  # Mt/yr

    def copper_residual(self) -> pd.Series:
        """Steel-system copper balance residual, relative to copper inputs, per year."""
        f = self.flows
        d_stock = f["cu_stock"].diff().fillna(f["cu_stock"].iloc[0])
        inputs = f["cu_attached"] + f["cu_primary"]
        outputs = d_stock + f["cu_landfill"] + f["cu_stockpiled"] + f["cu_process_loss"]
        return (inputs - outputs) / inputs.where(inputs > 0, 1.0)

    def iron_residual(self) -> pd.Series:
        """(primary + scrap used - iron sources) / iron sources, per year."""
        f = self.flows
        src = f["iron_sources"].where(f["iron_sources"] > 0, 1.0)
        return (f["primary_iron"] + f["scrap_used"] - f["iron_sources"]) / src

    def crossover_year(self, kind: str = "lp") -> int | None:
        """First year copper makes scrap unusable (``lp``) or aggregate copper supply exceeds
        aggregate tolerance (``aggregate``, the Daehn et al. 2017 Fig. 2 test)."""
        f = self.flows
        if kind == "lp":
            hit = f.index[f["unusable_scrap"] > 1e-6]
        elif kind == "aggregate":
            hit = f.index[f["cu_supply_aggregate"] > f["cu_tolerable_aggregate"]]
        else:
            raise ValueError(kind)
        return int(hit[0]) if len(hit) else None


def run(
    years: np.ndarray,
    products: np.ndarray,
    prm: SteelParams,
    attached_cu: np.ndarray | None = None,
    target_stock: np.ndarray | None = None,
    max_age: int = 300,
) -> SteelResult:
    """Run the steel stock model.

    ``products`` (Mt/yr) is supply-driven where finite; NaN years are demand-driven towards
    ``target_stock`` (Mt): either a total per year (inflow split by the fixed sector shares) or
    a (years, sectors) array of sector targets. ``attached_cu`` is (years, sectors) Mt Cu/yr
    from the copper link (zeros if omitted)."""
    years = np.asarray(years)
    products = np.asarray(products, dtype=float)
    T, S, B = len(years), len(prm.sectors), len(prm.bands)
    demand_years = np.isnan(products)
    if demand_years.any() and target_stock is None:
        raise ValueError("NaN product values require target_stock")
    by_sector = target_stock is not None and np.ndim(target_stock) == 2
    if by_sector and np.shape(target_stock) != (T, S):
        raise ValueError(f"sector targets must have shape {(T, S)}")
    A_cu = np.zeros((T, S)) if attached_cu is None else np.asarray(attached_cu, dtype=float)
    if A_cu.shape != (T, S):
        raise ValueError(f"attached_cu must have shape {(T, S)}")
    pmf = lifetime_pmf(prm.lifetime, prm.sigma_frac, max_age)
    n_age = pmf.shape[1]
    lam, rho, c_p = prm.loss_allowance, prm.recovery, prm.primary_cu
    # Copper in the charge that ends up in products, and the tolerance expressed per tonne of
    # charge, given the share of melt concentration carried by the process losses.
    pi_loss = prm.cu_to_losses
    to_product = (1 + lam - pi_loss * lam) / (1 + lam)
    tol_charge = prm.tolerance / (1 + lam - pi_loss * lam)
    c_p = prm.primary_cu / (1 + lam - pi_loss * lam)  # primary steel at 0.02 % in products

    inflow = np.zeros((T, S))
    inflow_cu = np.zeros((T, S))
    eol = np.zeros((T, S))
    stock = np.zeros((T, S))
    scrap_c = np.zeros((T, S))
    prod_c = np.zeros((T, S))
    unusable = np.zeros((T, S))
    flows = np.zeros((T, len(FLOW_COLUMNS)))
    col = {k: i for i, k in enumerate(FLOW_COLUMNS)}
    prev, prev_cu, stockpile = np.zeros(S), np.zeros(S), 0.0

    for t in range(T):
        n = min(t, n_age)
        if n:
            eol[t] = np.einsum("ag,ga->g", inflow[t - n : t][::-1], pmf[:, :n])
            b_cu = np.einsum("ag,ga->g", inflow_cu[t - n : t][::-1], pmf[:, :n])
        else:
            b_cu = np.zeros(S)
        E = eol[t].sum()
        if not demand_years[t]:
            F = products[t]
            inflow[t] = prm.share * F
        elif by_sector:
            inflow[t] = np.maximum(target_stock[t] - prev + eol[t], 0.0)
            F = inflow[t].sum()
        else:
            F = max(target_stock[t] - prev.sum() + E, 0.0)
            inflow[t] = prm.share * F

        rec = rho * eol[t]
        cu_rec = rho * b_cu + A_cu[t]
        conc = np.divide(cu_rec, rec, out=np.zeros(S), where=rec > 0)
        scrap_c[t] = conc
        sinks = ((1 + lam) * inflow[t])[:, None] * prm.band_share  # (S, B)
        band_mass = sinks.sum(axis=0)
        # Sinks with the same tolerance are interchangeable, so the LP runs on band totals and
        # each band's copper is shared across sectors in proportion to mass.
        alloc = dilution.allocate(rec, conc, band_mass, tol_charge, c_p, prm.allocation)
        band_c = np.divide(
            alloc.sink_copper, band_mass, out=np.zeros(B), where=band_mass > 0
        )  # fmt: skip
        cu_charge = sinks * band_c[None, :]
        inflow_cu[t] = cu_charge.sum(axis=1) * to_product
        prod_c[t] = np.divide(inflow_cu[t], inflow[t], out=np.zeros(S), where=inflow[t] > 0)
        cu_loss = cu_charge.sum() * (1 - to_product)
        unusable[t] = alloc.scrap_unused
        cu_unused = float(conc @ alloc.scrap_unused)
        stockpile += alloc.scrap_unused.sum()

        stock[t] = prev + inflow[t] - eol[t]
        stock_cu = prev_cu + inflow_cu[t] - b_cu
        prev, prev_cu = stock[t], stock_cu
        supply, tolerable = dilution.copper_balance(
            rec, conc, band_mass, tol_charge, c_p
        )  # fmt: skip

        r = flows[t]
        r[col["products"]] = F
        r[col["iron_sources"]] = (1 + lam) * F
        r[col["primary_iron"]] = alloc.primary_total
        r[col["mass_only_primary"]] = alloc.mass_only_primary
        r[col["copper_induced_primary"]] = alloc.copper_induced_primary
        r[col["eol"]] = E
        r[col["scrap_recovered"]] = rec.sum()
        r[col["scrap_used"]] = alloc.scrap_used.sum()
        r[col["scrap_unused"]] = alloc.scrap_unused.sum()
        r[col["surplus_scrap"]] = alloc.surplus_scrap
        r[col["unusable_scrap"]] = alloc.unusable_scrap
        r[col["not_recovered"]] = (1 - rho) * E
        r[col["stock"]] = stock[t].sum()
        r[col["stockpile"]] = stockpile
        r[col["cu_attached"]] = A_cu[t].sum()
        r[col["cu_eol_bulk"]] = b_cu.sum()
        r[col["cu_scrap"]] = cu_rec.sum()
        r[col["cu_primary"]] = c_p * alloc.primary_total
        r[col["cu_products"]] = inflow_cu[t].sum()
        r[col["cu_process_loss"]] = cu_loss
        r[col["cu_landfill"]] = (1 - rho) * b_cu.sum()
        r[col["cu_stockpiled"]] = cu_unused
        r[col["cu_stock"]] = stock_cu.sum()
        r[col["cu_supply_aggregate"]] = supply
        r[col["cu_tolerable_aggregate"]] = tolerable
        r[col["scrap_cu_mean"]] = cu_rec.sum() / rec.sum() if rec.sum() > 0 else 0.0
        used = alloc.scrap_used.sum()
        r[col["secondary_cu"]] = float(conc @ alloc.scrap_used) / used if used > 0 else np.nan
        r[col["product_cu_mean"]] = inflow_cu[t].sum() / F if F > 0 else np.nan
        if target_stock is None:
            r[col["target_stock"]] = np.nan
        else:
            r[col["target_stock"]] = np.sum(target_stock[t]) if by_sector else target_stock[t]
        r[col["demand_driven"]] = float(demand_years[t])

    idx = pd.Index(years, name="year")

    def frame(a):
        return pd.DataFrame(a, index=idx, columns=list(prm.sectors))

    return SteelResult(
        years=years,
        flows=pd.DataFrame(flows, index=idx, columns=list(FLOW_COLUMNS)),
        stock_by_sector=frame(stock),
        eol_by_sector=frame(eol),
        scrap_cu_by_sector=frame(scrap_c),
        product_cu_by_sector=frame(prod_c),
        unusable_by_sector=frame(unusable),
    )


def historical_products(reg: Registry, years: np.ndarray) -> np.ndarray:
    """Final steel products (Mt/yr) = product_yield x crude steel, for historical ``years``."""
    from metalwheels.steel import production

    crude = production.load()
    y = reg.value("st.product_yield")
    return np.array([crude[int(t)] * y for t in years])
