"""Task 4.3: carbon-cup lever - Ellingham eligibility and the scenario stream.

**Idea.** Carbon-reducible metals are housed in burnable, separable carbon sub-matrices ("cups").
At end of life the product is burnt, the metal concentrates in the ash as oxide, and carbon
reduces it back to metal. The CO2 from the matrix and from the reductant is a concentrated
stream that can be captured; part of the carbon may be biogenic.

**Eligibility (Ellingham).** A metal qualifies if carbon can reduce its most stable oxide at or
below a working temperature ``cc.t_max``: for some T <= t_max,

    dG(2C + O2 -> 2CO, T) < dG(oxide formation per mol O2, T).

Oxide lines come from NIST-JANAF dfG(T) where JANAF has the oxide, otherwise from the NBS 1982
298 K values with the Ulich approximation dG(T) = dH298 - T dS298 (constant dH, dS; phase changes
ignored; for PdO, Rh2O3, Pt3O4, IrO2 and RuO2, whose S298 NBS does not give, dS = -S(O2) per mol
O2). The CO line is JANAF. Gold forms no stable oxygen compound and always qualifies.
``data/processed/oxide_thermo.csv`` lists every value and its page; the Ulich rows are checked
against JANAF in the tests.

**Lever (copper only, the stock-modelled metal).** A share s(t) of end-of-life copper is in cups.
That share is collected at ``cc.collection`` and processed at ``cc.processing`` instead of the
scenario's c and p, so effective rates are (1 - s) c + s c_cc and (1 - s) p + s p_cc. Raising
effective processing shrinks the "other scrap markets" flow, which is the copper that Phase 2
routes into steel scrap, and raises old scrap, which lowers primary demand. The cascade then
sees a lower primary-copper share and a lower primary-iron share.

**CO2 stream** (Mt CO2/yr): carbon in the collected cups (``cc.carbon_per_metal`` kg C per kg Cu)
plus reductant carbon (one C per O in CuO, the highest oxide, which is the conservative case) for
the recovered copper, all burnt to CO2. A share ``cc.capture_fraction`` is captured; a share
``cc.biogenic_share`` of the carbon is biogenic. Net atmospheric CO2 = fossil CO2 emitted minus
biogenic CO2 captured.

**Matrix energy** (``cc.matrix_net_energy``, MJ per kg C, default 0): energy to make the matrix net
of energy recovered by burning it, added to the copper column of A(t) per tonne of copper
supplied. Timing is simplified: the matrix energy for the copper reaching end of life in year t is
booked in year t.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from metalwheels.logging_setup import get_logger
from metalwheels.registry import Registry

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[3]
THERMO_CSV = REPO / "data" / "processed" / "oxide_thermo.csv"
JANAF_DIR = REPO / "data" / "raw" / "janaf"

#: S(O2, g, 298.15 K), J/(mol K): wagman_1982_nbs p. 2-37.
S_O2 = 205.138
#: Molar masses, g/mol (as used by the NBS tables, wagman_1982_nbs; CO2 = C + 2 O).
M_C, M_O, M_CU = 12.011, 15.9994, 63.54
M_CO2 = M_C + 2 * M_O
CO_TABLE = "C-093"
T_GRID = np.arange(298.0, 3001.0, 1.0)

#: Groups used by the done-condition test.
PGMS = ("Pt", "Pd", "Rh", "Ir", "Ru", "Os")
RARE_EARTHS = ("La", "Ce", "Nd", "Y")


# ------------------------------------------------------------------------------- thermodynamics
def read_janaf(path: Path) -> pd.DataFrame:
    """T (K) and delta-f G (kJ/mol) rows of a JANAF text table (transition rows skipped)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[2:]:
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        try:
            t, g = float(parts[0]), float(parts[6])
        except ValueError:
            continue
        rows.append((t, g))
    df = pd.DataFrame(rows, columns=["T", "dfG"]).drop_duplicates("T", keep="last")
    return df.sort_values("T").reset_index(drop=True)


@dataclass(frozen=True)
class Oxide:
    element: str
    compound: str
    n_metal: float
    n_o: float
    method: str
    janaf_table: str | None
    dhf: float | None  # kJ/mol compound
    s_compound: float | None
    s_metal: float | None
    page: str | None
    note: str

    def dg_o2(self, t: np.ndarray, janaf: dict[str, pd.DataFrame]) -> np.ndarray:
        """dG of formation per mol O2 (kJ) at temperatures ``t`` (K)."""
        t = np.asarray(t, float)
        if self.method == "noble":
            return np.full_like(t, np.inf)
        per_o2 = 2.0 / self.n_o
        if self.method == "janaf":
            tab = janaf[self.janaf_table]
            inside = (t >= tab["T"].iloc[0]) & (t <= tab["T"].iloc[-1])
            # NaN outside the tabulated range: no extrapolation
            return np.where(inside, per_o2 * np.interp(t, tab["T"], tab["dfG"]), np.nan)
        if self.method in ("nbs_ulich", "nbs_check"):
            ds = self.s_compound - self.n_metal * self.s_metal - 0.5 * self.n_o * S_O2
        elif self.method == "nbs_ulich_approx_s":
            ds = -0.5 * self.n_o * S_O2
        else:
            raise ValueError(f"unknown method {self.method!r}")
        return per_o2 * (self.dhf - t * ds / 1000.0)


class Thermo:
    """Oxide lines, the CO line and carbon-reducibility."""

    def __init__(self, csv_path: Path = THERMO_CSV, janaf_dir: Path = JANAF_DIR):
        df = pd.read_csv(csv_path, dtype={"janaf_table": str, "nbs_page": str, "note": str})
        self.oxides: list[Oxide] = []
        self.checks: list[Oxide] = []
        for r in df.itertuples(index=False):
            ox = Oxide(
                element=r.element,
                compound=r.compound,
                n_metal=float(r.n_metal),
                n_o=float(r.n_o),
                method=r.method,
                janaf_table=r.janaf_table if isinstance(r.janaf_table, str) else None,
                dhf=None if pd.isna(r.dHf_kJ) else float(r.dHf_kJ),
                s_compound=None if pd.isna(r.S_compound) else float(r.S_compound),
                s_metal=None if pd.isna(r.S_metal) else float(r.S_metal),
                page=r.nbs_page if isinstance(r.nbs_page, str) else None,
                note=r.note if isinstance(r.note, str) else "",
            )
            (self.checks if ox.method == "nbs_check" else self.oxides).append(ox)
        tables = {o.janaf_table for o in self.oxides + self.checks if o.janaf_table} | {CO_TABLE}
        self.janaf = {t: read_janaf(janaf_dir / f"{t}.txt") for t in tables}

    @cached_property
    def elements(self) -> list[str]:
        return list(dict.fromkeys(o.element for o in self.oxides))

    def oxides_of(self, element: str) -> list[Oxide]:
        out = [o for o in self.oxides if o.element == element]
        if not out:
            raise KeyError(f"no oxide data for {element!r}")
        return out

    def dg_co(self, t) -> np.ndarray:
        """dG of 2C + O2 -> 2CO (kJ per mol O2), JANAF."""
        tab = self.janaf[CO_TABLE]
        return 2.0 * np.interp(np.asarray(t, float), tab["T"], tab["dfG"])

    def dg_o2(self, element: str, t) -> np.ndarray:
        """Most stable oxide line (lowest dG per mol O2) of ``element``."""
        lines = np.vstack([o.dg_o2(np.atleast_1d(t), self.janaf) for o in self.oxides_of(element)])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN beyond every table
            return np.nanmin(lines, axis=0)

    def onset_temperature(self, element: str) -> float:
        """Lowest T (K, 298-3000) at which carbon reduces the element's most stable oxide (only
        temperatures inside the oxide's tabulated range count)."""
        ok = self.dg_o2(element, T_GRID) > self.dg_co(T_GRID)
        return float(T_GRID[np.argmax(ok)]) if ok.any() else float("nan")

    def reducible(self, element: str, t_max: float) -> bool:
        t0 = self.onset_temperature(element)
        return bool(t0 == t0 and t0 <= t_max)

    def table(self, t_max: float) -> pd.DataFrame:
        rows = []
        for el in self.elements:
            oxs = self.oxides_of(el)
            line = self.dg_o2(el, np.array([t_max]))[0]
            stable = min(
                oxs, key=lambda o: np.nan_to_num(o.dg_o2(np.array([t_max]), self.janaf)[0], nan=1e9)
            )
            rows.append(
                {
                    "element": el,
                    "oxide": stable.compound,
                    "method": stable.method,
                    "dG_oxide_at_tmax": line,
                    "dG_CO_at_tmax": float(self.dg_co(t_max)),
                    "onset_K": self.onset_temperature(el),
                    "eligible": self.reducible(el, t_max),
                }
            )
        return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- scenario lever
@dataclass(frozen=True)
class CarbonCup:
    share: float
    start_year: int
    ramp_years: float
    collection: float
    processing: float
    carbon_per_metal: float
    capture_fraction: float
    biogenic_share: float
    matrix_net_energy: float  # MJ per kg C
    t_max: float
    metal: str = "Cu"

    def share_path(self, years: np.ndarray) -> np.ndarray:
        from metalwheels.copper import demand

        policy = {"start_year": self.start_year, "ramp_years": self.ramp_years}
        return self.share * demand.ramp_fraction(np.asarray(years), policy)


_CC_KEYS = {"share", "metal"}


def resolve_policy(
    policy: dict | None, reg: Registry, thermo: Thermo | None = None
) -> tuple[dict | None, CarbonCup | None]:
    """Turn a policy's ``carbon_cup`` block into effective collection/processing targets.

    Effective rates are linear in the cup share, and the share follows the policy's ramp, so the
    result is an ordinary ramped policy with targets (1 - S) x + S x_cc, where x is the policy's
    own target (or the registry base rate) and S the full cup share.
    """
    if not policy or "carbon_cup" not in policy:
        return policy, None
    block = dict(policy["carbon_cup"] or {})
    bad = set(block) - _CC_KEYS
    if bad:
        raise ValueError(f"unknown carbon_cup keys: {sorted(bad)}")
    metal = block.get("metal", "Cu")
    if metal != "Cu":
        raise ValueError("the carbon cup is applied to copper only (the stock-modelled metal)")
    share = float(block["share"])
    if not 0 <= share <= 1:
        raise ValueError(f"carbon_cup share must lie in [0, 1], got {share}")
    thermo = thermo or Thermo()
    t_max = reg.value("cc.t_max")
    if not thermo.reducible(metal, t_max):
        raise ValueError(f"{metal} is not carbon-reducible at {t_max} K")
    out = {k: v for k, v in policy.items() if k != "carbon_cup"}
    from metalwheels.copper.demand import LAST_HISTORICAL_YEAR

    spec = CarbonCup(
        share=share,
        start_year=int(policy.get("start_year", LAST_HISTORICAL_YEAR + 1)),
        ramp_years=float(policy.get("ramp_years", 0)),
        collection=reg.value("cc.collection"),
        processing=reg.value("cc.processing"),
        carbon_per_metal=reg.value("cc.carbon_per_metal"),
        capture_fraction=reg.value("cc.capture_fraction"),
        biogenic_share=reg.value("cc.biogenic_share"),
        matrix_net_energy=reg.value("cc.matrix_net_energy"),
        t_max=t_max,
    )
    for key, base_key, cc in (
        ("collection", "cu.eol_collection_rate", spec.collection),
        ("processing", "cu.eol_processing_rate", spec.processing),
    ):
        base = float(policy.get(key, reg.value(base_key)))
        out[key] = (1 - share) * base + share * cc
    return out, spec


#: Time-series columns written by :func:`stream` (all runs carry them; zero without a cup).
STREAM_COLUMNS = (
    "cc_share",
    "cc_eol_cu",
    "cc_collected_cu",
    "cc_recovered_cu",
    "cc_matrix_carbon",
    "cc_reductant_carbon",
    "cc_co2_total",
    "cc_co2_captured",
    "cc_co2_emitted_fossil",
    "cc_co2_emitted_biogenic",
    "cc_co2_net",
    "cc_matrix_energy_mj_per_t",
)


def stream(ts: pd.DataFrame, spec: CarbonCup | None) -> pd.DataFrame:
    """Carbon-cup flows (Mt Cu, Mt C, Mt CO2 per year) from a run's time series."""
    out = pd.DataFrame(0.0, index=ts.index, columns=list(STREAM_COLUMNS))
    if spec is None:
        return out
    s = spec.share_path(ts.index.to_numpy())
    eol = ts["eol"].to_numpy()
    cup = s * eol
    collected = cup * spec.collection
    recovered = collected * spec.processing
    matrix_c = collected * spec.carbon_per_metal
    reductant_c = recovered * (1.0 / M_CU) * M_C  # CuO + C -> Cu + CO, one C per Cu
    co2 = (matrix_c + reductant_c) * M_CO2 / M_C
    f, b = spec.capture_fraction, spec.biogenic_share
    out["cc_share"] = s
    out["cc_eol_cu"] = cup
    out["cc_collected_cu"] = collected
    out["cc_recovered_cu"] = recovered
    out["cc_matrix_carbon"] = matrix_c
    out["cc_reductant_carbon"] = reductant_c
    out["cc_co2_total"] = co2
    out["cc_co2_captured"] = f * co2
    out["cc_co2_emitted_fossil"] = (1 - f) * (1 - b) * co2
    out["cc_co2_emitted_biogenic"] = (1 - f) * b * co2
    out["cc_co2_net"] = (1 - f) * (1 - b) * co2 - f * b * co2
    fab = ts["fabrication_input"].to_numpy()
    # Mt C x MJ/kg C -> MJ (x 1e9), per t of copper supplied (Mt x 1e6)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(fab > 0, matrix_c * spec.matrix_net_energy * 1e3 / fab, 0.0)
    out["cc_matrix_energy_mj_per_t"] = e
    return out


# ------------------------------------------------------------------------------- comparison
YEARS = (2050, 2100, 2200, 2500)


def compare(base, cup) -> tuple[pd.DataFrame, dict[str, tuple[float, str]]]:
    """Scenario outputs: delta maintenance energy and delta rho per bracket, CO2 captured and
    emitted, effective processing rate. ``base`` and ``cup`` are :class:`RunOutput`."""
    from metalwheels.scenarios import BRACKETS

    tb, tc = base.timeseries, cup.timeseries
    df = pd.DataFrame(index=tc.index)
    out: dict[str, tuple[float, str]] = {}
    for br in BRACKETS:
        e_b, e_c = tb[f"cas_{br}_maintenance_energy_ej"], tc[f"cas_{br}_maintenance_energy_ej"]
        df[f"d_maintenance_energy_{br}"] = e_c - e_b
        df[f"d_rho_{br}"] = tc[f"cas_{br}_rho"] - tb[f"cas_{br}_rho"]
        both = e_b.notna() & e_c.notna()
        out[f"d_maintenance_energy_cum_{br}"] = (float((e_c - e_b)[both].sum()), "EJ")
        out[f"common_valid_until_{br}"] = (float(tc.index[both][-1]), "year")
        for y in YEARS:
            out[f"d_maintenance_energy_{br}_{y}"] = (
                float(df.loc[y, f"d_maintenance_energy_{br}"]),
                "EJ/yr",
            )
            out[f"d_rho_{br}_{y}"] = (float(df.loc[y, f"d_rho_{br}"]), "1")
        out[f"d_rho_max_{br}"] = (float(df[f"d_rho_{br}"].abs().max()), "1")
        for k in ("first_shortfall_year",):
            vb, vc = base.summary[f"{br}_{k}"][0], cup.summary[f"{br}_{k}"][0]
            out[f"{br}_{k}_base"] = (vb, "year")
            out[f"{br}_{k}_cup"] = (vc, "year")
    for c in (
        "cc_co2_captured",
        "cc_co2_emitted_fossil",
        "cc_co2_emitted_biogenic",
        "cc_co2_net",
        "cc_co2_total",
    ):
        df[c] = tc[c]
        out[f"{c}_cum"] = (float(tc[c].sum()), "Mt CO2")
        for y in YEARS:
            out[f"{c}_{y}"] = (float(tc.loc[y, c]), "Mt CO2/yr")
    df["eol_recycling_rate"] = cup.timeseries["old_scrap"] / cup.timeseries["eol"].where(
        cup.timeseries["eol"] > 0
    )
    df["st_cu_scrap_base"] = tb["st_scrap_cu_mean"]
    df["st_cu_scrap_cup"] = tc["st_scrap_cu_mean"]
    for kind in ("lp", "aggregate"):
        k = f"st_crossover_year_{kind}"
        out[f"{k}_base"] = (base.summary[k][0], "year")
        out[f"{k}_cup"] = (cup.summary[k][0], "year")
    out["st_cum_cu_penalty_base"] = base.summary["st_cum_cu_penalty_ej"]
    out["st_cum_cu_penalty_cup"] = cup.summary["st_cum_cu_penalty_ej"]
    out["eol_recycling_rate_end_base"] = base.summary["eol_recycling_rate_end"]
    out["eol_recycling_rate_end_cup"] = cup.summary["eol_recycling_rate_end"]
    return df, out
