"""Task 3.1-3.3: build A(t) from model state and compute the cascade time series.

Sectors (units of output): copper (t copper supplied to fabrication: refined plus directly melted
scrap), steel (t crude steel), aluminium (t primary Al), cement (t), ammonia (t NH3), water (m3
supplied), energy (MJ primary). Coefficients a_ij come from ``params/cascade.yaml`` and, where they
depend on the state of the model, from the Phase 1-2 run:

* copper energy: primary share x E(g, D) (Phase 1, per bracket) + secondary share x secondary
  energy. With ``cas.cu_energy_net_of_inputs`` = 1, the energy embodied in the copper column's
  modelled consumables is netted out, because Norgate's figures are gross (cradle-to-gate);
* copper steel, explosives and water scale with ore and rock moved per tonne of primary copper
  (Phase 1 grade and strip ratio);
* steel energy, explosives and water follow the primary/scrap split of the iron charge (Phase 2).
  This includes the primary iron forced in by copper contamination.

Everything else is constant. Final demand is held at its 2025 level, y = (I - A_2025) x_2025, with
x_2025 the observed world outputs. The cascade therefore reproduces 2025 output exactly, and later
changes come only from A(t). Maintenance energy is the energy output needed to deliver that same y.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metalwheels.cascade import leontief
from metalwheels.logging_setup import get_logger
from metalwheels.registry import Registry

log = get_logger(__name__)

SECTORS = ("copper", "steel", "aluminium", "cement", "ammonia", "water", "energy")
IDX = {s: i for i, s in enumerate(SECTORS)}
UNITS = ("t", "t", "t", "t", "t", "m3", "MJ")
BASE_YEAR = 2025

#: Justification for every structurally zero cell (input, output).
_CAPITAL = "capital equipment, excluded outside the energy sector by the agreed boundary"
ZERO_NOTES: dict[tuple[str, str], str] = {
    ("copper", "copper"): "own use netted out: the sector is defined net of internal scrap",
    ("copper", "steel"): "copper in steel is a contaminant handled in Phase 2, not an input",
    ("copper", "aluminium"): f"busbars and cabling in smelters are {_CAPITAL}",
    ("copper", "cement"): f"{_CAPITAL}",
    ("copper", "ammonia"): f"{_CAPITAL}",
    ("copper", "water"): f"pumps and motors are {_CAPITAL}",
    ("steel", "steel"): "internal (home and fabrication) scrap is netted out of the sector",
    ("steel", "aluminium"): f"cathode collector bars and potshells are {_CAPITAL}",
    ("steel", "cement"): "grinding-media wear in cement mills is below 0.1 kg/t (not sourced)",
    ("steel", "ammonia"): f"reformers and converters are {_CAPITAL}",
    ("steel", "water"): f"pipes and plant are {_CAPITAL}",
    ("aluminium", "copper"): "no aluminium consumable in copper production",
    ("aluminium", "steel"): (
        "deoxidation aluminium (about 1-2 kg/t steel by industry practice) is not yet sourced; "
        "candidate for a future cell"
    ),
    ("aluminium", "aluminium"): "internal scrap is netted out; sector is primary aluminium",
    ("aluminium", "cement"): "no aluminium consumable",
    ("aluminium", "ammonia"): "no aluminium consumable",
    ("aluminium", "water"): f"{_CAPITAL}",
    ("cement", "copper"): "paste backfill in some underground mines is not separable globally",
    ("cement", "steel"): "refractories are not cement",
    ("cement", "aluminium"): "no cement consumable",
    ("cement", "cement"): "no cement input to cement",
    ("cement", "ammonia"): "no cement consumable",
    ("cement", "water"): f"dams, pipes and treatment works are {_CAPITAL}",
    ("cement", "energy"): (
        "concrete in power plants, dams and foundations is counted in construction final "
        "demand; no energy-sector share was sourced"
    ),
    ("ammonia", "aluminium"): "bauxite is mostly strip-mined without blasting",
    ("ammonia", "ammonia"): "own use is inside the ammonia energy intensity",
    ("ammonia", "water"): "no ammonia consumable",
    ("water", "cement"): "cement-plant water use not sourced (small, mostly cooling)",
    ("water", "ammonia"): "not reported in the IEA Ammonia Technology Roadmap",
    ("water", "water"): "own use is inside the water-sector energy intensity",
}


#: Registry parameters behind every non-zero cell (input, output). Copper energy also uses the
#: Phase 1 energy and geology parameters, and steel energy the Phase 2 route energies.
CELL_PARAMS: dict[tuple[str, str], tuple[str, ...]] = {
    ("energy", "copper"): ("en.norgate.coarse.a", "en.norgate.coarse.b", "cas.cu.secondary_energy",
                           "cas.cu_energy_net_of_inputs"),
    ("steel", "copper"): ("cas.cu.grinding_steel_per_ore", "cas.cu.loading_wear_per_rock"),
    ("ammonia", "copper"): ("cas.powder_factor", "cas.ammonia_per_explosive"),
    ("water", "copper"): ("cas.cu.water_per_ore",),
    ("energy", "steel"): ("st.energy.bf_bof", "st.energy.dri_eaf", "st.energy.scrap_eaf",
                          "st.dri_share_of_primary"),
    ("ammonia", "steel"): ("cas.st.ore_per_primary_iron", "cas.st.rock_per_ore",
                           "cas.powder_factor", "cas.ammonia_per_explosive"),
    ("water", "steel"): ("cas.st.water_bf_bof", "cas.st.water_eaf"),
    ("energy", "aluminium"): ("cas.al.smelting_electricity", "cas.al.alumina_per_al",
                              "cas.al.alumina_energy", "cas.electricity_primary_factor"),
    ("water", "aluminium"): ("cas.al.water",),
    ("energy", "cement"): ("cas.ce.kiln_thermal", "cas.ce.clinker_ratio", "cas.ce.electricity",
                           "cas.electricity_primary_factor"),
    ("ammonia", "cement"): ("cas.ce.limestone_per_clinker", "cas.ce.clinker_ratio",
                            "cas.powder_factor", "cas.ammonia_per_explosive"),
    ("energy", "ammonia"): ("cas.nh.energy",),
    ("energy", "water"): ("cas.wa.energy_conventional", "cas.wa.desal_electricity",
                          "cas.wa.desal_share", "cas.electricity_primary_factor"),
    ("energy", "energy"): ("cas.en.own_use",),
    ("copper", "energy"): ("cas.en.copper_share", "cas.x2025.energy"),
    ("steel", "energy"): ("cas.en.steel_use", "cas.x2025.energy"),
    ("aluminium", "energy"): ("cas.en.aluminium_share", "cas.x2025.aluminium", "cas.x2025.energy"),
    ("water", "energy"): ("cas.en.water_consumption", "cas.x2025.energy"),
    ("ammonia", "energy"): ("cas.en.coal_per_mj", "cas.en.coal_explosives_per_t",
                            "cas.ammonia_per_explosive"),
}  # fmt: skip


@dataclass(frozen=True)
class Coefficients:
    """Constant cascade parameters (see params/cascade.yaml)."""

    ef: float
    nh3_per_explosive: float
    powder: float
    cu_steel_ore: float
    cu_wear_rock: float
    cu_water_ore: float
    cu_secondary_energy: float
    cu_net: bool
    st_water_bf: float
    st_water_eaf: float
    st_ore_iron: float
    st_rock_ore: float
    st_e_primary: float
    st_e_scrap: float
    al_elec: float
    al_alumina: float
    al_alumina_energy: float
    al_water: float
    ce_thermal: float
    ce_clinker: float
    ce_elec: float
    ce_limestone: float
    nh_energy: float
    wa_conv: float
    wa_desal_elec: float
    wa_desal_share: float
    en_own: float
    en_cu_share: float
    en_steel_use: float
    en_al_share: float
    en_water: float
    en_coal_expl: float
    en_coal_per_mj: float
    x_steel: float
    x_al: float
    x_cement: float
    x_nh3: float
    x_water: float
    x_energy: float

    @classmethod
    def from_registry(cls, reg: Registry) -> Coefficients:
        v = reg.value
        dri = v("st.dri_share_of_primary")
        e_prim = (1 - dri) * v("st.energy.bf_bof") + dri * v("st.energy.dri_eaf")
        return cls(
            ef=v("cas.electricity_primary_factor"),
            nh3_per_explosive=v("cas.ammonia_per_explosive"),
            powder=v("cas.powder_factor") / 1e3,
            cu_steel_ore=v("cas.cu.grinding_steel_per_ore") / 1e3,
            cu_wear_rock=v("cas.cu.loading_wear_per_rock") / 1e3,
            cu_water_ore=v("cas.cu.water_per_ore"),
            cu_secondary_energy=v("cas.cu.secondary_energy") * 1e3,
            cu_net=v("cas.cu_energy_net_of_inputs") >= 0.5,
            st_water_bf=v("cas.st.water_bf_bof"),
            st_water_eaf=v("cas.st.water_eaf"),
            st_ore_iron=v("cas.st.ore_per_primary_iron"),
            st_rock_ore=v("cas.st.rock_per_ore"),
            st_e_primary=e_prim * 1e3,
            st_e_scrap=v("st.energy.scrap_eaf") * 1e3,
            al_elec=v("cas.al.smelting_electricity"),
            al_alumina=v("cas.al.alumina_per_al"),
            al_alumina_energy=v("cas.al.alumina_energy"),
            al_water=v("cas.al.water"),
            ce_thermal=v("cas.ce.kiln_thermal"),
            ce_clinker=v("cas.ce.clinker_ratio"),
            ce_elec=v("cas.ce.electricity"),
            ce_limestone=v("cas.ce.limestone_per_clinker"),
            nh_energy=v("cas.nh.energy"),
            wa_conv=v("cas.wa.energy_conventional"),
            wa_desal_elec=v("cas.wa.desal_electricity"),
            wa_desal_share=v("cas.wa.desal_share"),
            en_own=v("cas.en.own_use"),
            en_cu_share=v("cas.en.copper_share"),
            en_steel_use=v("cas.en.steel_use") * 1e6,
            en_al_share=v("cas.en.aluminium_share"),
            en_water=v("cas.en.water_consumption"),
            en_coal_expl=v("cas.en.coal_explosives_per_t") / 1e3,
            en_coal_per_mj=v("cas.en.coal_per_mj"),
            x_steel=v("cas.x2025.steel") * 1e6,
            x_al=v("cas.x2025.aluminium") * 1e6,
            x_cement=v("cas.x2025.cement") * 1e6,
            x_nh3=v("cas.x2025.ammonia") * 1e6,
            x_water=v("cas.wa.global_withdrawal"),
            x_energy=v("cas.x2025.energy") * 1e12,
        )


@dataclass
class State:
    """Per-year model state the copper and steel columns depend on (aligned arrays)."""

    years: np.ndarray
    cu_energy: np.ndarray  # MJ/kg primary copper (Phase 1 intensity, one bracket)
    ore_per_cu: np.ndarray  # t ore per t primary copper
    rock_per_cu: np.ndarray  # t rock moved per t primary copper
    cu_primary_share: np.ndarray  # primary share of copper supplied to fabrication
    cu_supply_2025: float  # t copper supplied to fabrication in 2025
    st_primary_share: np.ndarray  # primary iron share of the iron charge

    @classmethod
    def from_timeseries(cls, ts: pd.DataFrame, bracket: str) -> State:
        def ratio(num, den):
            r = (ts[num] / ts[den].where(ts[den] > 0)).replace([np.inf, -np.inf], np.nan)
            return r.ffill().bfill().to_numpy()

        cu_e = ts[f"{bracket}_energy_intensity_mj_per_kg"].ffill().bfill().to_numpy()
        prim = ratio("primary_refined", "fabrication_input")
        if "st_primary_iron" in ts.columns:
            st_share = (
                (ts["st_primary_iron"] / (ts["st_primary_iron"] + ts["st_scrap_used"]))
                .fillna(1.0)
                .to_numpy()
            )
        else:
            raise ValueError("cascade needs the steel coupling (scenario option steel: true)")
        return cls(
            years=ts.index.to_numpy(),
            cu_energy=cu_e,
            ore_per_cu=ratio(f"{bracket}_ore", f"{bracket}_primary_cu"),
            rock_per_cu=ratio(f"{bracket}_rock_moved", f"{bracket}_primary_cu"),
            cu_primary_share=prim,
            cu_supply_2025=float(ts.loc[BASE_YEAR, "fabrication_input"]) * 1e6,
            st_primary_share=st_share,
        )


def x_base(cf: Coefficients, st: State) -> np.ndarray:
    """Observed 2025 gross outputs, in sector units."""
    return np.array(
        [st.cu_supply_2025, cf.x_steel, cf.x_al, cf.x_cement, cf.x_nh3, cf.x_water, cf.x_energy]
    )


def build(cf: Coefficients, st: State) -> np.ndarray:
    """A(t) as a (years, 7, 7) array."""
    T, n = len(st.years), len(SECTORS)
    a = np.zeros((T, n, n))
    c, s, al, ce, nh, wa, en = (IDX[k] for k in SECTORS)
    sp = st.cu_primary_share
    nh3_rock = cf.powder * cf.nh3_per_explosive  # t NH3 per t rock blasted

    # copper column
    a[:, en, c] = sp * st.cu_energy * 1e3 + (1 - sp) * cf.cu_secondary_energy
    a[:, s, c] = sp * (st.ore_per_cu * cf.cu_steel_ore + st.rock_per_cu * cf.cu_wear_rock)
    a[:, nh, c] = sp * st.rock_per_cu * nh3_rock
    a[:, wa, c] = sp * st.ore_per_cu * cf.cu_water_ore
    # steel column
    ps = st.st_primary_share
    a[:, en, s] = ps * cf.st_e_primary + (1 - ps) * cf.st_e_scrap
    a[:, nh, s] = ps * cf.st_ore_iron * cf.st_rock_ore * nh3_rock
    a[:, wa, s] = ps * cf.st_water_bf + (1 - ps) * cf.st_water_eaf
    # aluminium column
    a[:, en, al] = cf.al_elec * 3.6 * cf.ef + cf.al_alumina * cf.al_alumina_energy
    a[:, wa, al] = cf.al_water
    # cement column
    a[:, en, ce] = cf.ce_thermal * cf.ce_clinker + cf.ce_elec * 3.6 * cf.ef
    a[:, nh, ce] = cf.ce_limestone * cf.ce_clinker * nh3_rock
    # ammonia column
    a[:, en, nh] = cf.nh_energy
    # water column
    desal = cf.wa_desal_elec * 3.6 * cf.ef
    a[:, en, wa] = cf.wa_conv + cf.wa_desal_share * (desal - cf.wa_conv)
    # energy column (held at 2025 shares)
    a[:, en, en] = cf.en_own
    a[:, c, en] = cf.en_cu_share * st.cu_supply_2025 / cf.x_energy
    a[:, s, en] = cf.en_steel_use / cf.x_energy
    a[:, al, en] = cf.en_al_share * cf.x_al / cf.x_energy
    a[:, wa, en] = cf.en_water / cf.x_energy
    a[:, nh, en] = cf.en_coal_per_mj * cf.en_coal_expl * cf.nh3_per_explosive

    if cf.cu_net:
        gross = a[:, en, c].copy()
        for _ in range(4):  # embodied energy of consumables depends weakly on a_en,cu itself
            mult = leontief.leontief_inverse(a)[:, en, :]  # MJ per unit final demand of each input
            embodied = sum(a[:, i, c] * mult[:, i] for i in (s, nh, wa))
            a[:, en, c] = np.maximum(gross - embodied, 0.0)
    return a


@dataclass
class CascadeResult:
    years: np.ndarray
    a: np.ndarray  # (T, 7, 7)
    y: np.ndarray  # 2025 final demand (sector units)
    x: np.ndarray  # (T, 7) gross output needed to deliver y
    rho: np.ndarray  # (T,)
    maintenance_energy_ej: np.ndarray  # (T,)

    def frame(self, prefix: str) -> pd.DataFrame:
        """Columns for the run time series."""
        idx = pd.Index(self.years, name="year")
        lmat = leontief.leontief_inverse(self.a)
        e = IDX["energy"]
        df = pd.DataFrame(index=idx)
        df[f"{prefix}rho"] = self.rho
        df[f"{prefix}maintenance_energy_ej"] = self.maintenance_energy_ej
        df[f"{prefix}copper_energy_multiplier_gj_per_t"] = lmat[:, e, IDX["copper"]] / 1e3
        df[f"{prefix}copper_direct_energy_ej"] = (
            self.a[:, e, IDX["copper"]] * self.x[:, IDX["copper"]] / 1e12
        )
        df[f"{prefix}steel_direct_energy_ej"] = (
            self.a[:, e, IDX["steel"]] * self.x[:, IDX["steel"]] / 1e12
        )
        return df


def run(ts: pd.DataFrame, reg: Registry, bracket: str) -> CascadeResult:
    cf = Coefficients.from_registry(reg)
    st = State.from_timeseries(ts, bracket)
    a = build(cf, st)
    k = int(np.flatnonzero(st.years == BASE_YEAR)[0])
    x0 = x_base(cf, st)
    y = (np.eye(len(SECTORS)) - a[k]) @ x0
    if np.any(y < 0):
        bad = [SECTORS[i] for i in np.flatnonzero(y < 0)]
        log.warning("2025 final demand negative for %s: intermediate use exceeds output", bad)
    x = np.einsum("tij,j->ti", leontief.leontief_inverse(a), y)
    rho = leontief.spectral_radius(a, label=bracket)
    return CascadeResult(st.years, a, y, x, np.asarray(rho), x[:, IDX["energy"]] / 1e12)


def zero_cells(a: np.ndarray) -> list[tuple[str, str]]:
    """(input, output) pairs that are zero in every year of A."""
    z = np.all(np.atleast_3d(a) == 0, axis=0) if a.ndim == 3 else (a == 0)
    return [(SECTORS[i], SECTORS[j]) for i, j in zip(*np.nonzero(z), strict=True)]


#: Parameters that need no range: observed 2025 outputs, and the Norgate fit coefficients, whose
#: uncertainty is carried by the copper energy option and the Monte Carlo on grade and depth.
EXEMPT_RANGE = ("cas.x2025.", "en.norgate.")


def provenance_gaps(reg: Registry) -> dict[tuple[str, str], list[str]]:
    """For each non-zero cell, the parameters that are unsourced assumptions or have no range
    (``x2025`` outputs are observations and need no range)."""
    gaps = {}
    for cell, keys in CELL_PARAMS.items():
        bad = []
        for k in keys:
            p = reg[k]
            if p.status == "assumption":
                bad.append(f"{k} (assumption)")
            elif (p.low is None or p.high is None) and not k.startswith(EXEMPT_RANGE):
                bad.append(f"{k} (no range)")
        if bad:
            gaps[cell] = bad
    return gaps


def sensitivity_table(result: CascadeResult, year: int) -> pd.DataFrame:
    """Elasticities of rho and of maintenance energy to every non-zero a_ij in ``year``."""
    k = int(np.flatnonzero(result.years == year)[0])
    a = result.a[k]
    e = IDX["energy"]
    er = leontief.rho_elasticities(a)
    ee = leontief.energy_elasticities(a, result.y, e)
    rows = [
        {"input": SECTORS[i], "output": SECTORS[j], "a_ij": a[i, j],
         "rho_elasticity": er[i, j], "energy_elasticity": ee[i, j]}
        for i, j in zip(*np.nonzero(a), strict=True)
    ]  # fmt: skip
    df = pd.DataFrame(rows)
    df["rank"] = df["energy_elasticity"].abs().rank(ascending=False, method="first").astype(int)
    return df.sort_values("rank").reset_index(drop=True)
