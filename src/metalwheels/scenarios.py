"""Scenario definitions (``scenarios/*.yaml``) and the Phase 1 run pipeline.

A scenario file::

    name: recycling_95
    description: ...
    overrides: {cu.eol_collection_rate: 0.7}   # registry values replaced for the whole run
    policy:                                     # optional: EoL rates changed from start_year
      start_year: 2026
      ramp_years: 0
      collection: 0.95
      processing: 0.95
    options:
      end_year: 3025
      energy_option: norgate_coarse
      steel: true                               # Phase 2 steel coupling (default on)

Every run produces both endowment brackets (Harmsen 2013 and Kesler & Wilkinson 2008) side by
side; neither is a default.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metalwheels import registry as _registry
from metalwheels.copper import demand, energy, fates, geology, mass
from metalwheels.logging_setup import get_logger
from metalwheels.registry import Registry

log = get_logger(__name__)

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"
BRACKETS = ("harmsen", "kesler")
_OPTION_KEYS = {"end_year", "energy_option", "steel"}

#: Steel flow columns copied into the run time series with an ``st_`` prefix.
STEEL_COLUMNS = (
    "products",
    "iron_sources",
    "primary_iron",
    "copper_induced_primary",
    "eol",
    "scrap_recovered",
    "scrap_used",
    "unusable_scrap",
    "stock",
    "stockpile",
    "cu_attached",
    "cu_eol_bulk",
    "cu_scrap",
    "cu_products",
    "cu_stock",
    "cu_supply_aggregate",
    "cu_tolerable_aggregate",
    "scrap_cu_mean",
    "secondary_cu",
    "product_cu_mean",
)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    overrides: dict[str, float] = field(default_factory=dict)
    policy: dict | None = None
    end_year: int = 3025
    energy_option: str = "norgate_coarse"
    steel: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        unknown = set(d) - {"name", "description", "overrides", "policy", "options"}
        if unknown:
            raise ValueError(f"unknown scenario keys: {sorted(unknown)}")
        opts = d.get("options") or {}
        bad = set(opts) - _OPTION_KEYS
        if bad:
            raise ValueError(f"unknown scenario options: {sorted(bad)}")
        if opts.get("energy_option", "norgate_coarse") not in energy.OPTIONS:
            raise ValueError(f"unknown energy option {opts['energy_option']!r}")
        return cls(
            name=d["name"],
            description=(d.get("description") or "").strip(),
            overrides={k: float(v) for k, v in (d.get("overrides") or {}).items()},
            policy=d.get("policy") or None,
            end_year=int(opts.get("end_year", 3025)),
            energy_option=opts.get("energy_option", "norgate_coarse"),
            steel=bool(opts.get("steel", True)),
        )


def list_scenarios(directory: Path = SCENARIO_DIR) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_scenario(name_or_path: str, directory: Path = SCENARIO_DIR) -> Scenario:
    path = Path(name_or_path)
    if not path.suffix:
        path = directory / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"scenario not found: {path} (available: {list_scenarios(directory)})"
        )
    sc = Scenario.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    if path.stem != sc.name:
        raise ValueError(f"scenario name {sc.name!r} does not match file name {path.stem!r}")
    return sc


@dataclass
class RunOutput:
    scenario: Scenario
    timeseries: pd.DataFrame  # index year; schema in metalwheels.export
    summary: dict[str, tuple[float, str]]  # metric -> (value, unit)
    runtime_s: float
    calibration: dict[str, float]


def run_scenario(
    sc: Scenario, reg: Registry | None = None, progress: Callable[[str], None] | None = None
) -> RunOutput:
    """Run the Phase 1 copper chain and, unless ``sc.steel`` is false, the Phase 2 steel
    coupling for one scenario."""
    t0 = time.perf_counter()
    step = progress or (lambda _msg: None)
    reg = (reg or _registry.load()).with_overrides(sc.overrides, label=sc.name)
    tau = reg.value("cu.tailings_share_of_ore_cu")
    s_loss = reg.value("cu.smelting_loss_fraction")

    step("stock-flow and demand")
    from metalwheels.ratchets import carbon_cup

    policy, cup = carbon_cup.resolve_policy(sc.policy, reg)
    pj = demand.project(end_year=sc.end_year, reg=reg, policy=policy)
    flows = pj.result.flows
    years = flows.index.to_numpy()
    ore_cu = flows["mine"].to_numpy() / (1 - tau)

    step("receiving media")
    media = fates.partition(pj.result, tau)

    ts = flows.drop(columns=["target_stock", "demand_driven"]).copy()
    ts["demand_driven"] = flows["demand_driven"].astype(bool)
    ts["target_stock"] = flows["target_stock"]
    ts["population"] = pj.population
    ts["stock_per_capita"] = pj.per_capita_stock
    ts["ore_cu_demand"] = ore_cu
    for col in media.columns:
        if col != "stock":
            ts[f"media_{col}"] = media[col]
    ts = pd.concat([ts, carbon_cup.stream(ts, cup)], axis=1)

    em = energy.EnergyModel.from_registry(reg, sc.energy_option)
    mp = mass.MassParams.from_registry(reg)
    calib = {"growth_rate": pj.growth_rate, "k0": pj.k0, "k_sat": pj.k_sat}
    summary: dict[str, tuple[float, str]] = {}
    for br in BRACKETS:
        step(f"geology, energy and mass ({br})")
        prof, ex = geology.extraction_for(reg, br, years, ore_cu)
        calib[f"{br}_depth_multiplier"] = ex.multiplier
        calib[f"{br}_lead_years"] = ex.lead_years
        primary = ex.extracted * (1 - tau) * (1 - s_loss)
        intensity = em.intensity(ex.grade / 100, ex.depth)
        ej = np.where(np.isfinite(intensity), energy.annual_energy_ej(primary, intensity), 0.0)
        mf = mass.mass_flows(years, ex.extracted, ex.grade, mp)
        p = f"{br}_"
        ts[p + "grade_pct"] = ex.grade
        ts[p + "depth_m"] = ex.depth
        ts[p + "discovered_fraction"] = ex.discovered_fraction
        ts[p + "extracted_cu"] = ex.extracted
        ts[p + "shortfall_cu"] = ex.shortfall
        ts[p + "primary_cu"] = primary
        ts[p + "energy_intensity_mj_per_kg"] = intensity
        ts[p + "primary_energy_ej"] = ej
        for col in ("ore", "waste_rock", "tailings", "tailings_volume_km3", "rock_moved"):
            ts[p + col] = mf[col]
            ts[p + "cum_" + col] = mf["cum_" + col]
        for layer in ex.remaining_by_layer.columns:
            ts[f"{p}remaining_{layer}"] = ex.remaining_by_layer[layer]
        ts[p + "remaining_total"] = ex.remaining_by_layer.sum(axis=1)

        ey = ex.first_shortfall_year
        summary[p + "endowment"] = (prof.endowment, "Mt Cu")
        summary[p + "first_shortfall_year"] = (float(ey) if ey else float("nan"), "year")
        for layer, yr in ex.layer_exhaustion_years(prof.layer_totals()).items():
            summary[f"{p}layer_{layer}_exhausted_year"] = (
                float(yr) if yr else float("nan"),
                "year",
            )
        for y in (2010, 2050, 2100, 2200, 2500, sc.end_year):
            if y in ts.index:
                summary[f"{p}grade_{y}"] = (float(ts.loc[y, p + "grade_pct"]), "wt% Cu")
                summary[f"{p}depth_{y}"] = (float(ts.loc[y, p + "depth_m"]), "m")
                summary[f"{p}energy_intensity_{y}"] = (
                    float(ts.loc[y, p + "energy_intensity_mj_per_kg"]),
                    "MJ/kg",
                )
                summary[f"{p}primary_energy_{y}"] = (
                    float(ts.loc[y, p + "primary_energy_ej"]),
                    "EJ/yr",
                )
        summary[p + "cum_primary_energy"] = (float(ts[p + "primary_energy_ej"].sum()), "EJ")
        summary[p + "cum_tailings_volume_end"] = (
            float(mf["cum_tailings_volume_km3"].iloc[-1]),
            "km3",
        )

    fut = ts.index > demand.LAST_HISTORICAL_YEAR
    summary["cum_mine_output_future"] = (float(ts.loc[fut, "mine"].sum()), "Mt Cu")
    summary["cum_ore_cu_future"] = (float(ts.loc[fut, "ore_cu_demand"].sum()), "Mt Cu")
    for y in (2026, 2050, 2100, 2200, sc.end_year):
        if y in ts.index:
            summary[f"refined_{y}"] = (float(ts.loc[y, "refined"]), "Mt/yr")
            summary[f"mine_{y}"] = (float(ts.loc[y, "mine"]), "Mt/yr")
            summary[f"stock_{y}"] = (float(ts.loc[y, "stock"]), "Mt")
    summary["eol_recycling_rate_end"] = (
        float(ts["old_scrap"].iloc[-1] / ts["eol"].iloc[-1]),
        "1",
    )
    if cup is not None:
        summary["cc_co2_captured_cum"] = (float(ts["cc_co2_captured"].sum()), "Mt CO2")
        summary["cc_co2_emitted_cum"] = (
            float((ts["cc_co2_emitted_fossil"] + ts["cc_co2_emitted_biogenic"]).sum()),
            "Mt CO2",
        )
        summary["cc_co2_net_cum"] = (float(ts["cc_co2_net"].sum()), "Mt CO2")
    if sc.steel:
        step("steel coupling")
        _steel(ts, summary, calib, pj.result, reg)
        ts = ts.copy()  # defragment before the cascade adds its columns
        step("cascade")
        _cascade(ts, summary, reg)
    for k, v in calib.items():
        summary[f"calib_{k}"] = (float(v), "")
    return RunOutput(sc, ts, summary, time.perf_counter() - t0, calib)


def _steel(ts: pd.DataFrame, summary: dict, calib: dict, cu_result, reg: Registry) -> None:
    """Phase 2: copper -> steel link, steel stock model, tolerances and energy penalty."""
    from metalwheels.steel import coupled, link
    from metalwheels.steel import energy as st_energy

    cs = coupled.run(cu_result, reg)
    cal, lk, sp = cs.calibration, cs.link, cs.projection
    sf = sp.result.flows
    en = st_energy.SteelEnergy.from_registry(reg)
    for c in STEEL_COLUMNS:
        ts["st_" + c] = sf[c].to_numpy()
    ts["st_stock_per_capita"] = sp.per_capita_stock
    ts["st_cu_to_other_sinks"] = lk.to_other_sinks.to_numpy()
    ts["st_energy_ej"] = st_energy.steel_energy_ej(sf["primary_iron"], sf["scrap_used"], en)
    ts["st_cu_penalty_ej"] = st_energy.penalty_ej(sf["copper_induced_primary"], en)
    for j, s_name in enumerate(link.SECTORS):
        calib[f"steel_growth_rate_{s_name}"] = float(sp.growth_rate[j])
        calib[f"steel_k0_{s_name}"] = float(sp.k0[j])
        calib[f"steel_cu_link_{s_name}"] = float(cal.phi[j])
        summary[f"st_scrap_cu_2008_{s_name}"] = (float(cal.achieved[j]), "1")
        summary[f"st_scrap_cu_2008_bulk_share_{s_name}"] = (float(cal.bulk_share[j]), "1")
    for kind in ("lp", "aggregate"):
        y = sp.result.crossover_year(kind)
        summary[f"st_crossover_year_{kind}"] = (float(y) if y else float("nan"), "year")
    summary["st_cum_cu_penalty_ej"] = (float(ts["st_cu_penalty_ej"].sum()), "EJ")
    summary["st_stockpile_end"] = (float(ts["st_stockpile"].iloc[-1]), "Mt")
    for y in (2008, 2050, 2100, 2200, 2500, int(ts.index[-1])):
        if y in ts.index:
            summary[f"st_scrap_cu_{y}"] = (float(ts.loc[y, "st_scrap_cu_mean"]), "1")
            summary[f"st_cu_penalty_{y}"] = (float(ts.loc[y, "st_cu_penalty_ej"]), "EJ/yr")
            summary[f"st_copper_induced_primary_{y}"] = (
                float(ts.loc[y, "st_copper_induced_primary"]),
                "Mt/yr",
            )


#: Cascade columns per bracket (prefix ``cas_<bracket>_``).
CASCADE_COLUMNS = (
    "rho",
    "maintenance_energy_ej",
    "copper_energy_multiplier_gj_per_t",
    "copper_direct_energy_ej",
    "steel_direct_energy_ej",
)


def _cascade(ts: pd.DataFrame, summary: dict, reg: Registry) -> None:
    """Phase 3: A(t), rho and maintenance energy for each endowment bracket."""
    from metalwheels.cascade import matrix as cascade

    for br in BRACKETS:
        res = cascade.run(ts, reg, br)
        frame = res.frame(f"cas_{br}_")
        # After the bracket's first supply shortfall, 2025 final demand can no longer be met
        # from mines, so maintenance energy (and the multipliers behind it) is undefined.
        sy = summary.get(f"{br}_first_shortfall_year", (float("nan"), ""))[0]
        if sy == sy:
            frame.loc[frame.index >= sy, [c for c in frame.columns if not c.endswith("rho")]] = (
                np.nan
            )
        for c in frame.columns:
            ts[c] = frame[c].to_numpy()
        p = f"cas_{br}_"
        for y in (2025, 2050, 2100, 2200, 2500, int(ts.index[-1])):
            if y in ts.index:
                summary[f"{p}rho_{y}"] = (float(ts.loc[y, p + "rho"]), "1")
                summary[f"{p}maintenance_energy_{y}"] = (
                    float(ts.loc[y, p + "maintenance_energy_ej"]),
                    "EJ/yr",
                )
        summary[f"{p}rho_max"] = (float(ts[p + "rho"].max()), "1")
        summary[f"{p}valid_until"] = (float(sy - 1) if sy == sy else float(ts.index[-1]), "year")
