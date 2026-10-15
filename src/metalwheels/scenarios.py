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
_OPTION_KEYS = {"end_year", "energy_option"}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    overrides: dict[str, float] = field(default_factory=dict)
    policy: dict | None = None
    end_year: int = 3025
    energy_option: str = "norgate_coarse"

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
    """Run the full Phase 1 chain for one scenario."""
    t0 = time.perf_counter()
    step = progress or (lambda _msg: None)
    reg = (reg or _registry.load()).with_overrides(sc.overrides, label=sc.name)
    tau = reg.value("cu.tailings_share_of_ore_cu")
    s_loss = reg.value("cu.smelting_loss_fraction")

    step("stock-flow and demand")
    pj = demand.project(end_year=sc.end_year, reg=reg, policy=sc.policy)
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

        ey = ex.exhausted_year
        summary[p + "endowment"] = (prof.endowment, "Mt Cu")
        summary[p + "exhausted_year"] = (float(ey) if ey else float("nan"), "year")
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
    for k, v in calib.items():
        summary[f"calib_{k}"] = (float(v), "")
    return RunOutput(sc, ts, summary, time.perf_counter() - t0, calib)
