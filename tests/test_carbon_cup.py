"""Task 4.3 done condition: Ellingham eligibility rejects Al, Mg, Ti, rare earths and Li and
accepts Cu, Ni, Co, Sn, Pb, Ag, Au and the PGMs; the scenario runs end to end and reports
delta maintenance energy, delta rho and CO2 captured/emitted."""

import dataclasses
import logging

import numpy as np
import pytest

from metalwheels import export, registry, scenarios
from metalwheels.ratchets import carbon_cup as cc

ACCEPT = ("Cu", "Ni", "Co", "Sn", "Pb", "Ag", "Au", *cc.PGMS)
REJECT = ("Al", "Mg", "Ti", "Li", *cc.RARE_EARTHS)


@pytest.fixture(scope="module")
def thermo():
    return cc.Thermo()


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.mark.parametrize("t_max_key", ["value", "low", "high"])
def test_eligibility_over_the_whole_t_max_range(thermo, reg, t_max_key):
    p = reg["cc.t_max"]
    t = getattr(p, t_max_key)
    for el in ACCEPT:
        assert thermo.reducible(el, t), (el, t)
    for el in REJECT:
        assert not thermo.reducible(el, t), (el, t)


def test_onset_temperatures_match_textbook_order(thermo):
    """Carbon reduces Cu < Ni < Fe < Zn < Cr < Mn < Si < Mg < Al < Ti (Ellingham order)."""
    order = ["Cu", "Ni", "Fe", "Zn", "Cr", "Mn", "Si", "Mg", "Al", "Ti"]
    onsets = [thermo.onset_temperature(e) for e in order]
    assert onsets == sorted(onsets)
    # carbothermic magnesium needs about 1850 C; alumina about 2000 C or more
    assert 2000 < thermo.onset_temperature("Mg") < 2200
    assert thermo.onset_temperature("Al") > 2200


def test_ulich_lines_agree_with_janaf(thermo):
    """NBS 298 K values with constant dH, dS reproduce JANAF dG(T) within 25 kJ/mol O2 to 1500 K
    (the Ulich approximation errs towards easier reduction, by up to about 20 kJ here)."""
    t = np.array([500.0, 1000.0, 1500.0])
    assert thermo.checks, "no cross-check rows"
    for ck in thermo.checks:
        ref = next(o for o in thermo.oxides if o.compound == ck.compound and o.method == "janaf")
        diff = ck.dg_o2(t, thermo.janaf) - ref.dg_o2(t, thermo.janaf)
        assert np.all(np.abs(diff) < 25.0), (ck.compound, diff)
        assert np.all(diff > -5.0), (ck.compound, diff)


def test_co_line_matches_janaf_table(thermo):
    # JANAF C-093: dfG(CO, 1500 K) = -243.740 kJ/mol
    assert thermo.dg_co(1500.0) == pytest.approx(-487.48, abs=1e-6)


def test_resolve_policy_gives_effective_rates(reg):
    pol = {"start_year": 2030, "ramp_years": 10, "carbon_cup": {"share": 0.5}}
    out, spec = cc.resolve_policy(pol, reg)
    c, p = reg.value("cu.eol_collection_rate"), reg.value("cu.eol_processing_rate")
    assert "carbon_cup" not in out
    assert out["collection"] == pytest.approx(0.5 * c + 0.5 * reg.value("cc.collection"))
    assert out["processing"] == pytest.approx(0.5 * p + 0.5 * reg.value("cc.processing"))
    assert spec.share_path(np.array([2029, 2030, 2035, 2039, 2050])) == pytest.approx(
        [0.0, 0.05, 0.3, 0.5, 0.5]
    )
    with pytest.raises(ValueError, match="unknown carbon_cup"):
        cc.resolve_policy({"carbon_cup": {"share": 0.5, "colour": 1}}, reg)
    with pytest.raises(ValueError, match="copper only"):
        cc.resolve_policy({"carbon_cup": {"share": 0.5, "metal": "Al"}}, reg)


@pytest.fixture(scope="module")
def runs(reg):
    logging.getLogger("metalwheels").setLevel(logging.ERROR)
    base = scenarios.run_scenario(scenarios.load_scenario("baseline"), reg)
    cup = scenarios.run_scenario(scenarios.load_scenario("carbon_cup"), reg)
    logging.getLogger("metalwheels").setLevel(logging.INFO)
    return base, cup


def test_scenario_runs_end_to_end_and_reports_three_outputs(runs):
    base, cup = runs
    export.validate(cup.timeseries)
    df, out = cc.compare(base, cup)
    for br in scenarios.BRACKETS:
        assert np.isfinite(out[f"d_maintenance_energy_cum_{br}"][0])
        assert np.isfinite(out[f"d_rho_{br}_2100"][0])
    assert out["cc_co2_captured_cum"][0] > 0
    assert out["cc_co2_emitted_fossil_cum"][0] > 0
    assert np.isfinite(out["cc_co2_net_cum"][0])
    assert {"d_maintenance_energy_harmsen", "d_rho_kesler", "cc_co2_captured"} <= set(df.columns)


def test_lever_raises_recycling_and_cleans_steel(runs):
    base, cup = runs
    tb, tc = base.timeseries, cup.timeseries
    assert tc.loc[:2029, "old_scrap"].to_numpy() == pytest.approx(tb.loc[:2029, "old_scrap"])
    assert tc.loc[2100, "old_scrap"] > tb.loc[2100, "old_scrap"]
    assert tc.loc[2100, "media_other_scrap_markets"] < tb.loc[2100, "media_other_scrap_markets"]
    assert tc.loc[2100, "st_scrap_cu_mean"] < tb.loc[2100, "st_scrap_cu_mean"]
    assert cup.summary["st_cum_cu_penalty_ej"][0] < base.summary["st_cum_cu_penalty_ej"][0]
    _, out = cc.compare(base, cup)
    assert out["d_maintenance_energy_harmsen_2100"][0] < 0


def test_co2_stream_balances(runs):
    _, cup = runs
    t = cup.timeseries
    total = t["cc_co2_total"]
    parts = t["cc_co2_captured"] + t["cc_co2_emitted_fossil"] + t["cc_co2_emitted_biogenic"]
    assert parts.to_numpy() == pytest.approx(total.to_numpy())
    carbon = t["cc_matrix_carbon"] + t["cc_reductant_carbon"]
    assert (carbon * cc.M_CO2 / cc.M_C).to_numpy() == pytest.approx(total.to_numpy())
    assert (t.loc[:2029, "cc_co2_total"] == 0).all()


def test_capture_and_biogenic_share_are_configurable(reg, runs):
    _, _cup = runs
    sc = scenarios.load_scenario("carbon_cup")
    sc2 = dataclasses.replace(
        sc, overrides={"cc.capture_fraction": 0.0, "cc.biogenic_share": 1.0}, steel=False
    )
    logging.getLogger("metalwheels").setLevel(logging.ERROR)
    r2 = scenarios.run_scenario(sc2, reg)
    t2 = r2.timeseries
    assert (t2["cc_co2_captured"] == 0).all()
    assert (t2["cc_co2_emitted_fossil"] == 0).all()
    assert t2["cc_co2_emitted_biogenic"].to_numpy() == pytest.approx(t2["cc_co2_total"].to_numpy())


def test_baseline_carries_zero_stream(runs):
    base, _ = runs
    for c in cc.STREAM_COLUMNS:
        assert (base.timeseries[c] == 0).all(), c


def test_matrix_energy_enters_the_cascade(reg):
    sc = scenarios.load_scenario("carbon_cup")
    logging.getLogger("metalwheels").setLevel(logging.ERROR)
    hi = scenarios.run_scenario(
        dataclasses.replace(sc, overrides={"cc.matrix_net_energy": 25.0}), reg
    )
    lo = scenarios.run_scenario(
        dataclasses.replace(sc, overrides={"cc.matrix_net_energy": 0.0}), reg
    )
    e = "cas_kesler_maintenance_energy_ej"
    assert hi.timeseries.loc[2100, e] > lo.timeseries.loc[2100, e]
