"""Task 1.5 done condition: median 0.52 +/- 0.01 % Cu; analytic cumulative-above-cutoff reproduced
to 1e-6; replaying historical production gives a 2010 mined grade within 0.5-0.75 % Cu (observed
0.62 %); layer exhaustion under a constant extraction rate matches the analytic quotient."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate

from metalwheels import registry
from metalwheels.copper import geology as geo
from metalwheels.copper import production


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.fixture(scope="module")
def gerst(reg):
    return geo.GradeTonnage(
        reg.value("geo.grade.gerst.mu_w"), reg.value("geo.grade.gerst.sigma2_w"), 1000.0
    )


def test_median_grade(gerst):
    assert gerst.median_grade == pytest.approx(0.52, abs=0.01)


@pytest.mark.parametrize("cutoff", [0.1, 0.3, 0.52, 1.0, 2.5])
def test_copper_above_cutoff_matches_numerical_integral(gerst, cutoff):
    mu, s = gerst.mu, gerst.sigma

    def ore_density(g):  # Mt ore per unit grade (wt%)
        return (
            gerst.total_ore
            * math.exp(-((math.log(g) - mu) ** 2) / (2 * s * s))
            / (g * s * math.sqrt(2 * math.pi))
        )

    cu, _ = integrate.quad(
        lambda g: g / 100 * ore_density(g), cutoff, np.inf, epsabs=1e-12, limit=200
    )
    ore, _ = integrate.quad(ore_density, cutoff, np.inf, epsabs=1e-12, limit=200)
    assert float(gerst.copper_above(cutoff)) == pytest.approx(cu, rel=1e-6)
    assert float(gerst.ore_above(cutoff)) == pytest.approx(ore, rel=1e-6)


def test_cutoff_inverse_roundtrip(gerst):
    for x in (1.0, 100.0, 500.0, 999.0):
        assert float(gerst.copper_above(gerst.cutoff_for_extracted(x))) == pytest.approx(
            x, rel=1e-9
        )


@pytest.fixture(scope="module")
def history(reg):
    series = production.load()
    years = np.array(sorted(series))
    ore_cu = np.array([series[y] for y in years]) / (1 - reg.value("cu.tailings_share_of_ore_cu"))
    return years, ore_cu


@pytest.mark.parametrize("bracket", ["harmsen", "kesler"])
def test_mined_grade_2010_from_history(reg, history, bracket):
    years, ore_cu = history
    _, r = geo.extraction_for(reg, bracket, years, ore_cu)
    g2010 = r.grade[list(years).index(2010)]
    assert 0.50 <= g2010 <= 0.75


def test_depth_multiplier_hits_calibration_target(reg, history):
    years, ore_cu = history
    prof = geo.DepthProfile.from_registry(reg, "harmsen")
    args = (reg.value("en.norgate.coarse.b"), reg.value("en.depth.energy_per_m"),
            reg.value("en.depth.reference_depth"))  # fmt: skip
    target = reg.value("geo.calib.produced_share_top500_2010")
    m = geo.calibrate_multiplier(prof, *args, years, ore_cu, target)
    ex = geo.Extraction(prof, *args, m)
    cut = ex._solve(prof.discovered, ore_cu[years <= 2010].sum())
    cu = prof.discovered * ex._cu_frac(cut)
    assert cu[prof.mids < 500].sum() / cu.sum() == pytest.approx(target, abs=1e-4)


def test_profiles_sum_to_sourced_endowments(reg):
    h = geo.DepthProfile.from_registry(reg, "harmsen")
    assert h.endowment == pytest.approx(24_890, abs=1)
    assert h.discovered.sum() == pytest.approx(2_104.4, abs=0.5)
    k = geo.DepthProfile.from_registry(reg, "kesler")
    assert k.endowment == pytest.approx(89_000, rel=1e-9)
    assert k.mids.max() < 3_300


def test_layer_exhaustion_matches_analytic_quotient():
    """Strict depth order (huge multiplier), constant rate: layer l is exhausted at
    t0 + (cumulative endowment through l) / rate, to within one year."""
    mids = np.array([250.0, 750.0, 1500.0, 2500.0])
    layer = np.array(["a", "a", "b", "c"])
    disc = np.array([150.0, 150.0, 500.0, 400.0])
    prof = geo.DepthProfile("test", mids, layer, disc, np.zeros(4), mu=-0.6616, sigma2=0.1567)
    rate, years = 10.0, np.arange(2000, 2120)
    # all slices deeper than d0 = 0 get offsets; make them huge so order is strictly by depth
    ex = geo.Extraction(prof, b=0.59, k_depth=0.029, d0=0.0, multiplier=1e6, calibration_year=3000)
    r = ex.run(years, np.full(len(years), rate))
    got = r.layer_exhaustion_years(prof.layer_totals())
    cum = {"a": 300.0, "b": 800.0, "c": 1200.0}
    for name, c in cum.items():
        expected = years[0] + c / rate - 1  # year in which the cumulative total is reached
        assert got[name] == pytest.approx(expected, abs=1)


def test_conservation_and_shortfall(reg, history):
    years, ore_cu = history
    demand = np.concatenate([ore_cu, np.full(600, 80.0)])  # force exhaustion of Harmsen
    yrs = np.arange(years[0], years[0] + len(demand))
    prof, r = geo.extraction_for(reg, "harmsen", yrs, demand)
    total_extracted = r.extracted.sum()
    np.testing.assert_allclose(
        total_extracted + r.remaining_by_layer.iloc[-1].sum(), prof.endowment, rtol=1e-9
    )
    assert r.exhausted_year is not None
    np.testing.assert_allclose(r.extracted + r.shortfall, demand, rtol=1e-9, atol=1e-9)


def test_paced_discovery_keeps_lead(reg, history):
    years, ore_cu = history
    demand = np.concatenate([ore_cu, np.full(100, 40.0)])
    yrs = np.arange(years[0], years[0] + len(demand))
    prof, r = geo.extraction_for(reg, "kesler", yrs, demand)
    after = yrs > 2010
    avail = prof.discovered.sum() + r.discovered_fraction * prof.undiscovered.sum()
    k = int(reg.value("geo.discovery_smoothing_years"))
    trend = pd.Series(demand).rolling(k, min_periods=1).mean().to_numpy()
    lead = (avail - np.cumsum(r.extracted)) / trend
    assert (lead[after] >= r.lead_years * (1 - 1e-9)).all()
    assert (np.diff(r.discovered_fraction) >= 0).all()
