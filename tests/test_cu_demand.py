"""Task 1.4 done condition: refined demand in 2026 within 26-32 Mt; stock converges to within
1 % of the saturation target once population is flat; per-capita stock never exceeds the ceiling."""

import numpy as np
import pytest

from metalwheels import registry
from metalwheels.copper import demand
from metalwheels.copper import stockflow as sf


@pytest.fixture(scope="module")
def proj():
    return demand.project(end_year=3025)


def test_refined_demand_2026(proj):
    assert 26 <= proj.result.flows.loc[2026, "refined"] <= 32


def test_converges_to_saturation_once_population_flat(proj):
    pop_flat = proj.population[-1]
    saturation_stock = proj.k_sat * pop_flat / 1e9
    stock = proj.result.flows["stock"]
    # within 1 % at the horizon, and from some year onward it stays within 1 %
    assert stock.loc[3025] == pytest.approx(saturation_stock, rel=0.01)
    within = (stock / saturation_stock - 1).abs() < 0.01
    first = within[within].index.min()
    assert within.loc[first:].all()
    assert first > 2100  # population is flat only after 2100


def test_per_capita_never_exceeds_ceiling(proj):
    pc = proj.per_capita_stock
    assert np.nanmax(pc) <= proj.k_sat * (1 + 1e-9)


def test_stock_tracks_target_in_projection(proj):
    f = proj.result.flows
    fut = f.index > demand.LAST_HISTORICAL_YEAR
    np.testing.assert_allclose(f.loc[fut, "stock"], f.loc[fut, "target_stock"], rtol=1e-9)
    assert (f.loc[fut, "mine"] >= 0).all()
    assert (f.loc[fut, "demand_driven"] == 1).all()


def test_projection_continuous_with_history(proj):
    hist = sf.run_historical()
    f = proj.result.flows
    np.testing.assert_allclose(f.loc[:2025, "stock"], hist.flows["stock"], rtol=1e-12)
    jump = f.loc[2026, "refined"] / f.loc[2025, "refined"] - 1
    assert abs(jump) < 0.05


def test_mass_balance_whole_run(proj):
    assert proj.result.balance_residual().abs().max() < 1e-6


def test_growth_rate_fit_on_logistic_data():
    """Fitting a pure logistic series recovers g to within 2 %."""
    g, k_sat, k0 = 0.03, 200.0, 20.0
    years = np.arange(1990, 2026)
    k = demand.logistic_path(k0, k_sat, g, (years - years[0]).astype(float))
    fitted = demand.fit_growth_rate(dict(zip(years.tolist(), k, strict=True)), k_sat, 30, 2025)
    assert fitted == pytest.approx(g, rel=0.02)


def test_population_held_flat_after_hold_year():
    pop = demand.load_population()
    years = np.array([2099, 2100, 2101, 2500])
    p = demand.population_path(years, pop, 2100)
    assert p[1] == p[2] == p[3] == pop[2100]


def test_mine_clamped_when_target_falls():
    """If the target stock collapses, mine output is clamped to zero, not negative."""
    prm = sf.CopperParams.from_registry(registry.load())
    years = np.arange(1900, 2041)
    mine = np.where(years <= 2025, 10.0, np.nan)
    target = np.where(years > 2025, 1.0, np.nan)  # absurdly low target
    r = sf.run(years, mine, prm, target_stock=target)
    f = r.flows
    assert (f.loc[2026:, "mine"] == 0).all()
    assert (f.loc[2026:, "stock"] > f.loc[2026:, "target_stock"]).all()
    assert r.balance_residual().abs().max() < 1e-6


def test_nan_mine_without_target_rejected():
    prm = sf.CopperParams.from_registry(registry.load())
    with pytest.raises(ValueError, match="target_stock"):
        sf.run(np.arange(3), np.array([1.0, np.nan, 1.0]), prm)
