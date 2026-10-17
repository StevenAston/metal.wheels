"""Tasks 2.1-2.2: steel stock model with copper tracking, and the copper -> steel link."""

import logging

import numpy as np
import pytest

from metalwheels import registry
from metalwheels.copper import demand as cu_demand
from metalwheels.copper import stockflow as csf
from metalwheels.steel import calibrate, link
from metalwheels.steel import coupled as st_coupled
from metalwheels.steel import stockflow as ssf

logging.getLogger("metalwheels").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.fixture(scope="module")
def prm(reg):
    return ssf.SteelParams.from_registry(reg)


@pytest.fixture(scope="module")
def run_all(reg):
    pj = cu_demand.project(end_year=2300, reg=reg)
    return pj, st_coupled.run(pj.result, reg)


@pytest.fixture(scope="module")
def coupled(run_all):
    pj, cs = run_all
    return pj, cs.link, cs.projection


# --- Task 2.2: cross-module copper conservation --------------------------------------------


def test_link_conserves_copper_every_year(coupled):
    """Copper leaving the copper module = copper into the steel pool + declared other sinks."""
    pj, lk, _ = coupled
    leaving = pj.result.flows["other_scrap_markets"]
    assert np.allclose(lk.other_scrap.sum(axis=1), leaving, rtol=1e-12, atol=0)
    assert lk.conservation_residual().abs().max() < 1e-6


def test_steel_receives_exactly_the_linked_copper(coupled):
    _, lk, sp = coupled
    assert np.allclose(sp.result.flows["cu_attached"], lk.to_steel.sum(axis=1), rtol=1e-12)


def test_route_rejects_bad_shares(coupled):
    pj, _, _ = coupled
    with pytest.raises(ValueError):
        link.route(pj.result, np.array([0.5, 1.2, 0.1, 0.1]))
    with pytest.raises(ValueError):
        link.route(pj.result, np.array([0.5, 0.5]))


def test_every_copper_category_is_mapped(reg):
    cats = {g.split("_", 1)[0] for g in csf.CopperParams.from_registry(reg).groups}
    mapped = {c for cs in link.CU_CATEGORY_MAP.values() for c in cs}
    assert cats == mapped


def test_full_processing_sends_no_copper_to_steel(reg):
    pj = cu_demand.project(
        end_year=2100, reg=reg, policy={"start_year": 2026, "processing": 1.0}
    )  # fmt: skip
    lk = st_coupled.run(pj.result, reg).link
    assert lk.to_steel.loc[2026:].to_numpy().max() == pytest.approx(0.0, abs=1e-12)


# --- Task 2.1: steel stock model ------------------------------------------------------------


def test_copper_and_iron_balances_close(coupled):
    _, _, sp = coupled
    assert sp.result.copper_residual().abs().max() < 1e-9
    assert sp.result.iron_residual().abs().max() < 1e-9


def test_stock_balance(coupled):
    _, _, sp = coupled
    f = sp.result.flows
    d = f["stock"].diff().fillna(f["stock"].iloc[0])
    assert np.allclose(f["products"] - f["eol"], d, rtol=1e-9, atol=1e-9)


def test_demand_driven_stock_hits_sector_targets(coupled):
    _, _, sp = coupled
    f = sp.result.flows
    fut = (f["demand_driven"] > 0).to_numpy()
    target = sp.per_capita_target[fut] * sp.population[fut, None] / 1e9
    assert np.allclose(sp.result.stock_by_sector.to_numpy()[fut], target, rtol=1e-9)
    assert np.allclose(f.loc[fut, "stock"], f.loc[fut, "target_stock"], rtol=1e-9)


# --- Task 2.1 validation against Pauliuk et al. 2013a ---------------------------------------


def _pc(sp, year):
    return sp.per_capita_stock[list(sp.result.years).index(year)]


def test_per_capita_stock_path_near_pauliuk(coupled, reg):
    """Pauliuk 2013a p. 3451: global average 3.7 t (2008), ca. 6.5 t (2050), ca. 10 t (2100)."""
    _, _, sp = coupled
    assert _pc(sp, 2008) == pytest.approx(reg.value("st.target.stock_per_capita_2008"), rel=0.15)
    assert _pc(sp, 2050) == pytest.approx(6500, rel=0.15)
    assert _pc(sp, 2100) == pytest.approx(10000, rel=0.15)


def test_construction_share_of_stock(coupled, reg):
    _, _, sp = coupled
    st = sp.result.stock_by_sector.loc[2008]
    share = st["construction"] / st.sum()
    assert share == pytest.approx(reg.value("st.target.construction_stock_share"), abs=0.06)


def test_final_demand_2008_near_pauliuk(coupled, reg):
    """Model products exclude castings, so they sit a little below Pauliuk's 1100 Mt/yr."""
    _, _, sp = coupled
    assert sp.result.flows.loc[2008, "products"] == pytest.approx(
        reg.value("st.target.final_demand_2008"), rel=0.15
    )


def test_saturates_at_sector_levels(reg):
    pj = cu_demand.project(end_year=3025, reg=reg)
    sp = st_coupled.run(pj.result, reg).projection
    pc = sp.result.stock_by_sector.loc[3025].to_numpy() * 1e9 / sp.population[-1]
    assert pc == pytest.approx(sp.k_sat, rel=0.01)


# --- Task 2.1 done condition: 2008 scrap copper against Daehn et al. 2017 -------------------


def test_2008_scrap_copper_matches_daehn(run_all, reg):
    """Calibrated link shares reproduce Daehn's 2008 expected scrap copper by sector.
    (Stated ranges for each sector await the Daehn SI; until then the test checks the
    calibration hits the expected values and that the shares are physical.)"""
    _, cs = run_all
    cal = cs.calibration
    assert cal.max_miss < 1e-3
    assert np.all((cal.phi >= 0) & (cal.phi <= 1))
    got = cs.projection.result.scrap_cu_by_sector.loc[2008].to_numpy()
    assert got == pytest.approx(calibrate.targets(reg), rel=1e-3)


def test_construction_scrap_copper_is_mostly_returning_copper(run_all):
    """The cohort model explains most of Daehn's 0.10 wt% in construction scrap by copper that
    went into construction steel decades earlier, not by fresh attachment."""
    _, cs = run_all
    j = list(link.SECTORS).index("construction")
    assert cs.calibration.bulk_share[j] > 0.5


def test_no_attached_copper_means_primary_concentration_everywhere(reg, prm):
    years = np.arange(1900, 2101)
    products = np.full(len(years), 500.0)
    r = ssf.run(years, products, prm)
    later = r.product_cu_by_sector.loc[1950:]
    assert np.allclose(later, prm.primary_cu, rtol=1e-9)
    assert np.allclose(r.scrap_cu_by_sector.loc[1950:], prm.primary_cu, rtol=1e-9)
    assert r.flows["copper_induced_primary"].max() == pytest.approx(0.0, abs=1e-9)
    assert r.crossover_year() is None


def test_copper_returns_with_its_cohort(prm):
    """A one-year pulse of attached copper raises product copper, which comes back in scrap
    about one lifetime later even though no copper is attached in that later year."""
    years = np.arange(1900, 2101)
    products = np.full(len(years), 500.0)
    attached = np.zeros((len(years), 4))
    attached[years == 1960, :] = 2.0  # Mt Cu, all sectors
    base = ssf.run(years, products, prm)
    pulse = ssf.run(years, products, prm, attached)
    s = list(prm.sectors).index("construction")
    life = int(prm.lifetime[s])
    excess = pulse.flows["cu_eol_bulk"] - base.flows["cu_eol_bulk"]
    assert excess.loc[1950:1960].abs().max() < 1e-12
    assert excess.loc[1960 + life - 10 : 1960 + life + 10].sum() > 0


def test_tolerance_bands_ordered_and_above_primary(prm):
    assert list(prm.tolerance) == sorted(prm.tolerance, reverse=True)
    assert np.all(prm.tolerance > prm.primary_cu)


def test_2008_tolerable_copper_near_daehn(coupled):
    """Daehn et al. 2017 p. 6602: about 1.8 Mt copper could be tolerated in 2008."""
    _, _, sp = coupled
    assert sp.result.flows.loc[2008, "cu_tolerable_aggregate"] == pytest.approx(1.8, rel=0.15)


def test_crossover_methods_return_years_or_none(coupled):
    _, _, sp = coupled
    for kind in ("lp", "aggregate"):
        y = sp.result.crossover_year(kind)
        assert y is None or 2026 <= y <= 2300
    with pytest.raises(ValueError):
        sp.result.crossover_year("bogus")


def test_sector_above_saturation_declines_smoothly(reg):
    """Monte Carlo can draw a saturation level below today's stock; the path then declines."""
    low = reg.with_overrides({"st.saturation_stock_per_capita.metal_goods": 300.0}, label="low")
    pj = cu_demand.project(end_year=2300, reg=low)
    sp = st_coupled.run(pj.result, low).projection
    j = list(link.SECTORS).index("metal_goods")
    path = sp.per_capita_target[sp.result.years > 2025, j]
    assert np.all(np.diff(path) <= 1e-9) and path[-1] == pytest.approx(300.0, rel=0.01)
