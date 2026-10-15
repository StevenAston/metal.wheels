"""Tasks 2.1-2.2: steel stock model with copper tracking, and the copper -> steel link."""

import logging

import numpy as np
import pytest

from metalwheels import registry
from metalwheels.copper import demand as cu_demand
from metalwheels.copper import stockflow as csf
from metalwheels.steel import demand as st_demand
from metalwheels.steel import link
from metalwheels.steel import stockflow as ssf

logging.getLogger("metalwheels").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.fixture(scope="module")
def prm(reg):
    return ssf.SteelParams.from_registry(reg)


@pytest.fixture(scope="module")
def coupled(reg):
    pj = cu_demand.project(end_year=2300, reg=reg)
    lk = link.route(pj.result, link.link_shares(reg))
    sp = st_demand.project(pj.result.years, reg, lk.to_steel.to_numpy())
    return pj, lk, sp


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


def test_link_shares_bounds(reg):
    phi = link.link_shares(reg)
    assert phi.shape == (4,) and np.all((phi >= 0) & (phi <= 1))


def test_every_copper_category_is_mapped(reg):
    cats = {g.split("_", 1)[0] for g in csf.CopperParams.from_registry(reg).groups}
    mapped = {c for cs in link.CU_CATEGORY_MAP.values() for c in cs}
    assert cats == mapped


def test_full_processing_sends_no_copper_to_steel(reg):
    pj = cu_demand.project(
        end_year=2100, reg=reg, policy={"start_year": 2026, "processing": 1.0}
    )  # fmt: skip
    lk = link.route(pj.result, link.link_shares(reg))
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


def test_demand_driven_stock_hits_target(coupled):
    _, _, sp = coupled
    f = sp.result.flows
    fut = f["demand_driven"] > 0
    assert np.allclose(f.loc[fut, "stock"], f.loc[fut, "target_stock"], rtol=1e-9)


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
