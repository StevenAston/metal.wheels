"""Task 1.2 done condition: in-use stock 2010 within 320-380 Mt; EoL outflow 2011 within
11.1 +/- 1.5 Mt (Glöser-Chahoud Fig. 5.8/5.12); mass balance < 1e-6 relative every year."""

import numpy as np
import pytest

from metalwheels import registry
from metalwheels.copper import stockflow as sf


@pytest.fixture(scope="module")
def result():
    return sf.run_historical()


@pytest.fixture(scope="module")
def prm():
    return sf.CopperParams.from_registry(registry.load())


def test_stock_2010_in_range(result):
    assert 320 <= result.flows.loc[2010, "stock"] <= 380


def test_eol_2011_in_range(result):
    assert result.flows.loc[2011, "eol"] == pytest.approx(11.1, abs=1.5)


def test_stock_balance_every_year(result):
    assert result.stock_residual().abs().max() < 1e-6


def test_whole_system_balance_every_year(result):
    assert result.balance_residual().abs().max() < 1e-6


def test_group_stocks_sum_and_nonnegative(result):
    s = result.stock_by_group
    assert (s.values >= -1e-12).all()
    np.testing.assert_allclose(s.sum(axis=1), result.flows["stock"], rtol=1e-12)


def test_sixteen_groups_and_shares(prm):
    assert len(prm.groups) == 16
    assert prm.share.sum() == pytest.approx(1.0)
    by_cat = {c: prm.share[[k == c for k in prm.category]].sum() for c in sf.CATEGORIES}
    assert by_cat == pytest.approx(
        {
            "construction": 0.28,
            "infrastructure": 0.16,
            "industry": 0.12,
            "transport": 0.13,
            "consumer": 0.31,
        }
    )


def test_lifetime_pmf_normalised_and_centred(prm):
    pmf = sf.lifetime_pmf(prm.lifetime, prm.sigma_frac, 200)
    np.testing.assert_allclose(pmf.sum(axis=1), 1.0, rtol=1e-12)
    ages = np.arange(1, 201)
    np.testing.assert_allclose(pmf @ ages, prm.lifetime, rtol=0.01)


def test_single_pulse_retires_with_expected_lifetime(prm):
    """A one-year pulse of mine output leaves use with the share-weighted mean lifetime."""
    years = np.arange(2000, 2200)
    mine = np.zeros(len(years))
    mine[0] = 1.0
    no_recycling = sf.CopperParams(**{**prm.__dict__, "collection": 0.0})
    r = sf.run(years, mine, no_recycling)
    eol = r.flows["eol"].to_numpy()
    mean_age = (eol * (years - years[0])).sum() / eol.sum()
    goods = prm.fab_eff * prm.share
    expected = (goods * prm.lifetime).sum() / goods.sum()
    assert mean_age == pytest.approx(expected, rel=0.01)
    assert r.flows["stock"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_no_scrap_without_collection(prm):
    years = np.arange(1900, 2026)
    mine = np.ones(len(years))
    r = sf.run(years, mine, sf.CopperParams(**{**prm.__dict__, "collection": 0.0}))
    assert r.flows["old_scrap"].abs().max() == 0.0
    assert (r.flows["new_scrap"] > 0).all()  # the new-scrap loop still runs


def test_overrides_validated():
    with pytest.raises(ValueError, match="unknown overrides"):
        sf.CopperParams.from_registry(registry.load(), colection=0.9)


def test_2011_flows_close_to_helbig_s11(result):
    """Diagnostic for Task 1.3: key 2011 flows within 10 % of Helbig 2020 Fig. S11 (Mt)."""
    s11 = {
        "refined": 19.70,
        "semis": 25.25,
        "final_goods": 20.85,
        "old_scrap": 5.20,
        "other_scrap_markets": 2.30,
        "not_recycled": 3.60,
        "dissipation": 0.45,
    }
    for k, v in s11.items():
        assert result.flows.loc[2011, k] == pytest.approx(v, rel=0.10), k
