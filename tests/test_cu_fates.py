"""Task 1.3 done condition: 2011 flows each within +/-15 % of Helbig Fig. S11; cumulative
landfilled Cu in 2010 within 105-170 Mt; whole-system mass balance < 0.1 % every year."""

import pytest

from metalwheels import registry
from metalwheels.copper import fates
from metalwheels.copper import stockflow as sf

S11_2011 = {  # Mt Cu, Helbig et al. 2020 SI Fig. S11
    "tailings": 3.55,
    "old_scrap": 5.20,
    "other_scrap_markets": 2.30,
    "not_recycled": 3.60,
    "dissipation": 0.45,
}


@pytest.fixture(scope="module")
def run():
    reg = registry.load()
    res = sf.run_historical(reg)
    return res, fates.partition(res, reg.value("cu.tailings_share_of_ore_cu"))


@pytest.mark.parametrize(("flow", "target"), S11_2011.items())
def test_2011_flows_within_15pct(run, flow, target):
    res, media = run
    value = media.loc[2011, "tailings"] if flow == "tailings" else res.flows.loc[2011, flow]
    assert value == pytest.approx(target, rel=0.15)


def test_cumulative_landfill_2010(run):
    _, media = run
    assert 105 <= media.loc[2010, "cum_landfill"] <= 170


def test_mass_balance_every_year(run):
    _, media = run
    assert fates.balance_residual(media).abs().max() < 1e-3


def test_receiving_media_match_s11_nodes(run):
    """ENV.1 = tailings + smelting (3,900 kt); WMR.2 = 2,650 kt (within 15 %)."""
    _, media = run
    assert media.loc[2011, "tailings_and_slag"] == pytest.approx(3.90, rel=0.15)
    assert media.loc[2011, "other_scrap_markets"] == pytest.approx(2.65, rel=0.15)


def test_cumulative_columns_monotonic(run):
    _, media = run
    for m in fates.MEDIA:
        assert media[f"cum_{m}"].is_monotonic_increasing


def test_bad_tailings_share_rejected(run):
    res, _ = run
    with pytest.raises(ValueError):
        fates.partition(res, 1.0)
