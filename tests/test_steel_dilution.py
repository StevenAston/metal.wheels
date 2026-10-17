"""Task 2.3 core: dilution share and the scrap allocation LP (hand-computed cases)."""

import numpy as np
import pytest

from metalwheels.steel import dilution as dl

CP = 0.0002  # 0.02 wt% primary steel (Daehn et al. 2017 Fig. 1)


def test_plan_case_0p4_to_0p1():
    # (0.4 - 0.1) / (0.4 - 0.02) = 0.789...
    assert dl.dilution_share(0.004, 0.001, CP) == pytest.approx(0.3 / 0.38, rel=1e-12)
    assert round(dl.dilution_share(0.004, 0.001, CP), 2) == 0.79


def test_no_dilution_when_scrap_within_tolerance():
    assert dl.dilution_share(0.001, 0.004, CP) == 0.0


def test_vectorised():
    x = dl.dilution_share(np.array([0.004, 0.002, 0.0005]), 0.001, CP)
    assert x == pytest.approx([0.3 / 0.38, 0.1 / 0.18, 0.0])


def test_primary_must_be_cleaner_than_tolerance():
    with pytest.raises(ValueError):
        dl.dilution_share(0.004, 0.0002, 0.0002)


def test_single_source_single_sink_matches_dilution_share():
    a = dl.allocate([100.0], [0.004], [100.0], [0.001], CP)
    x = dl.dilution_share(0.004, 0.001, CP)
    assert a.primary_total == pytest.approx(100 * x, rel=1e-6)
    assert a.scrap_used[0] == pytest.approx(100 * (1 - x), rel=1e-6)
    assert a.sink_concentration[0] == pytest.approx(0.001, rel=1e-6)
    # Scrap equals demand, so the mass-only requirement is zero: all primary is copper-induced.
    assert a.mass_only_primary == pytest.approx(0.0, abs=1e-9)
    assert a.copper_induced_primary == pytest.approx(100 * x, rel=1e-6)
    assert a.unusable_scrap == pytest.approx(100 * x, rel=1e-6)


def test_tolerant_sink_absorbs_dirty_scrap():
    # 50 Mt scrap at 0.3 %; sinks: 50 Mt rebar at 0.4 %, 100 Mt flats at 0.06 %.
    a = dl.allocate([50.0], [0.003], [50.0, 100.0], [0.004, 0.0006], CP)
    assert a.scrap_used[0] == pytest.approx(50.0, rel=1e-6)
    assert a.primary_total == pytest.approx(100.0, rel=1e-6)
    assert a.copper_induced_primary == pytest.approx(0.0, abs=1e-6)
    assert np.all(a.sink_concentration <= np.array([0.004, 0.0006]) + 1e-12)


def test_dirtiest_scrap_left_over():
    # Two scrap grades, one sink that cannot take all of either.
    a = dl.allocate([60.0, 60.0], [0.001, 0.004], [100.0], [0.0015], CP)
    assert a.scrap_unused[1] > a.scrap_unused[0]
    # Hand calc: all clean scrap (60) + y dirty + p primary = 100;
    # 0.001*60 + 0.004 y + 0.0002 p = 0.15 -> y = (0.15 - 0.06 - 0.0002*(40 - y)) / 0.004
    y = (0.15 - 0.06 - 0.0002 * 40) / (0.004 - 0.0002)
    assert a.scrap_used[0] == pytest.approx(60.0, rel=1e-6)
    assert a.scrap_used[1] == pytest.approx(y, rel=1e-6)
    assert a.primary_total == pytest.approx(40 - y, rel=1e-6)


def test_surplus_mass_is_not_counted_as_copper_unusable():
    a = dl.allocate([200.0], [0.0005], [100.0], [0.004], CP)
    assert a.primary_total == pytest.approx(0.0, abs=1e-9)
    assert a.surplus_scrap == pytest.approx(100.0)
    assert a.unusable_scrap == pytest.approx(0.0, abs=1e-9)


def test_mass_and_copper_conservation():
    rng = np.random.default_rng(1)
    s = rng.uniform(10, 100, 4)
    c = rng.uniform(0.0005, 0.004, 4)
    m = rng.uniform(50, 300, 6)
    tau = rng.choice([0.0006, 0.0008, 0.00125, 0.00175, 0.0025, 0.004], 6)
    a = dl.allocate(s, c, m, tau, CP)
    assert a.scrap_used + a.scrap_unused == pytest.approx(s)
    assert a.sink_mass == pytest.approx(m, rel=1e-9)
    assert a.sink_copper.sum() == pytest.approx(c @ a.scrap_used + CP * a.primary_total)
    assert np.all(a.sink_copper <= tau * m + 1e-9)


def test_aggregate_copper_balance():
    supply, tolerable = dl.copper_balance([290.0], [0.002], [500.0, 500.0], [0.004, 0.0006], CP)
    assert supply == pytest.approx(290 * 0.002)
    assert tolerable == pytest.approx(500 * 0.004 + 500 * 0.0006)


# --- Task 2.3 done condition: crossover with Daehn-equivalent inputs -------------------------
# Iron sources and end products read from Daehn et al. 2017 Fig. 3 (best-estimate case built on
# Pauliuk et al. 2013a): (Mt, wt% Cu) steps, read from the figure to about +/- 10 Mt.
DAEHN_FIG3 = {
    2040: {
        "sources": [(220, 0.30), (180, 0.30), (190, 0.20), (210, 0.175)],
        "pig_iron": 730,
        "products": [(220, 0.4), (110, 0.25), (160, 0.175), (560, 0.125), (210, 0.08), (270, 0.06)],
    },
    2050: {
        "sources": [(300, 0.30), (180, 0.30), (230, 0.20), (275, 0.175)],
        "pig_iron": 335,
        "products": [(200, 0.4), (110, 0.25), (150, 0.175), (670, 0.125), (140, 0.08), (50, 0.06)],
    },
}


def _daehn_case(year):
    d = DAEHN_FIG3[year]
    s = np.array([m for m, _ in d["sources"]], float)
    c = np.array([w for _, w in d["sources"]]) / 100
    m = np.array([m for m, _ in d["products"]], float)
    tau = np.array([w for _, w in d["products"]]) / 100
    return s, c, m, tau


def test_daehn_equivalent_crossover_between_2040_and_2060():
    """Daehn's best estimate puts the crossover at about 2050; our balance and LP, fed the same
    inputs, cross between the 2040 and 2050 snapshots."""
    gap = {}
    for year in DAEHN_FIG3:
        s, c, m, tau = _daehn_case(year)
        supply, tolerable = dl.copper_balance(s, c, m, tau, CP)
        gap[year] = supply - tolerable
        unusable = dl.allocate(s, c, m, tau, CP).unusable_scrap
        assert (unusable > 0) == (gap[year] > 0), year
    assert gap[2040] < 0 < gap[2050]
    crossing = 2040 + 10 * (-gap[2040]) / (gap[2050] - gap[2040])
    assert 2040 <= crossing <= 2060


def test_daehn_2008_copper_in_system():
    """Daehn et al. 2017 p. 6602: about 0.7 Mt copper in the 2008 system (scrap), 1.8 Mt
    tolerable.
    2008 Fig. 3 steps: transport 125 Mt + goods 50 Mt at 0.3 %, industrial 70 Mt at 0.2 %,
    construction 70 Mt at 0.1 %; pig iron to 1,080 Mt; products as read below."""
    s = np.array([125.0, 50.0, 70.0, 70.0])
    c = np.array([0.003, 0.003, 0.002, 0.001])
    m = np.array([185.0, 85.0, 120.0, 360.0, 170.0, 160.0])
    tau = np.array([0.004, 0.0025, 0.00175, 0.00125, 0.0008, 0.0006])
    supply, tolerable = dl.copper_balance(s, c, m, tau, CP)
    assert supply == pytest.approx(0.7, rel=0.15)
    assert tolerable == pytest.approx(1.8, rel=0.15)


def test_mixed_spreads_copper_pinch_concentrates_it():
    s, c, m, tau = [50.0], [0.003], [100.0, 100.0], [0.004, 0.0006]
    mixed = dl.allocate(s, c, m, tau, CP, mode="mixed")
    pinch = dl.allocate(s, c, m, tau, CP, mode="pinch")
    # Same primary and scrap use either way.
    assert mixed.primary_total == pytest.approx(pinch.primary_total, rel=1e-9)
    assert mixed.scrap_used.sum() == pytest.approx(50.0, rel=1e-9)
    # Mixed: total Cu 0.15 + 150 x CP = 0.18 Mt over 200 Mt; tight sink capped at 0.06 %.
    assert mixed.sink_concentration[1] == pytest.approx(0.0006, rel=1e-6)
    assert mixed.sink_concentration[0] == pytest.approx((0.18 - 0.06) / 100, rel=1e-6)
    # Pinch: all scrap into the tolerant sink.
    assert pinch.sink_concentration[0] == pytest.approx((0.15 + 50 * CP) / 100, rel=1e-6)
    assert pinch.sink_concentration[1] == pytest.approx(CP, rel=1e-6)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        dl.allocate([1.0], [0.003], [1.0], [0.004], CP, mode="bogus")
