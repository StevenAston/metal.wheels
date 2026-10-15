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
    assert supply == pytest.approx(290 * 0.002 + 710 * CP)
    assert tolerable == pytest.approx(500 * 0.004 + 500 * 0.0006)
