"""Phase 3 done conditions: coefficient provenance (3.1), Leontief engine (3.2), maintenance energy
(3.3) and sensitivities (3.4)."""

import logging

import numpy as np
import pytest

from metalwheels import registry, scenarios
from metalwheels.cascade import leontief
from metalwheels.cascade import matrix as cascade


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.fixture(scope="module")
def baseline(reg):
    logging.getLogger("metalwheels").setLevel(logging.ERROR)
    run = scenarios.run_scenario(scenarios.load_scenario("baseline"), reg)
    logging.getLogger("metalwheels").setLevel(logging.INFO)
    return run


@pytest.fixture(scope="module")
def harmsen(baseline, reg):
    return cascade.run(baseline.timeseries, reg, "harmsen")


# --- Task 3.1: coefficient provenance ------------------------------------------------------


def test_every_zero_cell_is_justified_and_no_note_is_stale(harmsen):
    zeros = set(cascade.zero_cells(harmsen.a))
    assert zeros == set(cascade.ZERO_NOTES)
    assert all(len(note) > 10 for note in cascade.ZERO_NOTES.values())


def test_every_nonzero_cell_maps_to_registry_parameters(harmsen, reg):
    nz = {
        (cascade.SECTORS[i], cascade.SECTORS[j])
        for i, j in zip(*np.nonzero(np.any(harmsen.a != 0, axis=0)), strict=True)
    }
    assert nz == set(cascade.CELL_PARAMS)
    for keys in cascade.CELL_PARAMS.values():
        for k in keys:
            assert k in reg, k


#: Cells resting partly on unsourced assumptions (reported by `metalwheels cascade gaps`).
KNOWN_PENDING = {
    "cas.cu.secondary_energy",
    "cas.cu_energy_net_of_inputs",
    "cas.ammonia_per_explosive",
    "cas.st.rock_per_ore",
    "cas.al.smelting_electricity",
    "cas.al.alumina_per_al",
    "cas.ce.limestone_per_clinker",
    "cas.en.coal_per_mj",
}


def test_provenance_gaps_are_only_known_assumptions(reg):
    """Done condition (3.1), with an explicit, reported list of exceptions: every non-zero cell
    has sourced parameters with ranges except the listed assumptions, which are all ranged."""
    gaps = cascade.provenance_gaps(reg)
    flagged = {item.split(" ")[0] for items in gaps.values() for item in items}
    assert flagged == KNOWN_PENDING
    assert all("(no range)" not in item for items in gaps.values() for item in items)
    for k in KNOWN_PENDING - {"cas.cu_energy_net_of_inputs"}:
        p = reg[k]
        assert p.low is not None and p.high is not None, k


# --- Task 3.2: Leontief engine --------------------------------------------------------------


def test_inverse_matches_numpy_and_hand_case():
    a = np.array([[0.0, 0.5], [0.25, 0.0]])
    want = np.array([[1.0, 0.5], [0.25, 1.0]]) / (1 - 0.125)
    assert leontief.leontief_inverse(a) == pytest.approx(want, abs=1e-12)
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = rng.uniform(0, 0.12, (7, 7))
        assert np.allclose(
            leontief.leontief_inverse(m), np.linalg.inv(np.eye(7) - m), rtol=0, atol=1e-9
        )


def test_stacked_inverse(harmsen):
    lm = leontief.leontief_inverse(harmsen.a[:5])
    for t in range(5):
        assert np.allclose(lm[t], np.linalg.inv(np.eye(7) - harmsen.a[t]), atol=1e-9, rtol=1e-12)


def test_rho_warning_and_error_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="metalwheels"):
        leontief.spectral_radius(np.array([[0.5]]))
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger="metalwheels"):
        leontief.spectral_radius(np.array([[0.0, 0.95], [0.95, 0.0]]))
    assert any(r.levelno == logging.WARNING and "0.95" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="metalwheels"):
        leontief.spectral_radius(np.array([[1.02]]))
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_rho_invariant_to_sector_units(harmsen):
    """Changing a sector's unit (t -> kt, MJ -> GJ) is a similarity transform: rho unchanged."""
    a = harmsen.a[-1]
    d = np.diag([1e-3, 1e-6, 1.0, 1e-3, 10.0, 1e-9, 1e-3])
    scaled = d @ a @ np.linalg.inv(d)
    assert leontief.spectral_radius(scaled) == pytest.approx(leontief.spectral_radius(a))


def test_cascade_reproduces_2025_output(harmsen, baseline, reg):
    cf = cascade.Coefficients.from_registry(reg)
    st = cascade.State.from_timeseries(baseline.timeseries, "harmsen")
    k = int(np.flatnonzero(harmsen.years == 2025)[0])
    assert harmsen.x[k] == pytest.approx(cascade.x_base(cf, st), rel=1e-9)
    assert np.all(harmsen.y > 0)
    assert harmsen.maintenance_energy_ej[k] == pytest.approx(600.0, rel=1e-9)


def test_rho_well_below_one(harmsen):
    assert np.all((harmsen.rho > 0) & (harmsen.rho < 0.5))


# --- Task 3.3: maintenance energy -----------------------------------------------------------


def test_static_matrix_gives_constant_energy_equal_to_direct_calculation():
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 0.1, (7, 7))
    y = rng.uniform(1, 10, 7)
    stack = np.repeat(a[None], 30, axis=0)
    series = leontief.maintenance_energy(stack, y, 6)
    direct = (np.linalg.inv(np.eye(7) - a) @ y)[6]
    assert np.allclose(series, direct, rtol=1e-12)
    assert np.ptp(series) == pytest.approx(0.0, abs=1e-9)


def test_baseline_produces_series_for_both_brackets(baseline):
    """Defined until the bracket's first supply shortfall, NaN afterwards (2025 final demand can
    no longer be delivered); rho is defined throughout."""
    ts, s = baseline.timeseries, baseline.summary
    for br in scenarios.BRACKETS:
        e = ts[f"cas_{br}_maintenance_energy_ej"]
        until = int(s[f"cas_{br}_valid_until"][0])
        assert e.loc[2025] == pytest.approx(600.0, rel=1e-6)
        assert e.loc[:until].notna().all() and (e.loc[:until] > 0).all()
        assert e.loc[until + 1 :].isna().all()
        assert ts[f"cas_{br}_rho"].between(0, 1).all()


def test_copper_energy_multiplier_rises_with_declining_grade(baseline):
    m = baseline.timeseries["cas_harmsen_copper_energy_multiplier_gj_per_t"]
    assert m.loc[2300] > m.loc[2025]


# --- Task 3.4: sensitivities ----------------------------------------------------------------


@pytest.mark.parametrize("year", [2025, 2100, 2500])
def test_analytic_elasticities_match_finite_differences(harmsen, year):
    k = int(np.flatnonzero(harmsen.years == year)[0])
    a, y, e = harmsen.a[k], harmsen.y, cascade.IDX["energy"]
    er = leontief.rho_elasticities(a)
    ee = leontief.energy_elasticities(a, y, e)

    def rho(m):
        return leontief.perron_vectors(m)[0]

    def energy(m):
        return leontief.maintenance_energy(m, y, e)

    for i, j in zip(*np.nonzero(a), strict=True):
        fd_r = leontief.finite_difference_elasticity(rho, a, i, j)
        fd_e = leontief.finite_difference_elasticity(energy, a, i, j)
        assert fd_r == pytest.approx(er[i, j], rel=0.01, abs=1e-6), (i, j)
        assert fd_e == pytest.approx(ee[i, j], rel=0.01, abs=1e-6), (i, j)


def test_rho_elasticities_sum_to_one(harmsen):
    """rho is homogeneous of degree 1 in A, so elasticities over all cells sum to 1 (Euler)."""
    assert leontief.rho_elasticities(harmsen.a[-1]).sum() == pytest.approx(1.0, rel=1e-9)


def test_sensitivity_table_ranks_top_drivers(harmsen):
    df = cascade.sensitivity_table(harmsen, 2100)
    assert len(df) == len(cascade.CELL_PARAMS)
    assert list(df["rank"]) == list(range(1, len(df) + 1))
    top5 = df.head(5)
    assert (top5["energy_elasticity"].abs().diff().dropna() <= 1e-12).all()
