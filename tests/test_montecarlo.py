"""Task 1.9 done condition: 1,000 draws complete; worker count <= configured cap; identical seed
gives bit-identical summary; bands written. (The 1,000-draw run is exercised via the CLI and
recorded in the change log; tests use small draw counts for speed.)"""

import os

import numpy as np
import pandas as pd
import pytest

from metalwheels import montecarlo, registry, scenarios


@pytest.fixture(scope="module")
def sc():
    return scenarios.load_scenario("baseline")


def test_default_workers_respect_cap():
    cap = max(1, (os.cpu_count() or 1) - 2)
    assert montecarlo.default_workers() == cap
    assert montecarlo.resolve_workers(None) <= cap


def test_requested_workers_never_exceed_cpus():
    assert montecarlo.resolve_workers(10_000) <= (os.cpu_count() or 1)
    with pytest.raises(ValueError):
        montecarlo.resolve_workers(0)


def test_pool_uses_resolved_worker_count(sc, monkeypatch):
    seen = {}
    real = montecarlo.ProcessPoolExecutor

    class Spy(real):
        def __init__(self, *a, **kw):
            seen["max_workers"] = kw.get("max_workers")
            super().__init__(*a, **kw)

    monkeypatch.setattr(montecarlo, "ProcessPoolExecutor", Spy)
    workers = 2 if (os.cpu_count() or 1) >= 2 else 1
    res = montecarlo.run(sc, n_draws=2, seed=1, workers=workers)
    assert res.workers == workers
    if workers > 1:
        assert seen["max_workers"] == workers


def test_same_seed_bit_identical(sc):
    a = montecarlo.run(sc, n_draws=3, seed=123, workers=1)
    b = montecarlo.run(sc, n_draws=3, seed=123, workers=1)
    pd.testing.assert_frame_equal(a.summary_quantiles, b.summary_quantiles, check_exact=True)
    pd.testing.assert_frame_equal(a.bands, b.bands, check_exact=True)
    c = montecarlo.run(sc, n_draws=3, seed=124, workers=1)
    assert not a.draws.equals(c.draws)


def test_draws_independent_of_scheduling(sc):
    """Draw i is identical whether run alone or in a batch (per-draw seed streams)."""
    batch = montecarlo.run(sc, n_draws=3, seed=7, workers=1)
    seqs = np.random.SeedSequence(7).spawn(3)
    reg = registry.load()
    v2 = montecarlo.sample(reg, np.random.default_rng(seqs[2]))
    for k, v in v2.items():
        assert batch.draws.loc[2, f"param:{k}"] == v


def test_samples_within_bounds():
    reg = registry.load()
    rng = np.random.default_rng(0)
    for _ in range(200):
        for k, v in montecarlo.sample(reg, rng).items():
            p = reg[k]
            assert p.low <= v <= p.high, k


def test_bands_written_and_ordered(sc, tmp_path):
    res = montecarlo.run(sc, n_draws=4, seed=5, workers=1)
    montecarlo.write(res, tmp_path)
    bands = pd.read_parquet(tmp_path / "bands.parquet")
    assert (tmp_path / "summary_quantiles.csv").is_file()
    assert bands.index.min() == 1900 and bands.index.max() == 3025
    for name in montecarlo.BAND_SERIES:
        lo, mid, hi = (bands[f"{name}_p{q}"] for q in ("05", "50", "95"))
        ok = lo.isna() | ((lo <= mid + 1e-12) & (mid <= hi + 1e-12))
        assert ok.all(), name
