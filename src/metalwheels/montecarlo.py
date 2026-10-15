"""Task 1.9: Monte Carlo over registry parameter ranges.

* Every parameter whose ``distribution`` is not ``fixed`` is sampled (see registry docstring for
  semantics: uniform on [low, high]; triangular with mode ``value``; normal with [low, high] as
  the central 95 % interval, clipped to [low, high]).
* Draw ``i`` uses its own generator from ``SeedSequence(seed).spawn(n)[i]``, so results do not
  depend on how draws are scheduled across workers: the same seed gives bit-identical output.
* Draws run in a process pool capped at ``max(1, n_cpu - 2)`` workers by default. Worker
  processes are started with single-threaded maths libraries so the cap really bounds CPU use.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd

from metalwheels import registry as _registry
from metalwheels.logging_setup import get_logger
from metalwheels.registry import Registry
from metalwheels.scenarios import Scenario, run_scenario

log = get_logger(__name__)

#: Yearly series for which 5/50/95 % bands are produced.
BAND_SERIES = (
    "mine",
    "refined",
    "stock",
    "stock_per_capita",
    "eol",
    "old_scrap",
    "harmsen_grade_pct",
    "kesler_grade_pct",
    "harmsen_depth_m",
    "kesler_depth_m",
    "harmsen_energy_intensity_mj_per_kg",
    "kesler_energy_intensity_mj_per_kg",
    "harmsen_primary_energy_ej",
    "kesler_primary_energy_ej",
    "harmsen_cum_tailings_volume_km3",
    "kesler_cum_tailings_volume_km3",
    "harmsen_remaining_total",
    "kesler_remaining_total",
)
QUANTILES = (0.05, 0.50, 0.95)
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def default_workers() -> int:
    return max(1, (os.cpu_count() or 1) - 2)


def resolve_workers(requested: int | None) -> int:
    """Workers to use: the default cap unless explicitly requested; never above the CPU count."""
    n = default_workers() if requested is None else int(requested)
    if n < 1:
        raise ValueError("workers must be >= 1")
    cpus = os.cpu_count() or 1
    if n > cpus:
        log.warning("requested %d workers but only %d CPUs; using %d", n, cpus, cpus)
        n = cpus
    if n > default_workers():
        log.warning("using %d workers, above the default cap of %d", n, default_workers())
    return n


def sampled_keys(reg: Registry) -> list[str]:
    return sorted(k for k, p in reg.items() if p.distribution != "fixed")


def sample(
    reg: Registry, rng: np.random.Generator, keys: list[str] | None = None
) -> dict[str, float]:
    """One draw of every non-fixed parameter (or of ``keys``)."""
    out: dict[str, float] = {}
    for k in keys if keys is not None else sampled_keys(reg):
        p = reg[k]
        lo, hi = float(p.low), float(p.high)
        if p.distribution == "uniform":
            v = rng.uniform(lo, hi)
        elif p.distribution == "triangular":
            v = rng.triangular(lo, p.value, hi) if hi > lo else lo
        elif p.distribution == "normal":
            sd = (hi - lo) / (2 * 1.959963984540054)
            v = float(np.clip(rng.normal(p.value, sd), lo, hi))
        else:
            raise ValueError(f"cannot sample {k} with distribution {p.distribution!r}")
        out[k] = float(v)
    return out


def _draw(args: tuple) -> tuple[int, dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    """Run one draw (executed in a worker process)."""
    i, seed_seq, sc, reg = args
    rng = np.random.default_rng(seed_seq)
    values = sample(reg, rng)
    run = run_scenario(sc, reg.with_overrides(values, label=f"mc{i}"))
    summary = {k: v for k, (v, _unit) in run.summary.items()}
    series = {c: run.timeseries[c].to_numpy(dtype=float) for c in BAND_SERIES}
    return i, values, summary, series


def _init_worker() -> None:
    import logging

    logging.getLogger("metalwheels").setLevel(logging.ERROR)  # range warnings repeat per draw


@dataclass
class MonteCarloResult:
    scenario: str
    seed: int
    n_draws: int
    workers: int
    years: np.ndarray
    draws: pd.DataFrame  # one row per draw: sampled values + summary metrics
    bands: pd.DataFrame  # index year; columns "<series>_p05/_p50/_p95"
    summary_quantiles: pd.DataFrame  # index metric; columns p05/p50/p95


def run(
    sc: Scenario,
    n_draws: int = 1000,
    seed: int = 20261015,
    workers: int | None = None,
    reg: Registry | None = None,
    progress=None,
) -> MonteCarloResult:
    reg = (reg or _registry.load()).with_overrides(sc.overrides, label=sc.name)
    n_workers = resolve_workers(workers)
    seqs = np.random.SeedSequence(seed).spawn(n_draws)
    jobs = [(i, seqs[i], sc, reg) for i in range(n_draws)]
    results = [None] * n_draws

    saved = {v: os.environ.get(v) for v in _THREAD_VARS}
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    try:
        if n_workers == 1:
            _init_worker()
            it = map(_draw, jobs)
            for res in it:
                results[res[0]] = res
                if progress:
                    progress()
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=get_context("spawn"), initializer=_init_worker
            ) as pool:
                for res in pool.map(_draw, jobs, chunksize=max(1, n_draws // (n_workers * 8))):
                    results[res[0]] = res
                    if progress:
                        progress()
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old

    n_years = len(results[0][3][BAND_SERIES[0]])
    years = np.arange(sc.end_year - n_years + 1, sc.end_year + 1)
    rows = []
    for i, values, summary, _ in results:
        rows.append({"draw": i, **{f"param:{k}": v for k, v in values.items()}, **summary})
    draws = pd.DataFrame(rows).set_index("draw")

    band_cols = {}
    for name in BAND_SERIES:
        arr = np.vstack([r[3][name] for r in results])
        with warnings.catch_warnings():  # years after exhaustion are NaN in every draw
            warnings.simplefilter("ignore", RuntimeWarning)
            qs = np.nanquantile(arr, QUANTILES, axis=0)
        for q, row in zip(QUANTILES, qs, strict=True):
            band_cols[f"{name}_p{round(q * 100):02d}"] = row
    bands = pd.DataFrame(band_cols, index=pd.Index(years, name="year"))

    metric_cols = [c for c in draws.columns if not c.startswith("param:")]
    sq = draws[metric_cols].quantile(list(QUANTILES)).T
    sq.columns = [f"p{round(q * 100):02d}" for q in QUANTILES]
    sq.index.name = "metric"
    return MonteCarloResult(sc.name, seed, n_draws, n_workers, years, draws, bands, sq)


def write(res: MonteCarloResult, out_dir) -> None:
    from pathlib import Path

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    res.bands.to_parquet(d / "bands.parquet")
    res.draws.to_parquet(d / "draws.parquet")
    res.summary_quantiles.to_csv(d / "summary_quantiles.csv", float_format="%.6g")
    (d / "mc_meta.txt").write_text(
        f"scenario={res.scenario}\nseed={res.seed}\ndraws={res.n_draws}\nworkers={res.workers}\n",
        encoding="utf-8",
    )
