"""Task 1.8 done condition: `metalwheels run baseline` completes 1900-3025 in < 60 s on one core
and writes outputs matching the declared schema; each named scenario runs without error."""

import json
import time

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from metalwheels import export, registry, scenarios
from metalwheels.cli import app


@pytest.fixture(scope="module")
def baseline_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "baseline"
    t0 = time.perf_counter()
    result = CliRunner().invoke(app, ["run", "baseline", "--out", str(out)])
    elapsed = time.perf_counter() - t0
    assert result.exit_code == 0, result.output
    return out, elapsed


def test_baseline_runtime_under_60s(baseline_dir):
    _, elapsed = baseline_dir
    assert elapsed < 60


def test_outputs_match_schema(baseline_dir):
    out, _ = baseline_dir
    ts = pd.read_parquet(out / "timeseries.parquet")
    export.validate(ts)  # raises on any mismatch
    assert ts.index.min() == 1900 and ts.index.max() == 3025
    assert len(ts) == 1126


def test_summary_and_meta_written(baseline_dir):
    out, _ = baseline_dir
    summary = pd.read_csv(out / "summary.csv")
    assert {"metric", "value", "unit"} == set(summary.columns)
    assert "harmsen_exhausted_year" in set(summary["metric"])
    meta = json.loads((out / "meta.json").read_text())
    assert meta["scenario"] == "baseline"
    assert len(meta["registry_sha256"]) == 64


@pytest.mark.parametrize("name", scenarios.list_scenarios())
def test_each_scenario_runs(name):
    sc = scenarios.load_scenario(name)
    run = scenarios.run_scenario(sc)
    export.validate(run.timeseries)
    assert run.timeseries.index.max() == sc.end_year


def test_policy_leaves_history_unchanged():
    base = scenarios.run_scenario(scenarios.load_scenario("baseline")).timeseries
    pol = scenarios.run_scenario(scenarios.load_scenario("recycling_95")).timeseries
    cols = ["mine", "stock", "eol", "old_scrap", "harmsen_grade_pct"]
    pd.testing.assert_frame_equal(base.loc[:2025, cols], pol.loc[:2025, cols])
    assert pol.loc[2100, "mine"] < base.loc[2100, "mine"]


def test_policy_ramp():
    from metalwheels.copper import demand
    from metalwheels.copper import stockflow as sf

    prm = sf.CopperParams.from_registry(registry.load())
    years = np.arange(2020, 2040)
    c, p = demand.policy_rates(
        years, prm, {"start_year": 2026, "ramp_years": 10, "collection": 0.9}
    )
    assert c[years == 2025][0] == pytest.approx(prm.collection)
    assert c[years == 2030][0] == pytest.approx(prm.collection + 0.5 * (0.9 - prm.collection))
    assert c[years == 2039][0] == pytest.approx(0.9)
    np.testing.assert_allclose(p, prm.processing)
    with pytest.raises(ValueError, match="after the historical period"):
        demand.policy_rates(years, prm, {"start_year": 2000, "collection": 0.9})


def test_registry_overrides_bypass_bounds_and_reject_unknown():
    reg = registry.load()
    r2 = reg.with_overrides({"cu.eol_collection_rate": 0.98}, label="test")
    p = r2["cu.eol_collection_rate"]
    assert p.value == 0.98 and p.high is None and p.distribution == "fixed"
    assert reg["cu.eol_collection_rate"].value == 0.67  # original untouched
    with pytest.raises(KeyError):
        reg.with_overrides({"cu.nope": 1.0})


def test_bad_scenario_files(tmp_path):
    (tmp_path / "x.yaml").write_text("name: y\n")
    with pytest.raises(ValueError, match="does not match"):
        scenarios.load_scenario("x", directory=tmp_path)
    (tmp_path / "z.yaml").write_text("name: z\noptions: {energy_option: magic}\n")
    with pytest.raises(ValueError, match="energy option"):
        scenarios.load_scenario("z", directory=tmp_path)
    with pytest.raises(FileNotFoundError):
        scenarios.load_scenario("missing", directory=tmp_path)
