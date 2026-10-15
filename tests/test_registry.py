"""Task 0.2 done condition: parameter registry with provenance."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from metalwheels import registry
from metalwheels.cli import app

GOOD = {
    "value": 0.5,
    "unit": "1",
    "status": "sourced",
    "source": "some_key",
    "locator": "Table 1",
}


def _param(**overrides):
    fields = {**GOOD, **overrides}
    fields = {k: v for k, v in fields.items() if v is not None}
    return registry.Param(key="x", domain="test", **fields)


def test_valid_parameter_passes():
    assert _param().value == 0.5


def test_missing_unit_raises():
    fields = {k: v for k, v in GOOD.items() if k != "unit"}
    with pytest.raises(ValueError, match="unit"):
        registry.Param(key="x", domain="test", **fields)


def test_empty_or_unknown_unit_raises():
    with pytest.raises(ValueError):
        _param(unit="")
    with pytest.raises(ValueError, match="ALLOWED_UNITS"):
        _param(unit="percent")


@pytest.mark.parametrize("status", ["sourced", "derived"])
def test_missing_source_raises_unless_assumption(status):
    with pytest.raises(ValueError, match="source"):
        _param(status=status, source=None)


def test_assumption_may_omit_source():
    p = _param(status="assumption", source=None, locator=None)
    assert p.source is None


def test_sourced_requires_locator():
    with pytest.raises(ValueError, match="locator"):
        _param(locator=None)


@pytest.mark.parametrize(
    ("low", "value", "high"),
    [(0.6, 0.5, 0.9), (0.1, 0.5, 0.4), (0.9, 0.5, 0.1)],
)
def test_bounds_enforced(low, value, high):
    with pytest.raises(ValueError):
        _param(low=low, value=value, high=high)


def test_bounds_inclusive():
    assert _param(low=0.5, value=0.5, high=0.5).value == 0.5


def test_non_fixed_distribution_requires_bounds():
    with pytest.raises(ValueError, match="requires both low and high"):
        _param(distribution="uniform", low=0.1)


def test_unknown_field_rejected():
    with pytest.raises(ValueError):
        _param(valeu=1.0)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_loader_reports_file_and_key(tmp_path):
    _write(tmp_path, "bad.yaml", "domain: d\nparameters:\n  k1: {value: 1, status: assumption}\n")
    with pytest.raises(registry.RegistryError, match=r"bad\.yaml: k1"):
        registry.load(tmp_path)


def test_loader_rejects_duplicate_keys(tmp_path):
    body = "domain: d\nparameters:\n  k1: {value: 1, unit: '1', status: assumption}\n"
    _write(tmp_path, "a.yaml", body)
    _write(tmp_path, "b.yaml", body)
    with pytest.raises(registry.RegistryError, match="duplicate"):
        registry.load(tmp_path)


def test_loader_rejects_missing_top_level(tmp_path):
    _write(tmp_path, "x.yaml", "parameters: {}\n")
    with pytest.raises(registry.RegistryError, match="domain"):
        registry.load(tmp_path)


# --- The real registry -------------------------------------------------------------------

#: Parameters Phase 1 needs that are already known. Later tasks add to the registry; this set
#: guards against accidental deletion of the core.
PHASE1_CORE = {
    "cu.eol_collection_rate",
    "cu.eol_processing_rate",
    "cu.tailings_share_of_ore_cu",
    "cu.production_yield",
    "cu.fabrication_loss_fraction",
    "cu.in_use_dissipation_fraction",
    "cu.lifetime_sigma_fraction",
    "geo.grade.gerst.mu_w",
    "geo.grade.gerst.sigma2_w",
    "geo.grade.kesler_ext.mu_w",
    "geo.grade.kesler_ext.sigma2_w",
    "geo.endow.harmsen.layer_0_1km",
    "geo.endow.kesler.above_3_3km",
    "geo.tailings_dry_density",
    "en.norgate.coarse.a",
    "en.norgate.coarse.b",
    "en.norgate.fine.a",
    "en.norgate.fine.b",
    "en.depth.reference_depth",
    "en.depth.energy_per_m",
    "en.koppelaar.surface.a",
    "en.koppelaar.underground.b",
    "dem.saturation_stock_per_capita",
}


@pytest.fixture(scope="module")
def reg():
    return registry.load()


def test_real_registry_loads(reg):
    assert set(reg) >= PHASE1_CORE


def test_sixteen_product_groups(reg):
    lifetimes = [k for k in reg if k.startswith("cu.lifetime.")]
    effs = [k for k in reg if k.startswith("cu.fab_efficiency.")]
    assert len(lifetimes) == len(effs) == 16
    assert {k.split(".")[-1] for k in lifetimes} == {k.split(".")[-1] for k in effs}


def test_harmsen_layers_sum_to_table_total(reg):
    layers = [p.value for k, p in reg.items() if k.startswith("geo.endow.harmsen.layer_")]
    assert sum(layers) == pytest.approx(24_890, abs=1)  # Table 1 total 2.49E+04 Mt


def test_gerst_median(reg):
    import math

    assert math.exp(reg.value("geo.grade.gerst.mu_w")) == pytest.approx(0.52, abs=0.01)


def test_norgate_fits_reproduce_source_points(reg):
    for opt, targets in {"coarse": (72, 131, 249), "fine": (129, 244, 474)}.items():
        a, b = reg.value(f"en.norgate.{opt}.a"), reg.value(f"en.norgate.{opt}.b")
        for g, target in zip((0.01, 0.005, 0.0025), targets, strict=True):
            assert a + b / g == pytest.approx(target, rel=1e-9)


def test_cli_params_list_shows_every_parameter(reg):
    result = CliRunner().invoke(app, ["params", "list"])
    assert result.exit_code == 0, result.output
    for key in reg:
        assert key in result.stdout


def test_cli_params_list_domain_filter():
    result = CliRunner().invoke(app, ["params", "list", "--domain", "energy"])
    assert result.exit_code == 0
    assert "en.norgate.coarse.a" in result.stdout
    assert "cu.eol_collection_rate" not in result.stdout


def test_cli_params_list_unknown_domain_fails():
    result = CliRunner().invoke(app, ["params", "list", "--domain", "nope"])
    assert result.exit_code == 1
