"""Task 4.1 done condition: the catalogue validates against its schema; >= 10 sourced entries."""

import copy

import pytest
import yaml

from metalwheels import references, registry
from metalwheels.ratchets import catalogue
from metalwheels.ratchets import metal_wheel as mw


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.fixture(scope="module")
def cat(reg):
    return catalogue.load(reg=reg)


def test_catalogue_validates_with_ten_sourced_entries(cat):
    assert len(cat) >= 10
    sourced = [r for r in cat if r.sourced]
    assert len(sourced) >= 10


def test_every_mechanism_type_is_represented(cat):
    assert {r.mechanism for r in cat} == set(catalogue.MECHANISMS)


def test_plan_seed_ratchets_present(cat):
    """PLAN.md seeds: Cu and Sn in steel, Fe/Si in aluminium, Zn to EAF dust, glass colour
    mixing, polymer blending, PGM dissipation."""
    seeds = {
        "cu_in_steel", "sn_in_steel", "fe_in_aluminium", "si_in_aluminium", "zn_to_eaf_dust",
        "glass_colour_mixing", "polymer_blending", "pt_dissipation",
    }  # fmt: skip
    assert seeds <= set(cat.by_id)


def test_every_reference_resolves(cat, reg):
    bib = references.load()
    for r in cat:
        assert r.source in bib
        for k in r.param_keys():
            assert k in reg, (r.id, k)
            p = reg[k]
            assert p.status == "assumption" or p.source in bib


def test_energy_steps_and_tolerances(cat):
    v, lo, hi, unit = cat.energy_step("cu_in_steel")
    assert (v, unit) == (pytest.approx(12.6), "MJ/kg")
    assert lo <= v <= hi
    v, lo, hi, _ = cat.energy_step("glass_colour_mixing")
    assert v == pytest.approx(7.5 * 0.275)
    assert lo == pytest.approx(5.2 * 0.25) and hi == pytest.approx(8.7 * 0.30)
    assert cat.energy_step("polymer_blending") is None
    tol = cat.tolerance("cu_in_steel")
    assert tol == (pytest.approx(0.001), pytest.approx(0.0006), pytest.approx(0.004))
    assert cat.tolerance("sn_in_steel")[0] == pytest.approx(0.0004)
    for r in cat:
        es = cat.energy_step(r.id)
        if es:
            assert es[1] <= es[0] <= es[2], r.id


def test_wheel_expectations_hold(cat):
    assert catalogue.check_wheel(cat, mw.MetalWheel.load()) == []


def test_interactions_form_a_valid_graph(cat):
    g = cat.interactions()
    assert set(g["from"]) | set(g["to"]) <= set(cat.by_id)
    # copper's chain: other scrap markets feed copper-in-steel, which tin amplifies
    pairs = set(zip(g["from"], g["to"], strict=True))
    assert ("cu_to_other_scrap", "cu_in_steel") in pairs
    assert ("cu_in_steel", "sn_in_steel") in pairs


def _write(tmp_path, entries):
    doc = yaml.safe_load((registry.default_params_dir() / "ratchets.yaml").read_text())
    doc["ratchets"] = entries
    p = tmp_path / "ratchets.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda e: e.update(mechanism="corrosion"), "mechanism"),
        (lambda e: e.update(tolerance={"param": "no.such.key"}), "unknown parameter"),
        (lambda e: e.update(source="nobody_2099"), "not in bibliography"),
        (lambda e: e.update(energy_step=None, energy_note=None), "energy_note"),
        (
            lambda e: e.update(interactions=[{"with": "ghost", "kind": "feeds", "note": "x" * 6}]),
            "unknown ratchet",
        ),
        (
            lambda e: e.update(
                wheel_check=[{"route": "fe", "element": "Cu", "expect": ["melted"]}]
            ),
            "unknown fates",
        ),
        (lambda e: e.update(colour="red"), "Extra inputs"),
    ],
)
def test_schema_rejects_bad_entries(tmp_path, reg, mutate, msg):
    raw = catalogue._raw_entries(registry.default_params_dir() / "ratchets.yaml")
    bad = copy.deepcopy(raw)
    mutate(bad[0])
    with pytest.raises(catalogue.CatalogueError, match=msg):
        catalogue.load(_write(tmp_path, bad), reg=reg)


def test_duplicate_ids_rejected(tmp_path, reg):
    raw = catalogue._raw_entries(registry.default_params_dir() / "ratchets.yaml")
    with pytest.raises(catalogue.CatalogueError, match="duplicate"):
        catalogue.load(_write(tmp_path, [raw[0], raw[0]]), reg=reg)
