"""Task 4.2 done condition: >= 30 elements; every cell sourced or explicitly unknown; no blanks."""

import pytest
import yaml

from metalwheels import references, registry
from metalwheels.ratchets import metal_wheel as mw

PLAN_ROUTES = {"fe", "al", "cu", "pbzn", "ni", "sn"}


@pytest.fixture(scope="module")
def wheel():
    return mw.MetalWheel.load()


def test_covers_thirty_elements_and_the_plan_routes(wheel):
    assert len(wheel.elements) >= 30
    assert set(wheel.routes) == PLAN_ROUTES


def test_no_blank_cells(wheel):
    m = wheel.matrix()
    assert m.shape == (len(wheel.elements), len(PLAN_ROUTES))
    assert not m.isna().any().any()
    assert set(m.to_numpy().ravel()) <= set(mw.FATES)
    assert all(isinstance(v, str) and v for v in m.to_numpy().ravel())


def test_every_cell_sourced_or_explicitly_unknown(wheel):
    for c in wheel.cells():
        if c.fate == "unknown":
            assert c.rule in ("absent",) or c.rule.startswith("wheel-"), c
            assert c.note, c
        else:
            assert c.sourced and wheel.source in c.locator, c
    assert wheel.source in references.load()


def test_carriers_are_recovered(wheel):
    for r, spec in wheel.routes.items():
        for el in spec["carriers"]:
            assert wheel.cell(r, el).fate == "recovered"


@pytest.mark.parametrize(
    "route, element, fate",
    [
        ("fe", "Cu", "dissolved_harmful"),  # copper tramp in steel
        ("fe", "Sn", "dissolved_harmful"),  # tin tramp in steel
        ("fe", "Ni", "dissolved_benign"),  # alloying, stays in the steel
        ("fe", "Zn", "volatilised"),  # EAF dust
        ("fe", "Ti", "lost_to_slag"),
        ("fe", "REE", "lost_to_slag"),
        ("al", "Mg", "lost_to_slag"),  # oxidised to dross
        ("al", "Hg", "volatilised"),
        ("cu", "Au", "recovered"),  # precious metals follow copper
        ("cu", "Pt", "recovered"),
        ("cu", "Fe", "lost_to_slag"),
        ("pbzn", "Ag", "recovered"),  # radar only: recoverable in lead bullion
        ("ni", "Li", "unknown"),
    ],
)
def test_known_metallurgy(wheel, route, element, fate):
    assert wheel.cell(route, element).fate == fate


def test_radar_beats_wheel_ring_for_phase(wheel):
    """Fe route: wheel puts Cr in the metal ring (orange); radar puts it in the slag (alloying,
    oxidised in BOF/EAF). The radar decides the primary fate, the wheel ring is secondary."""
    c = wheel.cell("fe", "Cr")
    assert c.fate == "lost_to_slag" and c.rule == "radar-phase"


def test_volatile_rule_only_on_routes_without_radar(wheel):
    c = wheel.cell("ni", "Zn")
    assert (c.fate, c.rule) == ("volatilised", "wheel-compound")


def test_markdown_render_has_every_row(wheel):
    md = wheel.to_markdown()
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and not ln.startswith("| Element")]
    assert len(rows) == len(wheel.elements)
    assert "Legend" in md


def test_transcription_errors_are_caught(tmp_path):
    doc = yaml.safe_load((registry.default_params_dir() / "metal_wheel.yaml").read_text())
    doc["wheel"]["fe"]["metal"]["Cu"] = "X"
    p = tmp_path / "mw.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(mw.MetalWheelError, match="bad colour"):
        mw.MetalWheel.load(p)
    doc = yaml.safe_load((registry.default_params_dir() / "metal_wheel.yaml").read_text())
    doc["radar"]["fe"]["gas"]["Cu"] = "-"  # Cu already in the metal phase
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(mw.MetalWheelError, match="both"):
        mw.MetalWheel.load(p)
