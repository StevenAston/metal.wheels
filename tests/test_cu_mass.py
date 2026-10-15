"""Task 1.7 done condition: ore = Cu / grade exactly; tailings volume uses the configured dry
density (default 1.40 t/m^3); 23 Mt at 0.62 % returns 3.7 +/- 0.05 Gt ore."""

import numpy as np
import pytest

from metalwheels import registry
from metalwheels.copper import geology as geo
from metalwheels.copper import mass, production


def test_ore_is_cu_over_grade_exactly():
    cu = np.array([1.0, 23.0, 55.5])
    g = np.array([0.01, 0.0062, 0.0031])
    np.testing.assert_array_equal(mass.ore_mass(cu, g), cu / g)


def test_23_mt_at_062_pct():
    assert float(mass.ore_mass(23.0, 0.0062)) / 1e3 == pytest.approx(3.7, abs=0.05)


def test_volume_uses_configured_density():
    reg = registry.load()
    prm = mass.MassParams.from_registry(reg)
    assert prm.tailings_density == 1.40
    # 1,400 Mt at 1.40 t/m^3 = 1.0 km^3
    assert float(mass.volume_km3(1400.0, prm.tailings_density)) == pytest.approx(1.0)
    alt = mass.MassParams.from_registry(reg, tailings_density=1.94)
    assert float(mass.volume_km3(1400.0, alt.tailings_density)) == pytest.approx(1400 / 1940)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        mass.ore_mass(1.0, 0.0)
    with pytest.raises(ValueError):
        mass.volume_km3(1.0, 0.0)
    with pytest.raises(ValueError, match="unknown overrides"):
        mass.MassParams.from_registry(registry.load(), strip=2)


def test_historical_mass_flows_consistent():
    reg = registry.load()
    s = production.load()
    years = np.array(sorted(s))
    cu = np.array([s[y] for y in years]) / (1 - reg.value("cu.tailings_share_of_ore_cu"))
    _, r = geo.extraction_for(reg, "harmsen", years, cu)
    df = mass.mass_flows(years, cu, r.grade, mass.MassParams.from_registry(reg))
    # geology's own ore tonnage equals Cu / grade
    np.testing.assert_allclose(df["ore"], r.ore, rtol=1e-9)
    # tailings are ore minus concentrate and are most of the ore
    assert ((df["tailings"] / df["ore"]) > 0.90).all()  # ~93 % at 1900s grades, ~98 % today
    # 2018 rock moved reproduces Nassar's 9.4 Bt by construction of the strip ratio (+/- 2 %)
    assert df.loc[2018, "rock_moved"] == pytest.approx(9400, rel=0.02)
