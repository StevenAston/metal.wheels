"""Task 1.6 done condition: Norgate 72/131/249 MJ/kg at 1.0/0.5/0.25 % within 2 %; Koppelaar
surface 52/95/255 MJ/kg at 0.5/0.4/0.3 % (300 m) within 3 %; a WARNING is logged whenever an
option is evaluated outside its source data range."""

import logging

import numpy as np
import pytest

from metalwheels import registry
from metalwheels.copper import energy


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.mark.parametrize(
    ("option", "targets"),
    [("norgate_coarse", (72, 131, 249)), ("norgate_fine", (129, 244, 474))],
)
def test_norgate_points(reg, option, targets):
    m = energy.EnergyModel.from_registry(reg, option)
    got = m.intensity(np.array([0.01, 0.005, 0.0025]), 0.0)
    np.testing.assert_allclose(got, targets, rtol=0.02)


def test_koppelaar_surface_points(reg):
    m = energy.EnergyModel.from_registry(reg, "koppelaar_surface")
    got = m.intensity(np.array([0.005, 0.004, 0.003]), 300.0)
    np.testing.assert_allclose(got, [52, 95, 255], rtol=0.03)


def test_norgate_depth_term_only_beyond_reference(reg):
    m = energy.EnergyModel.from_registry(reg, "norgate_coarse")
    base = float(m.intensity(0.005, 0.0))
    assert float(m.intensity(0.005, 497.0)) == pytest.approx(base)
    assert float(m.intensity(0.005, 1497.0)) == pytest.approx(base + 29.0)  # 2.9 MJ/kg per 100 m


@pytest.mark.parametrize(
    ("option", "grade", "depth", "what"),
    [
        ("norgate_coarse", 0.0005, 500.0, "grade"),
        ("norgate_fine", 0.05, 500.0, "grade"),
        ("koppelaar_surface", 0.001, 300.0, "grade"),
        ("koppelaar_underground", 0.01, 2500.0, "depth"),
    ],
)
def test_warning_outside_source_range(reg, caplog, option, grade, depth, what):
    m = energy.EnergyModel.from_registry(reg, option)
    with caplog.at_level(logging.WARNING, logger="metalwheels.copper.energy"):
        m.intensity(grade, depth)
    assert any(f"outside its source {what} range" in r.getMessage() for r in caplog.records)


def test_no_warning_inside_range(reg, caplog):
    m = energy.EnergyModel.from_registry(reg, "koppelaar_surface")
    with caplog.at_level(logging.WARNING, logger="metalwheels.copper.energy"):
        m.intensity(np.array([0.005, 0.01]), np.array([300.0, 600.0]))
    assert not caplog.records


def test_unknown_option_rejected(reg):
    with pytest.raises(ValueError, match="unknown energy option"):
        energy.EnergyModel.from_registry(reg, "magic")


def test_annual_energy_units():
    # 20 Mt Cu at 100 MJ/kg = 2,000 PJ = 2 EJ
    assert energy.annual_energy_ej(20.0, 100.0) == pytest.approx(2.0)
