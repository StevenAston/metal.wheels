"""Task 2.4: energy penalty against hand calculations; p = 1 gives zero penalty."""

import logging

import numpy as np
import pytest

from metalwheels import registry
from metalwheels.copper import demand as cu_demand
from metalwheels.steel import coupled, energy

logging.getLogger("metalwheels").setLevel(logging.ERROR)


@pytest.fixture(scope="module")
def reg():
    return registry.load()


@pytest.fixture(scope="module")
def en(reg):
    return energy.SteelEnergy.from_registry(reg)


def test_primary_mix_hand_calc(en):
    assert en.primary == pytest.approx(0.934 * 20.6 + 0.066 * 20.6)
    assert en.penalty_per_kg == pytest.approx(20.6 - 8.0)


def test_penalty_hand_calc(en):
    # 100 Mt of copper-induced primary iron x 12.6 MJ/kg = 1.26e12 MJ = 1.26 EJ
    assert energy.penalty_ej(100.0, en) == pytest.approx(1.26)
    assert energy.penalty_ej(np.array([0.0, 50.0]), en) == pytest.approx([0.0, 0.63])


def test_steel_energy_hand_calc(en):
    # 1,000 Mt primary at 20.6 + 400 Mt scrap at 8.0 -> 20.6 + 3.2 EJ
    assert energy.steel_energy_ej(1000.0, 400.0, en) == pytest.approx(23.8)


def test_scrap_vs_ore_on_primary_basis(en):
    """IEA 2021: scrap EAF is 60-70 % less energy-intensive (primary basis)."""
    saving = 1 - en.scrap_eaf / en.bf_bof
    assert 0.60 <= saving <= 0.70


def _coupled(reg, end_year=2200, policy=None):
    pj = cu_demand.project(end_year=end_year, reg=reg, policy=policy)
    return coupled.run(pj.result, reg).projection


def test_full_copper_processing_gives_zero_penalty(reg, en):
    """p = 1 in every year: no copper reaches 'other scrap markets', so none reaches steel."""
    sp = _coupled(reg.with_overrides({"cu.eol_processing_rate": 1.0}, label="p1"))
    f = sp.result.flows
    assert f["cu_attached"].max() == 0.0
    assert energy.penalty_ej(f["copper_induced_primary"], en).max() == pytest.approx(0.0, abs=1e-9)


def test_processing_policy_from_2026_cuts_penalty(reg, en):
    """p = 1 from 2026 stops new attachment; copper already in steel keeps cycling."""
    base = _coupled(reg).result.flows
    pol = _coupled(reg, policy={"start_year": 2026, "processing": 1.0}).result.flows
    assert pol.loc[2026:, "cu_attached"].max() == 0.0
    assert pol["copper_induced_primary"].sum() < base["copper_induced_primary"].sum()
