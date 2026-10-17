"""Phases 2-3 report regenerates from one command and documents the cascade definitions."""

from metalwheels import report_phase23


def test_report_builds_with_all_checks_passing(tmp_path):
    path = report_phase23.build(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "**Result: 15 of 15 checks pass.**" in text
    assert "Maintenance energy** E(t)" in text
    assert "rho(A) is the spectral radius" in text
    for fig in ("p2_copper_balance.png", "p3_maintenance_energy.png", "p3_rho_multiplier.png"):
        assert (tmp_path / "figures" / fig).is_file()
