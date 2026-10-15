"""Task 1.10 done condition: report regenerates from one CLI command and contains a pass/fail
table for every calibration target in Tasks 1.1-1.7 (deviations explained)."""

import pytest
from typer.testing import CliRunner

from metalwheels import report, scenarios
from metalwheels.cli import app


@pytest.fixture(scope="module")
def checks():
    return report.run_checks(scenarios.run_scenario(scenarios.load_scenario("baseline")))


def test_every_task_has_checks(checks):
    tasks = {c.task for c in checks}
    assert {"1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"} <= tasks


def test_all_plan_targets_pass(checks):
    """Checks for Tasks 1.1-1.7 are PLAN done conditions and must all pass."""
    failing = [c.name for c in checks if c.task != "x" and not c.passed]
    assert failing == []


def test_failing_cross_checks_are_explained(checks):
    for c in checks:
        if not c.passed:
            assert c.name in report.DEVIATIONS, c.name


def test_report_regenerates_from_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "REPORTS", tmp_path)
    original = report.build
    monkeypatch.setattr(report, "build", lambda mc_dir=None: original(tmp_path, mc_dir))
    result = CliRunner().invoke(app, ["report", "phase1", "--mc-dir", str(tmp_path / "none")])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "phase1_validation.md").read_text()
    assert "## Calibration and validation checks" in text
    assert "| Task | Check | Model | Target | Result |" in text
    assert (tmp_path / "figures" / "mine_output.png").is_file()
