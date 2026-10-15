"""Task 0.1 done condition: package imports, CLI reports its version."""

from typer.testing import CliRunner

import metalwheels
from metalwheels.cli import app


def test_version_attribute_is_set():
    assert isinstance(metalwheels.__version__, str)
    assert metalwheels.__version__ != ""


def test_cli_version_prints_package_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"metalwheels {metalwheels.__version__}"


def test_cli_info_runs():
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Monte Carlo workers" in result.stdout
