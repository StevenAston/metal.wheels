"""Command-line interface for metal.wheels."""

from __future__ import annotations

import typer

from metalwheels import __version__
from metalwheels.logging_setup import configure_logging, get_logger

app = typer.Typer(
    help="metal.wheels: long-horizon metal stock-flow and maintenance-energy model.",
    no_args_is_help=True,
)
log = get_logger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"metalwheels {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the package version and exit.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG, INFO, WARNING or ERROR."),
) -> None:
    """Global options shared by all commands."""
    configure_logging(log_level)
    log.debug("CLI started (version %s)", __version__)


@app.command()
def info() -> None:
    """Show build and environment information."""
    import os
    import platform

    typer.echo(f"metalwheels {__version__}")
    typer.echo(f"Python {platform.python_version()} on {platform.system()}")
    typer.echo(f"Default Monte Carlo workers: {max(1, (os.cpu_count() or 1) - 2)}")
