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


params_app = typer.Typer(help="Inspect the parameter registry.", no_args_is_help=True)
app.add_typer(params_app, name="params")


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:g}"


@params_app.command("list")
def params_list(
    domain: str | None = typer.Option(None, "--domain", "-d", help="Only show one domain."),
    params_dir: str | None = typer.Option(
        None, "--params-dir", help="Parameter directory (default: repo params/)."
    ),
    notes: bool = typer.Option(False, "--notes", help="Include the note column."),
) -> None:
    """List every parameter with its value, unit, range and provenance."""
    from rich.console import Console
    from rich.table import Table

    from metalwheels import registry

    try:
        reg = registry.load(params_dir)
    except registry.RegistryError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=1) from exc

    items = [p for p in reg.values() if domain is None or p.domain == domain]
    if domain is not None and not items:
        log.error("no parameters in domain %r (domains: %s)", domain, ", ".join(reg.domains))
        raise typer.Exit(code=1)

    table = Table(title=f"Parameter registry ({len(items)} of {len(reg)})", show_lines=False)
    for col in ("key", "value", "unit", "low", "high", "dist", "status", "source", "locator"):
        table.add_column(col, overflow="fold")
    if notes:
        table.add_column("note", overflow="fold")
    for p in sorted(items, key=lambda q: (q.domain, q.key)):
        row = [
            p.key,
            _fmt(p.value),
            p.unit,
            _fmt(p.low),
            _fmt(p.high),
            p.distribution,
            p.status,
            p.source or "",
            p.locator or "",
        ]
        if notes:
            row.append(p.note or "")
        table.add_row(*row)
    Console(width=200).print(table)


data_app = typer.Typer(help="Build processed data series from data/raw.", no_args_is_help=True)
app.add_typer(data_app, name="data")


@data_app.command("cu-production")
def data_cu_production() -> None:
    """Rebuild data/processed/cu_mine_production.csv (world mine copper, 1900-2025)."""
    from metalwheels.copper import production

    try:
        rows = production.build()
    except production.IngestError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=1) from exc
    path = production.write(rows)
    total = sum(r["mine_production_mt"] for r in rows)
    typer.echo(f"{len(rows)} rows ({rows[0]['year']}-{rows[-1]['year']}), {total:.1f} Mt total")
    typer.echo(f"written to {path}")


copper_app = typer.Typer(help="Copper model views.", no_args_is_help=True)
app.add_typer(copper_app, name="copper")


@copper_app.command("history")
def copper_history(
    years: list[int] = typer.Option(  # noqa: B008
        [1950, 1980, 2000, 2010, 2011, 2025], "--year", "-y", help="Years to show (repeatable)."
    ),
    collection: float | None = typer.Option(None, help="Override EoL collection rate c."),
    processing: float | None = typer.Option(None, help="Override EoL processing rate p."),
) -> None:
    """Historical (1900-2025) supply-driven stock-flow run: key flows in Mt Cu."""
    from rich.console import Console
    from rich.table import Table

    from metalwheels.copper import stockflow as sf

    overrides = {
        k: v
        for k, v in {"collection": collection, "processing": processing}.items()
        if v is not None
    }
    res = sf.run_historical(**overrides)
    f = res.flows
    missing = [y for y in years if y not in f.index]
    if missing:
        log.error("years outside 1900-2025: %s", missing)
        raise typer.Exit(code=1)
    table = Table(title="Copper stock-flow, historical (Mt Cu)")
    table.add_column("flow")
    for y in years:
        table.add_column(str(y), justify="right")
    for name in sf.FLOW_COLUMNS:
        table.add_row(name, *(f"{f.loc[y, name]:.2f}" for y in years))
    Console(width=200).print(table)
    typer.echo(f"max relative mass-balance residual: {res.balance_residual().abs().max():.1e}")
