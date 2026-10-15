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


@copper_app.command("project")
def copper_project(
    end_year: int = typer.Option(3025, "--end", help="Last year of the projection."),
    years: list[int] = typer.Option(  # noqa: B008
        [2025, 2026, 2050, 2100, 2200, 2500, 3025], "--year", "-y", help="Years to show."
    ),
    saturation: float | None = typer.Option(None, "--saturation", help="k_sat, kg/person."),
    collection: float | None = typer.Option(None, help="Override EoL collection rate c."),
    processing: float | None = typer.Option(None, help="Override EoL processing rate p."),
) -> None:
    """Historical run then stock-driven demand to --end (Mt Cu unless noted)."""
    from rich.console import Console
    from rich.table import Table

    from metalwheels.copper import demand

    overrides = {
        k: v
        for k, v in {"collection": collection, "processing": processing}.items()
        if v is not None
    }
    pj = demand.project(end_year=end_year, k_sat=saturation, **overrides)
    f = pj.result.flows
    years = [y for y in years if y in f.index]
    table = Table(
        title=f"Copper projection (k_sat {pj.k_sat:.0f} kg/person, g {pj.growth_rate:.4f}/yr)"
    )
    table.add_column("quantity")
    for y in years:
        table.add_column(str(y), justify="right")
    pc = dict(zip(f.index, pj.per_capita_stock, strict=True))
    table.add_row("population (bn)", *(f"{pj.population[y - f.index[0]] / 1e9:.2f}" for y in years))
    table.add_row("stock per person (kg)", *(f"{pc[y]:.1f}" for y in years))
    for name in (
        "stock",
        "refined",
        "mine",
        "secondary_supply",
        "eol",
        "old_scrap",
        "not_recycled",
    ):
        table.add_row(name, *(f"{f.loc[y, name]:.2f}" for y in years))
    Console(width=200).print(table)
    fut = f.loc[f.index > 2025, "mine"].sum()
    typer.echo(f"cumulative mine output 2026-{end_year}: {fut / 1e3:.1f} Bt Cu")


@app.command("scenarios")
def scenarios_list() -> None:
    """List available scenarios."""
    from metalwheels import scenarios

    for name in scenarios.list_scenarios():
        sc = scenarios.load_scenario(name)
        typer.echo(f"{name:20s} {sc.description}")


@app.command("run")
def run_cmd(
    scenario: str = typer.Argument(..., help="Scenario name (scenarios/<name>.yaml) or path."),
    out: str | None = typer.Option(None, "--out", help="Output directory (default runs/<name>)."),
) -> None:
    """Run one scenario 1900-end year; write Parquet time series, summary CSV and metadata."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    from metalwheels import export, registry, scenarios

    try:
        sc = scenarios.load_scenario(scenario)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        raise typer.Exit(code=1) from exc
    reg = registry.load()
    console = Console(width=160)
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console
    ) as prog:
        task = prog.add_task(f"{sc.name}: starting", total=None)
        result = scenarios.run_scenario(
            sc, reg, progress=lambda m: prog.update(task, description=f"{sc.name}: {m}")
        )
        prog.update(task, description=f"{sc.name}: writing outputs")
        path = export.write(result, reg.with_overrides(sc.overrides, label=sc.name), out)
    s = result.summary
    table = Table(title=f"{sc.name} - key results")
    table.add_column("metric")
    table.add_column("Harmsen", justify="right")
    table.add_column("Kesler", justify="right")
    for label, key, fmt in (
        ("endowment (Mt Cu)", "endowment", "{:,.0f}"),
        ("exhausted (year)", "exhausted_year", "{:.0f}"),
        ("grade 2100 (wt%)", "grade_2100", "{:.3f}"),
        ("depth 2100 (m)", "depth_2100", "{:,.0f}"),
        ("energy 2100 (MJ/kg)", "energy_intensity_2100", "{:.0f}"),
        ("primary energy 2100 (EJ/yr)", "primary_energy_2100", "{:.2f}"),
        ("cumulative primary energy (EJ)", "cum_primary_energy", "{:,.0f}"),
    ):
        vals = [s.get(f"{b}_{key}", (float("nan"), ""))[0] for b in scenarios.BRACKETS]
        table.add_row(label, *("-" if v != v else fmt.format(v) for v in vals))
    console.print(table)
    console.print(
        f"cumulative mine output 2026-{sc.end_year}: "
        f"{s['cum_mine_output_future'][0] / 1e3:.1f} Bt Cu; "
        f"refined 2026 {s['refined_2026'][0]:.1f} Mt"
    )
    console.print(f"runtime {result.runtime_s:.1f} s; outputs in {path}")


@app.command("mc")
def mc_cmd(
    scenario: str = typer.Argument("baseline", help="Scenario name."),
    draws: int = typer.Option(1000, "--draws", "-n", help="Number of Monte Carlo draws."),
    seed: int = typer.Option(20261015, "--seed", help="Random seed (same seed, same result)."),
    workers: int | None = typer.Option(
        None, "--workers", "-w", help="Worker processes (default max(1, n_cpu - 2))."
    ),
    out: str | None = typer.Option(None, "--out", help="Output dir (default runs/<name>/mc)."),
) -> None:
    """Monte Carlo over registry parameter ranges; writes 5/50/95 % bands."""
    from pathlib import Path

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

    from metalwheels import export, montecarlo, scenarios

    sc = scenarios.load_scenario(scenario)
    n_workers = montecarlo.resolve_workers(workers)
    typer.echo(f"{sc.name}: {draws} draws on {n_workers} worker(s), seed {seed}")
    with Progress(
        "[progress.description]{task.description}", BarColumn(), MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as prog:  # fmt: skip
        task = prog.add_task("draws", total=draws)
        res = montecarlo.run(sc, draws, seed, n_workers, progress=lambda: prog.advance(task))
    d = Path(out) if out else export.RUNS_DIR / sc.name / "mc"
    montecarlo.write(res, d)
    q = res.summary_quantiles
    for key in (
        "cum_mine_output_future",
        "harmsen_exhausted_year",
        "kesler_grade_2100",
        "harmsen_primary_energy_2100",
        "kesler_primary_energy_2100",
    ):
        if key in q.index:
            r = q.loc[key]
            typer.echo(
                f"{key:32s} p05 {r['p05']:10.4g}  p50 {r['p50']:10.4g}  p95 {r['p95']:10.4g}"
            )
    typer.echo(f"written to {d}")


report_app = typer.Typer(help="Generate reports.", no_args_is_help=True)
app.add_typer(report_app, name="report")


@report_app.command("phase1")
def report_phase1(
    mc_dir: str | None = typer.Option(
        None, help="Monte Carlo output dir (default runs/baseline/mc)."
    ),
) -> None:
    """Regenerate reports/phase1_validation.md and its figures."""
    import logging
    from pathlib import Path

    from metalwheels import report

    logging.getLogger("metalwheels.copper.energy").setLevel(logging.ERROR)
    path = report.build(mc_dir=Path(mc_dir) if mc_dir else None)
    text = path.read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("**Result:"))
    typer.echo(line.strip("*"))
    typer.echo(f"written to {path}")
