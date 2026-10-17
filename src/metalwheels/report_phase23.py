"""Phases 2-3 report (``metalwheels report phase23``): steel coupling and cascade.

Regenerates ``reports/phase2_3.md`` and its figures from the baseline run plus the two allocation
and loss variants. It records every Phase 2-3 done condition, the validation checks against
Daehn et al. (2017) and Pauliuk et al. (2013), the definitions of rho(A) and maintenance energy,
and the coefficient provenance gaps.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np

from metalwheels import __version__, export, registry, scenarios
from metalwheels.cascade import leontief
from metalwheels.cascade import matrix as cascade
from metalwheels.report import (
    C_FOURTH,
    C_HARMSEN,
    C_KESLER,
    C_THIRD,
    INK2,
    REPORTS,
    SURFACE,
    Check,
    _fmt,
    _md_table,
    _style,
)
from metalwheels.steel import dilution, link

VARIANTS = {
    "pinch allocation": {"st.allocation_pinch": 1.0},
    "losses carry copper (pi = 1)": {"st.cu_to_process_losses": 1.0},
}


def _year(v: float) -> str:
    return "none" if v != v else f"{v:.0f}"


def _cell(c: Check) -> str:
    if c.unit == "year":
        return _year(c.value)
    return _fmt(c.value) + (f" {c.unit}" if c.unit else "")


def _run(reg, overrides=None, name="baseline"):
    sc = scenarios.load_scenario("baseline")
    if overrides:
        sc = dataclasses.replace(sc, name=name, overrides={**sc.overrides, **overrides})
    return scenarios.run_scenario(sc, reg)


def _daehn_fig3_crossing() -> float:
    """Task 2.3 done condition with Daehn-equivalent inputs (Fig. 3 snapshots, see tests)."""
    snaps = {
        2040: ([220, 180, 190, 210], [0.30, 0.30, 0.20, 0.175],
               [220, 110, 160, 560, 210, 270], [0.4, 0.25, 0.175, 0.125, 0.08, 0.06]),
        2050: ([300, 180, 230, 275], [0.30, 0.30, 0.20, 0.175],
               [200, 110, 150, 670, 140, 50], [0.4, 0.25, 0.175, 0.125, 0.08, 0.06]),
    }  # fmt: skip
    gap = {}
    for y, (s, c, m, tau) in snaps.items():
        sup, tol = dilution.copper_balance(s, np.array(c) / 100, m, np.array(tau) / 100)
        gap[y] = sup - tol
    return 2040 + 10 * (-gap[2040]) / (gap[2050] - gap[2040])


def checks(base, reg) -> list[Check]:
    ts, s = base.timeseries, base.summary
    out = []
    for sec in link.SECTORS:
        p = reg[f"st.target.scrap_cu_2008.{sec}"]
        v = s[f"st_scrap_cu_2008_{sec}"][0]
        out.append(Check("2.1", f"2008 scrap Cu, {sec.replace('_', ' ')}", 100 * v,
                         f"{100 * p.low:g}-{100 * p.high:g} wt%",
                         p.low - 1e-9 <= v <= p.high + 1e-9, "wt%",
                         "Daehn 2017 SI Table S21 range"))  # fmt: skip
    resid = (ts["st_cu_to_other_sinks"] + ts["st_cu_attached"] - ts["other_scrap_markets"]).abs()
    rel = float((resid / ts["other_scrap_markets"].where(ts["other_scrap_markets"] > 0)).max())
    out.append(Check("2.2", "copper -> steel conservation (max rel. residual)", rel, "< 1e-6",
                     rel < 1e-6, "", "every year 1900-3025"))  # fmt: skip
    x = dilution.dilution_share(0.004, 0.001, reg.value("st.primary_cu"))
    out.append(Check("2.3", "dilution share 0.4 -> 0.1 wt% (c_p 0.02)", x, "0.79",
                     round(x, 2) == 0.79))  # fmt: skip
    cross = _daehn_fig3_crossing()
    out.append(Check("2.3", "crossover with Daehn-equivalent inputs", cross, "2040-2060",
                     2040 <= cross <= 2060, "year", "Daehn Fig. 3 snapshots"))  # fmt: skip
    tol08 = float(ts.loc[2008, "st_cu_tolerable_aggregate"])
    out.append(Check("2.3", "copper tolerable 2008", tol08, "1.8 Mt +/- 15 %",
                     abs(tol08 / 1.8 - 1) <= 0.15, "Mt", "Daehn 2017 p. 6602"))  # fmt: skip
    pc = ts.loc[2008, "st_stock_per_capita"]
    out.append(Check("2.1", "steel stock per capita 2008", pc / 1e3, "3.7 t +/- 15 %",
                     abs(pc / 3700 - 1) <= 0.15, "t", "Pauliuk 2013a p. 3451"))  # fmt: skip
    pc = ts.loc[2100, "st_stock_per_capita"]
    out.append(Check("2.1", "steel stock per capita 2100", pc / 1e3, "ca. 10 t +/- 15 %",
                     abs(pc / 10000 - 1) <= 0.15, "t", "Pauliuk 2013a p. 3451"))  # fmt: skip
    scrap = float(ts.loc[2008, "st_scrap_recovered"])
    out.append(Check("2.1", "end-of-life scrap 2008", scrap, "298 Mt +/- 20 %",
                     abs(scrap / 298 - 1) <= 0.2, "Mt", "Daehn 2017 SI Table S22"))  # fmt: skip
    k = ts.index.get_loc(2025)
    for br in scenarios.BRACKETS:
        e = ts[f"cas_{br}_maintenance_energy_ej"]
        out.append(Check("3.3", f"maintenance energy 2025 reproduces supply ({br})",
                         float(e.iloc[k]), "600 EJ", abs(e.iloc[k] / 600 - 1) < 1e-6, "EJ/yr",
                         "y = (I - A_2025) x_2025"))  # fmt: skip
        r = float(ts[f"cas_{br}_rho"].max())
        out.append(Check("3.2", f"rho(A) below warning level ({br})", r, "< 0.9", r < 0.9))
    return out


def figures(base, variants, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    ts = base.timeseries
    names = []

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    sl = ts.loc[1950:2300]
    ax.plot(sl.index, sl["st_cu_supply_aggregate"], color=C_KESLER, label="copper in scrap")
    ax.plot(sl.index, sl["st_cu_tolerable_aggregate"], color=C_HARMSEN,
            label="copper tolerable by products")  # fmt: skip
    for (label, run), col in zip(variants.items(), (C_THIRD, C_FOURTH), strict=True):
        v = run.timeseries.loc[1950:2300]
        ax.plot(v.index, v["st_cu_supply_aggregate"], color=col, lw=1, ls="--",
                label=f"copper in scrap, {label}")  # fmt: skip
    _style(ax, "Copper in steel scrap vs what products tolerate (baseline)", "Mt Cu/yr")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(outdir / "p2_copper_balance.png", dpi=150)
    plt.close(fig)
    names.append("p2_copper_balance.png")

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    for br, col in (("harmsen", C_HARMSEN), ("kesler", C_KESLER)):
        e = ts[f"cas_{br}_maintenance_energy_ej"]
        ax.plot(e.index[e.index >= 2025], e[e.index >= 2025], color=col,
                label=f"{br.capitalize()} bracket")  # fmt: skip
    _style(ax, "Maintenance energy: energy to deliver 2025 final demand", "EJ/yr")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(outdir / "p3_maintenance_energy.png", dpi=150)
    plt.close(fig)
    names.append("p3_maintenance_energy.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), facecolor=SURFACE)
    for br, col in (("harmsen", C_HARMSEN), ("kesler", C_KESLER)):
        f = ts.loc[2025:]
        axes[0].plot(f.index, f[f"cas_{br}_rho"], color=col, label=br.capitalize())
        axes[1].plot(f.index, f[f"cas_{br}_copper_energy_multiplier_gj_per_t"], color=col,
                     label=br.capitalize())  # fmt: skip
    _style(axes[0], "Spectral radius rho(A)", "1")
    _style(axes[1], "Total energy per t copper (cascade)", "GJ/t")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(outdir / "p3_rho_multiplier.png", dpi=150)
    plt.close(fig)
    names.append("p3_rho_multiplier.png")
    return names


def build(out_dir: Path = REPORTS) -> Path:
    logging.getLogger("metalwheels").setLevel(logging.ERROR)
    reg = registry.load()
    base = _run(reg)
    variants = {k: _run(reg, v, name=f"variant_{i}") for i, (k, v) in enumerate(VARIANTS.items())}
    chk = checks(base, reg)
    figs = figures(base, variants, out_dir / "figures")
    s = base.summary
    n_pass = sum(c.passed for c in chk)
    phi = [base.calibration[f"steel_cu_link_{sec}"] for sec in link.SECTORS]

    md = ["# Phases 2-3 report — steel coupling and cascade\n",
          f"*Generated by `metalwheels report phase23` (metalwheels {__version__}, commit "
          f"`{export.git_commit()}`). Do not edit by hand; regenerate instead.*\n",
          f"**Result: {n_pass} of {len(chk)} checks pass.**\n",
          "## Done conditions and validation\n",
          _md_table([[c.task, c.name, _cell(c),
                      c.target, "PASS" if c.passed else "**FAIL**", c.note] for c in chk],
                    ["Task", "Check", "Model", "Target", "Result", "Source / note"])]  # fmt: skip

    md.append("\n## Steel coupling (Phase 2)\n")
    md.append(
        "Copper reaches steel scrap through the copper module's *other scrap markets* flow. "
        "Shares of it per steel sector are calibrated each run so 2008 scrap copper matches Daehn "
        "et al. (2017): " + ", ".join(f"{sec.replace('_', ' ')} {p:.2f}" for sec, p in
                                      zip(link.SECTORS, phi, strict=True)) + ". "
        "Copper put into steel stays with each cohort and comes back at end of life."
    )  # fmt: skip
    rows = []
    for label, run in {"baseline (mixed allocation, pi = 0)": base, **variants}.items():
        rs = run.summary
        rows.append(
            [
                label,
                _year(rs["st_crossover_year_lp"][0]),
                _year(rs["st_crossover_year_aggregate"][0]),
                _fmt(rs["st_cu_penalty_2200"][0]),
                _fmt(rs["st_cum_cu_penalty_ej"][0]),
            ]
        )
    md.append("\n" + _md_table(rows, ["Variant", "Crossover (LP)", "Crossover (aggregate)",
                                      "Cu-induced steel energy 2200 (EJ/yr)",
                                      "Cumulative to horizon (EJ)"]))  # fmt: skip
    md.append(f"\n![copper balance](figures/{figs[0]})\n")

    md.append("## Cascade (Phase 3)\n")
    md.append(
        "**Definitions.** A(t) is a 7 x 7 physical input-output matrix (copper, steel, "
        "aluminium, cement, ammonia, water, energy); a_ij is the input from sector i per unit of "
        "sector j's output. The boundary covers consumables for every sector plus the materials "
        "to build and maintain energy supply. rho(A) is the spectral radius: the Leontief series "
        "converges only for rho < 1, and rho -> 1 means each unit of final demand needs "
        "ever more gross output. Final demand is fixed at 2025, y = (I - A_2025) x_2025, using "
        "observed world outputs x_2025. **Maintenance energy** E(t) = [(I - A(t))^-1 y]_energy is "
        "the primary energy needed each year to keep delivering 2025's final demand, given that "
        "year's grades, depths and scrap quality. Changes in E come only from A(t)."
    )
    rows = []
    for label, key, unit in (
        ("rho 2025", "rho_2025", ""),
        ("rho 2100", "rho_2100", ""),
        ("rho max", "rho_max", ""),
        ("maintenance energy 2100", "maintenance_energy_2100", "EJ/yr"),
        ("maintenance energy 2500", "maintenance_energy_2500", "EJ/yr"),
    ):
        rows.append([label] + [_fmt(s[f"cas_{b}_{key}"][0]) + (f" {unit}" if unit else "")
                               for b in scenarios.BRACKETS])  # fmt: skip
    md.append("\n" + _md_table(rows, ["Metric", "Harmsen", "Kesler"]))
    md.append(f"\n![maintenance energy](figures/{figs[1]})\n\n![rho](figures/{figs[2]})\n")

    ts = base.timeseries
    md.append("**Reading the results (Harmsen bracket):**\n")
    e = ts["cas_harmsen_maintenance_energy_ej"].loc[2025:]
    ymin = int(e.idxmin())
    cu = ts["cas_harmsen_copper_direct_energy_ej"]
    stl = ts["cas_harmsen_steel_direct_energy_ej"]
    until = int(s["cas_harmsen_valid_until"][0])
    md += [
        f"- Maintenance energy falls from {e.loc[2025]:.0f} to {e.min():.1f} EJ/yr ({ymin}), "
        f"then recovers to {e.loc[min(2300, until)]:.1f} EJ/yr by {min(2300, until)}: the "
        "cascade holds 2025 final "
        "demand fixed, so changes are small next to total supply.",
        f"- Steel's direct energy falls from {stl.loc[2025]:.1f} to {stl.loc[2100]:.1f} EJ/yr "
        "by 2100 as the scrap share of the iron charge rises, then climbs back to "
        f"{stl.loc[2200]:.1f} EJ/yr by 2200 as copper contamination forces primary iron back "
        "in after the crossover.",
        f"- Copper's direct energy rises from {cu.loc[2025]:.2f} to {cu.loc[min(2300, until)]:.2f} "
        f"EJ/yr by {min(2300, until)} as grade falls and depth rises; values are blank after the "
        f"first supply shortfall ({until + 1 if until < int(ts.index[-1]) else 'none'}), when "
        "2025 demand can no longer be met from mines.",
        "- rho(A) stays near 0.12: the copper-energy loop (energy needs copper for grids, copper "
        "needs energy) is real but small next to the energy sector's own use (7.7 %), which "
        "dominates rho. rho would approach 1 only if energy supply itself became far more "
        "self-consuming (falling EROI), which this model does not yet represent.",
        "- After 2100 world population is held flat (PLAN decision 4); the switch from slowly "
        "falling to flat population makes steel demand step up in 2101, visible as a kink.",
        "",
    ]
    res = cascade.run(base.timeseries, reg, "harmsen")
    df = cascade.sensitivity_table(res, 2100).head(5)
    md.append(
        "**Top-5 drivers of maintenance energy, Harmsen 2100** (`metalwheels cascade "
        "sensitivity` for the full table):\n"
    )
    md.append(_md_table([[str(r.rank), f"{r.input} -> {r.output}", f"{r.energy_elasticity:.4f}",
                          f"{r.rho_elasticity:.4f}"] for r in df.itertuples()],
                        ["Rank", "Cell", "Elasticity of E", "Elasticity of rho"]))  # fmt: skip
    k = int(np.flatnonzero(res.years == 2100)[0])
    md.append(
        f"\nCheck: rho elasticities sum to "
        f"{leontief.rho_elasticities(res.a[k]).sum():.6f} (Euler, expected 1)."
    )

    gaps = cascade.provenance_gaps(reg)
    md.append(
        f"\n**Coefficient provenance.** {len(cascade.CELL_PARAMS) - len(gaps)} of "
        f"{len(cascade.CELL_PARAMS)} non-zero cells rest only on sourced parameters with "
        f"ranges; {len(cascade.ZERO_NOTES)} zero cells each carry a justification "
        f"(`cascade.matrix.ZERO_NOTES`). Cells with assumptions:\n"
    )
    md += [f"- {i} -> {o}: {'; '.join(items)}" for (i, o), items in gaps.items()]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase2_3.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
