"""Task 1.10: Phase 1 validation report (``metalwheels report phase1``).

Regenerates ``reports/phase1_validation.md`` and ``reports/figures/*.png`` from one command:
re-runs the baseline and the recycling scenarios, evaluates every calibration target from
Tasks 1.1-1.7 plus independent cross-checks, and folds in Monte Carlo bands from
``runs/baseline/mc`` when present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metalwheels import __version__, export, registry, scenarios
from metalwheels.copper import energy, fates, geology, mass, production
from metalwheels.copper import stockflow as sf

REPORTS = Path(__file__).resolve().parents[2] / "reports"

# Reference palette (dataviz skill, light mode): categorical slots 1-3, surface and inks.
C_HARMSEN, C_KESLER, C_THIRD, C_FOURTH = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


ICSG_ANNUAL = "refined copper vs ICSG 1960-2019 (max annual abs dev)"

#: Explanations for checks that fail; each must be accepted by Steve before Phase 2.
DEVIATIONS = {
    ICSG_ANNUAL: (
        "Independent cross-check (not a PLAN target). The worst single year is driven mostly by "
        "USGS and ICSG disagreeing on world *mine* production that year. The remaining pattern - "
        "model low in the 1960s (about -9 %), high in the 2000s (about +6 %) - is what constant "
        "collection/processing rates over 1900-2025 produce if real recycling was higher than "
        "0.67 x 0.69 in the 1960s and lower during the 2000s China boom. Decadal means stay within "
        "10 %. **Status: awaiting Steve's acceptance.**"
    ),
}


@dataclass
class Check:
    task: str
    name: str
    value: float
    target: str
    passed: bool
    unit: str = ""
    note: str = ""


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.1:
        return f"{v:.3f}"
    return f"{v:.2e}"


def _within(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def run_checks(base: scenarios.RunOutput) -> list[Check]:
    reg = registry.load()
    ts = base.timeseries
    out: list[Check] = []

    # --- 1.1 production ----------------------------------------------------------------------
    s = production.load()
    yrs = sorted(s)
    out.append(Check("1.1", "annual rows 1900-2025, no gaps", len(yrs), "126",
                     yrs == list(range(1900, 2026))))  # fmt: skip
    cum = sum(v for y, v in s.items() if y <= 2015)
    out.append(Check("1.1", "cumulative mine output 1900-2015", cum, "600 +/- 10 %",
                     _within(cum, 540, 660), "Mt", "USGS past production ~600 Mt (MCS 2026)"))  # fmt: skip
    out.append(Check("1.1", "mine output 2025", s[2025], "23 +/- 0.5", abs(s[2025] - 23) <= 0.5,
                     "Mt", "MCS 2026 estimate"))  # fmt: skip

    # --- 1.2 stock-flow ----------------------------------------------------------------------
    hist = sf.run_historical(reg)
    f = hist.flows
    out.append(Check("1.2", "in-use stock 2010", f.loc[2010, "stock"], "320-380",
                     _within(f.loc[2010, "stock"], 320, 380), "Mt", "Glöser-Chahoud Fig. 5.12"))  # fmt: skip
    out.append(Check("1.2", "EoL outflow 2011", f.loc[2011, "eol"], "11.1 +/- 1.5",
                     abs(f.loc[2011, "eol"] - 11.1) <= 1.5, "Mt", "Glöser-Chahoud Fig. 5.8"))  # fmt: skip
    resid = float(hist.stock_residual().abs().max())
    out.append(Check("1.2", "max stock mass-balance residual", resid, "< 1e-6", resid < 1e-6))

    # --- 1.3 fates ----------------------------------------------------------------------------
    media = fates.partition(hist, reg.value("cu.tailings_share_of_ore_cu"))
    s11 = {"tailings": 3.55, "old_scrap": 5.20, "other_scrap_markets": 2.30,
           "not_recycled": 3.60, "dissipation": 0.45}  # fmt: skip
    for k, target in s11.items():
        v = media.loc[2011, "tailings"] if k == "tailings" else f.loc[2011, k]
        out.append(Check("1.3", f"2011 {k.replace('_', ' ')}", v, f"{target} +/- 15 %",
                         abs(v / target - 1) <= 0.15, "Mt", "Helbig 2020 Fig. S11"))  # fmt: skip
    cl = media.loc[2010, "cum_landfill"]
    out.append(Check("1.3", "cumulative landfill 2010", cl, "105-170", _within(cl, 105, 170), "Mt"))
    br = float(fates.balance_residual(media).abs().max())
    out.append(Check("1.3", "max ore-to-media balance residual", br, "< 0.1 %", br < 1e-3))

    # --- 1.4 demand ----------------------------------------------------------------------------
    r26 = ts.loc[2026, "refined"]
    out.append(Check("1.4", "refined demand 2026", r26, "26-32", _within(r26, 26, 32), "Mt"))
    sat = base.calibration["k_sat"] * ts["population"].iloc[-1] / 1e9
    conv = ts["stock"].iloc[-1] / sat - 1
    out.append(Check("1.4", "stock vs saturation target at horizon", conv * 100, "within 1 %",
                     abs(conv) < 0.01, "%"))  # fmt: skip
    pcmax = float(np.nanmax(ts["stock_per_capita"]))
    out.append(Check("1.4", "max per-capita stock", pcmax, f"<= {base.calibration['k_sat']:.0f}",
                     pcmax <= base.calibration["k_sat"] * (1 + 1e-9), "kg/person"))  # fmt: skip

    # --- 1.5 geology ---------------------------------------------------------------------------
    gt = geology.GradeTonnage(
        reg.value("geo.grade.gerst.mu_w"), reg.value("geo.grade.gerst.sigma2_w"), 1
    )
    out.append(Check("1.5", "median grade (Gerst global porphyry)", gt.median_grade, "0.52 +/- 0.01",
                     abs(gt.median_grade - 0.52) <= 0.01, "wt%"))  # fmt: skip
    for b in scenarios.BRACKETS:
        g10 = ts.loc[2010, f"{b}_grade_pct"]
        out.append(Check("1.5", f"mined grade 2010 ({b})", g10, "0.50-0.75 (obs. 0.62)",
                         _within(g10, 0.5, 0.75), "wt%", "Northey 2014: ~0.62 %"))  # fmt: skip

    # --- 1.6 energy ---------------------------------------------------------------------------
    for opt, grades, targets, depth, tol in (
        ("norgate_coarse", (0.01, 0.005, 0.0025), (72, 131, 249), 0.0, 0.02),
        ("koppelaar_surface", (0.005, 0.004, 0.003), (52, 95, 255), 300.0, 0.03),
    ):
        m = energy.EnergyModel.from_registry(reg, opt)
        got = m.intensity(np.array(grades), depth)
        dev = float(np.max(np.abs(got / np.array(targets) - 1)))
        out.append(Check("1.6", f"{opt} reproduces source points", dev * 100,
                         f"within {tol * 100:.0f} %", dev <= tol, "% max dev"))  # fmt: skip

    # --- 1.7 mass -------------------------------------------------------------------------------
    ore = float(mass.ore_mass(23.0, 0.0062)) / 1e3
    out.append(
        Check("1.7", "23 Mt Cu at 0.62 % -> ore", ore, "3.7 +/- 0.05", abs(ore - 3.7) <= 0.05, "Gt")
    )

    # --- independent cross-checks ---------------------------------------------------------------
    rr = f.loc[2011, "old_scrap"] / f.loc[2011, "eol"]
    out.append(Check("x", "EoL recycling rate 2011", rr, "0.46 +/- 0.05", abs(rr - 0.46) <= 0.05,
                     "", "Glöser-Chahoud Tab. 5.3"))  # fmt: skip
    pc = [ts.loc[y, "stock_per_capita"] for y in range(2000, 2007)]
    out.append(Check("x", "global per-capita stock 2000-2006 (range)", max(pc), "35-55",
                     min(pc) >= 35 and max(pc) <= 55, "kg/person",
                     f"min {min(pc):.1f}; UNEP 2010 Table 1"))  # fmt: skip
    ic = pd.read_csv(
        production.RAW_DIR / "icsg" / "icsg_world_copper_1960_2019.csv", index_col="year"
    )
    dev = (f.loc[1960:2019, "refined"] / (ic["refined_production_kt"] / 1e3) - 1) * 100
    mine_dev = (f.loc[1960:2019, "mine"] / (ic["mine_kt"] / 1e3) - 1) * 100
    worst = int(dev.abs().idxmax())
    out.append(Check("x", ICSG_ANNUAL, float(dev.abs().max()), "within 10 %",
                     float(dev.abs().max()) <= 10, "%",
                     f"worst {worst} ({dev[worst]:+.1f} %), of which USGS vs ICSG *mine* data "
                     f"differ {mine_dev[worst]:+.1f} %; ICSG Factbook 2020 p. 59"))  # fmt: skip
    dec = dev.groupby((dev.index // 10) * 10).mean()
    out.append(Check("x", "refined copper vs ICSG, decadal means (max abs)", float(dec.abs().max()),
                     "within 10 % (added after the annual check failed)",
                     float(dec.abs().max()) <= 10, "%",
                     "decades " + ", ".join(f"{d}s {v:+.1f}" for d, v in dec.items())))  # fmt: skip
    return out


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _band(ax, bands, name, color, label, x0=1900):
    if bands is None or f"{name}_p50" not in bands:
        return
    b = bands.loc[x0:]
    ax.fill_between(b.index, b[f"{name}_p05"], b[f"{name}_p95"], color=color, alpha=0.15,
                    linewidth=0, label=f"{label} 5-95 %")  # fmt: skip


def figures(base: scenarios.RunOutput, others: dict[str, scenarios.RunOutput], bands, outdir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    ts = base.timeseries
    files = {}

    def save(fig, name):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        p = outdir / f"{name}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files[name] = p.name

    # 1. mine output: history + projection, scenarios
    fig, ax = plt.subplots(figsize=(8, 4))
    _band(ax, bands, "mine", C_HARMSEN, "baseline")
    ax.plot(ts.index, ts["mine"], color=C_HARMSEN, lw=2, label="baseline")
    for (name, run), col in zip(others.items(), (C_KESLER, C_THIRD, C_FOURTH), strict=False):
        ax.plot(run.timeseries.index, run.timeseries["mine"], color=col, lw=1.5, label=name)
    ax.axvline(2025.5, color=INK2, lw=0.8, ls=":")
    ax.text(2030, ax.get_ylim()[1] * 0.95, "projection →", color=INK2, fontsize=8, va="top")
    _style(ax, "World copper mine output", "Mt Cu per year")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "mine_output")

    # 2. per-capita stock
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _band(ax, bands, "stock_per_capita", C_HARMSEN, "baseline", 1950)
    t = ts.loc[1950:]
    ax.plot(t.index, t["stock_per_capita"], color=C_HARMSEN, lw=2, label="baseline")
    ax.axhspan(35, 55, xmin=0, xmax=1, color=C_THIRD, alpha=0.08, lw=0)
    ax.text(1955, 57, "UNEP 2010 global range, 2000-2006", color=INK2, fontsize=8)
    _style(ax, "Copper in use per person", "kg per person")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "stock_per_capita")

    # 3-4. grade and depth by bracket (separate charts: one axis each)
    for col, title, unit in (
        ("grade_pct", "Mined ore grade", "wt% Cu"),
        ("depth_m", "Mean extraction depth", "m"),
    ):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for b, c in ((("harmsen", C_HARMSEN)), (("kesler", C_KESLER))):
            _band(ax, bands, f"{b}_{col}", c, b.capitalize())
            ax.plot(ts.index, ts[f"{b}_{col}"], color=c, lw=2, label=f"{b.capitalize()} bracket")
        _style(ax, f"{title} (baseline)", unit)
        ax.legend(frameon=False, fontsize=8)
        save(fig, col.split("_")[0])

    # 5. energy intensity and 6. annual primary energy
    for col, title, unit in (
        (
            "energy_intensity_mj_per_kg",
            "Primary energy per kg of copper from ore (Norgate coarse)",
            "MJ/kg",
        ),
        ("primary_energy_ej", "Annual primary energy for primary copper", "EJ per year"),
    ):
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for b, c in (("harmsen", C_HARMSEN), ("kesler", C_KESLER)):
            _band(ax, bands, f"{b}_{col}", c, b.capitalize())
            ax.plot(ts.index, ts[f"{b}_{col}"], color=c, lw=2, label=f"{b.capitalize()} bracket")
        _style(ax, title, unit)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        save(fig, col)

    # 7. remaining endowment
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for b, c in (("harmsen", C_HARMSEN), ("kesler", C_KESLER)):
        ax.plot(
            ts.index,
            ts[f"{b}_remaining_total"] / 1e3,
            color=c,
            lw=2,
            label=f"{b.capitalize()} bracket",
        )
    _style(ax, "Copper remaining in the ground (baseline)", "Bt Cu")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "remaining")
    return files


def _md_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build(out_dir: Path = REPORTS, mc_dir: Path | None = None) -> Path:
    base = scenarios.run_scenario(scenarios.load_scenario("baseline"))
    others = {
        n: scenarios.run_scenario(scenarios.load_scenario(n))
        for n in scenarios.list_scenarios()
        if n != "baseline"
    }
    mc_dir = mc_dir or export.RUNS_DIR / "baseline" / "mc"
    bands = (
        pd.read_parquet(mc_dir / "bands.parquet") if (mc_dir / "bands.parquet").is_file() else None
    )
    mcq = (
        pd.read_csv(mc_dir / "summary_quantiles.csv", index_col="metric")
        if bands is not None
        else None
    )
    mc_meta = (mc_dir / "mc_meta.txt").read_text() if bands is not None else ""

    checks = run_checks(base)
    figs = figures(base, others, bands, out_dir / "figures")
    s = base.summary
    n_pass = sum(c.passed for c in checks)

    md = ["# Phase 1 validation report — copper core\n",
          f"*Generated by `metalwheels report phase1` (metalwheels {__version__}, commit "
          f"`{export.git_commit()}`). Do not edit by hand; regenerate instead.*\n",
          f"**Result: {n_pass} of {len(checks)} checks pass.**"]  # fmt: skip
    failed = [c for c in checks if not c.passed]
    if failed:
        md.append(
            "Failing checks need an explanation accepted by Steve before Phase 2 (PLAN Task 1.10); see *Deviations* below."
        )
    md.append("\n## Calibration and validation checks\n")
    md.append(_md_table(
        [[c.task, c.name, _fmt(c.value) + (f" {c.unit}" if c.unit else ""), c.target,
          "PASS" if c.passed else "**FAIL**", c.note] for c in checks],
        ["Task", "Check", "Model", "Target", "Result", "Source / note"]))  # fmt: skip

    if failed:
        md.append("\n### Deviations\n")
        for c in failed:
            why = DEVIATIONS.get(c.name, "Not yet explained.")
            md.append(f"- **{c.name}** ({_fmt(c.value)} {c.unit} vs {c.target}). {why}")
    md.append("\n## Baseline results (both endowment brackets, side by side)\n")
    rows = []
    for label, key, unit in (
        ("Endowment", "endowment", "Mt Cu"), ("Endowment exhausted", "exhausted_year", "year"),
        ("Mined grade 2050", "grade_2050", "wt%"), ("Mined grade 2100", "grade_2100", "wt%"),
        ("Mean depth 2050", "depth_2050", "m"), ("Mean depth 2100", "depth_2100", "m"),
        ("Energy intensity 2050", "energy_intensity_2050", "MJ/kg"),
        ("Energy intensity 2100", "energy_intensity_2100", "MJ/kg"),
        ("Primary energy 2100", "primary_energy_2100", "EJ/yr"),
        ("Cumulative primary energy to horizon", "cum_primary_energy", "EJ"),
        ("Cumulative tailings volume at horizon", "cum_tailings_volume_end", "km³"),
    ):  # fmt: skip
        vals = []
        for b in scenarios.BRACKETS:
            v = s.get(f"{b}_{key}", (float("nan"), ""))[0]
            cell = _fmt(v)
            if mcq is not None and f"{b}_{key}" in mcq.index:
                q = mcq.loc[f"{b}_{key}"]
                cell += f" ({_fmt(q['p05'])}–{_fmt(q['p95'])})"
            vals.append(cell)
        rows.append([label, unit, *vals])
    md.append(_md_table(rows, ["Quantity", "Unit", "Harmsen 2013", "Kesler & Wilkinson 2008"]))
    if mcq is not None:
        md.append(
            f"\nBrackets in parentheses are Monte Carlo 5–95 % ranges ({mc_meta.strip().replace(chr(10), ', ')})."
        )
    else:
        md.append("\nMonte Carlo bands not found; run `metalwheels mc baseline` and regenerate.")

    md.append("\n## Recycling scenarios (policy from 2026)\n")
    rows = []
    for name, run in {"baseline": base, **others}.items():
        r = run.summary
        hy = r["harmsen_exhausted_year"][0]
        rows.append([name, _fmt(r["cum_mine_output_future"][0] / 1e3), _fmt(r["mine_2100"][0]),
                     "-" if math.isnan(hy) else f"{hy:.0f}", _fmt(r["harmsen_primary_energy_2100"][0]),
                     _fmt(r["kesler_cum_primary_energy"][0])])  # fmt: skip
    md.append(_md_table(rows, ["Scenario", "Mine output 2026–3025 (Bt Cu)", "Mine output 2100 (Mt/yr)",
                               "Harmsen exhausted", "Primary energy 2100, Harmsen (EJ/yr)",
                               "Cumulative primary energy, Kesler (EJ)"]))  # fmt: skip

    md.append("\n## Framing against the literature\n")
    md.append(
        "- **Northey et al. 2014** (GeRS-DeMo): supply-constrained scenarios built on *identified* "
        "resources peak at 25–27.5 Mt in 2029–2033 and fall to ~0.1–3 Mt by 2100; their unconstrained "
        "'static demand' reaches 43.5 Mt (2050) and 99.1 Mt (2100). This model is demand-driven with "
        "paced discovery, so it has no near-term peak: baseline mine output is "
        f"{_fmt(s['mine_2050'][0])} Mt in 2050 and {_fmt(s['mine_2100'][0])} Mt in 2100, between "
        "Northey's two framings because recycling supplies a growing share of demand.\n"
        "- **Harmsen et al. 2013** frame scarcity as rising energy along a depth-layered supply curve. "
        "Here grade stabilises near 0.55–0.65 % once paced discovery opens the undiscovered pool, and "
        "**depth becomes the main escalator**; at today's recycling rates the Harmsen endowment "
        f"(24.9 Bt to 5 km) is exhausted in {_fmt(s['harmsen_exhausted_year'][0])}, all layers within ~10 years "
        "of each other, because cost-ordered extraction draws on every layer at once.\n"
        "- **Kesler & Wilkinson 2008**: 89 Bt above 3.3 km is not exhausted by 3025 in any scenario, "
        "but extraction settles near ~1.9 km depth."
    )

    md.append("\n## Known simplifications (carried into Phase 2)\n")
    md.append(
        "1. Collection and processing rates and end-use shares are constant over 1900–2025 (2002–2012 means).\n"
        "2. Zero in-use stock in 1900 (pre-1900 copper ignored); likely why 2010 landfill sits near the low end.\n"
        "3. Tailings share 18 % in all years (early-century concentrator losses were probably higher).\n"
        "4. ICA 'Equipment' mapped to Glöser-Chahoud's consumer & electronics group; equal split within categories.\n"
        "5. Saturation = midpoint of today's more-developed-country per-capita stock (140–300 kg).\n"
        "6. Discovery is uniform across depth; the depth-order multiplier is calibrated, not physical.\n"
        "7. Energy covers primary copper only; secondary (scrap) energy still needs a source.\n"
        "8. When an endowment is exhausted, demand is still assumed met by the stock-flow model "
        "(shortfall reported; no feedback in Phase 1).\n"
        "9. Koppelaar's fit covers mining + concentrating only; Norgate is cradle-to-cathode."
    )
    md.append("\n## Figures\n")
    for name, f in figs.items():
        md.append(f"![{name.replace('_', ' ')}](figures/{f})\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase1_validation.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
