"""Task 4.4: synthesis report (``metalwheels report synthesis``).

Regenerates ``reports/synthesis.md``, its figures and ``reports/synthesis_snapshot.json``.

Traceability rule: every table row that carries a number has a ``trace`` cell naming where it
comes from - ``run <id>`` (a run listed in the report's run table, written to ``runs/<scenario>``
with that ID in ``meta.json``), ``reg <key>`` (a parameter in the registry snapshot), or
``file <path>`` (a hashed input file in the snapshot). Numbers in prose carry the same markers in
square brackets. The snapshot holds the full registry, its SHA-256 and the hashes of the
structured inputs (Metal Wheel transcription, ratchet catalogue, oxide thermochemistry, JANAF
tables).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metalwheels import __version__, export, registry, scenarios
from metalwheels.ratchets import carbon_cup, catalogue
from metalwheels.ratchets import metal_wheel as mwm
from metalwheels.report import (
    C_FOURTH,
    C_HARMSEN,
    C_KESLER,
    C_THIRD,
    INK2,
    REPORTS,
    SURFACE,
    _fmt,
    _md_table,
    _style,
)

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = ("baseline", "recycling_85", "recycling_95", "recycling_98", "carbon_cup")
YEARS = (2025, 2050, 2100, 2200, 2500)
SNAPSHOT_NAME = "synthesis_snapshot.json"
INPUT_FILES = (
    "params/ratchets.yaml",
    "params/metal_wheel.yaml",
    "params/carbon_cup.yaml",
    "data/processed/oxide_thermo.csv",
)
COLOURS = {
    "baseline": INK2,
    "recycling_85": C_THIRD,
    "recycling_95": C_HARMSEN,
    "recycling_98": C_FOURTH,
    "carbon_cup": C_KESLER,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(reg) -> dict:
    files = {f: _sha(REPO / f) for f in INPUT_FILES}
    for p in sorted((REPO / "data" / "raw" / "janaf").glob("*.txt")):
        files[str(p.relative_to(REPO))] = _sha(p)
    return {
        "registry_sha256": export.registry_hash(reg),
        "git_commit": export.git_commit(),
        "metalwheels_version": __version__,
        "files": files,
        "parameters": {k: p.model_dump() for k, p in sorted(reg.items())},
    }


@dataclass
class Runs:
    out: dict[str, scenarios.RunOutput]
    ids: dict[str, str]

    def t(self, name: str) -> str:
        return f"run {self.ids[name]}"


def _year(v: float) -> str:
    return "none" if v != v else f"{v:.0f}"


def _num(v: float, signed: bool = False) -> str:
    """Three significant figures, thousands separators from 1,000; '-' for NaN."""
    if v != v:
        return "-"
    if abs(v) >= 1000:
        return f"{v:+,.0f}" if signed else f"{v:,.0f}"
    if v == 0:
        return "0"
    return f"{v:+.3g}" if signed else f"{v:.3g}"


#: Carbon-cup sensitivity variants (overrides on scenario carbon_cup), each its own run.
SENSITIVITY = {
    "cup rates at low end": {"cc.collection": 0.67, "cc.processing": 0.85},
    "matrix costs +25 MJ/kg C": {"cc.matrix_net_energy": 25.0},
    "matrix returns 25 MJ/kg C": {"cc.matrix_net_energy": -25.0},
    "5 kg C/kg Cu, +25 MJ/kg C": {"cc.carbon_per_metal": 5.0, "cc.matrix_net_energy": 25.0},
    "no capture, fossil carbon": {"cc.capture_fraction": 0.0, "cc.biogenic_share": 0.0},
}


def _run_all(reg, runs_dir: Path) -> tuple[Runs, Runs]:
    out, ids = {}, {}
    for name in SCENARIOS:
        r = scenarios.run_scenario(scenarios.load_scenario(name), reg)
        export.write(r, reg, runs_dir / name)
        out[name], ids[name] = r, export.run_id(r, reg)
    sens_out, sens_ids = {}, {}
    cup = scenarios.load_scenario("carbon_cup")
    for i, (label, ov) in enumerate(SENSITIVITY.items()):
        sc = dataclasses.replace(cup, name=f"carbon_cup_s{i + 1}", overrides=ov)
        r = scenarios.run_scenario(sc, reg)
        export.write(r, reg, runs_dir / sc.name)
        sens_out[label], sens_ids[label] = r, export.run_id(r, reg)
    return Runs(out, ids), Runs(sens_out, sens_ids)


# ------------------------------------------------------------------------------- figures
def figures(runs: Runs, thermo: carbon_cup.Thermo, t_max: float, outdir: Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    names = []
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE, sharey=True)
    for ax, br in zip(axes, scenarios.BRACKETS, strict=True):
        for name, r in runs.out.items():
            s = r.timeseries.loc[2025:, f"cas_{br}_maintenance_energy_ej"]
            ax.plot(s.index, s, color=COLOURS[name], label=name, linewidth=1.2)
        _style(ax, f"Maintenance energy ({br.capitalize()} bracket)", "EJ/yr")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(outdir / "p4_maintenance_energy.png", dpi=150)
    plt.close(fig)
    names.append("p4_maintenance_energy.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), facecolor=SURFACE)
    for name, r in runs.out.items():
        ts = r.timeseries.loc[2000:]
        axes[0].plot(ts.index, ts["st_scrap_cu_mean"] * 100, color=COLOURS[name], label=name,
                     linewidth=1.2)  # fmt: skip
    _style(axes[0], "Copper in steel scrap", "wt%")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK2)
    cc = runs.out["carbon_cup"].timeseries.loc[2025:]
    for col, lab, colour in (
        ("cc_co2_captured", "captured", C_HARMSEN),
        ("cc_co2_emitted_fossil", "emitted, fossil", C_KESLER),
        ("cc_co2_emitted_biogenic", "emitted, biogenic", C_THIRD),
        ("cc_co2_net", "net to atmosphere", INK2),
    ):
        axes[1].plot(cc.index, cc[col], color=colour, label=lab, linewidth=1.2)
    _style(axes[1], "Carbon-cup CO2 stream", "Mt CO2/yr")
    axes[1].legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(outdir / "p4_steel_cu_and_co2.png", dpi=150)
    plt.close(fig)
    names.append("p4_steel_cu_and_co2.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    t = np.arange(300.0, 2600.0, 10.0)
    show = ("Cu", "Ni", "Pb", "Fe", "Zn", "Cr", "Mn", "Si", "Li", "Mg", "Al", "Ti", "La")
    cmap = plt.get_cmap("viridis")
    for i, el in enumerate(show):
        ax.plot(t - 273.15, thermo.dg_o2(el, t), color=cmap(i / len(show)), linewidth=1)
        g = thermo.dg_o2(el, t)
        k = int(np.flatnonzero(np.isfinite(g))[-1])
        ax.annotate(el, (t[k] - 273.15, g[k]), fontsize=7, color=INK2)
    ax.plot(t - 273.15, thermo.dg_co(t), color="k", linewidth=2, label="2C + O2 -> 2CO")
    ax.axvline(t_max - 273.15, color=C_KESLER, linestyle="--", linewidth=1, label="cc.t_max")
    _style(ax, "Ellingham lines (most stable oxide, per mol O2)", "dG (kJ/mol O2)")
    ax.set_xlabel("T (C)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "p4_ellingham.png", dpi=150)
    plt.close(fig)
    names.append("p4_ellingham.png")
    return names


# ------------------------------------------------------------------------------- sections
def _run_table(runs: Runs, snap: dict) -> str:
    rows = []
    for name, r in runs.out.items():
        sc = r.scenario
        rows.append([f"`{runs.ids[name]}`", name, json.dumps(sc.overrides) if sc.overrides else
                     "-", json.dumps(sc.policy) if sc.policy else "-",
                     f"`{snap['registry_sha256'][:12]}`", f"`{snap['git_commit']}`"])  # fmt: skip
    return _md_table(rows, ["Run ID", "Scenario", "Overrides", "Policy", "Registry", "Commit"])


def _catalogue_section(cat: catalogue.Catalogue, wheel: mwm.MetalWheel) -> list[str]:
    df = cat.frame()
    md = ["\n## 1. How many ratchets?\n"]
    counts = df.groupby("mechanism").size()
    modelled = df[df["modelled"]]
    md.append(
        f"The catalogue holds **{len(df)}** ratchets, every one sourced, across all five "
        "mechanism types: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in counts.items())
        + f". {len(modelled)} are already inside the model "
        f"({', '.join(modelled['id'])}); the rest are characterised (tolerance, energy step) but "
        "not yet coupled into the stock-flow or the cascade. The count is a floor, not a census: "
        "each Metal Wheel cell rated *dissolved, harmful* is a candidate tramp ratchet, and there "
        f"are {sum(c.fate == 'dissolved_harmful' for c in wheel.cells())} of them "
        "[file params/metal_wheel.yaml].\n"
    )  # fmt: skip
    rows = []
    for r, entry in zip(df.itertuples(), cat, strict=True):
        tol = (
            "-"
            if r.tolerance != r.tolerance
            else f"{r.tolerance * 100:g} wt%"
            + (
                f" ({r.tolerance_low * 100:g}-{r.tolerance_high * 100:g})"
                if r.tolerance_high != r.tolerance_low
                else ""
            )
        )
        es = (
            "-"
            if r.energy_step != r.energy_step
            else (
                f"{_num(r.energy_step)} {r.energy_unit} "
                f"({_num(r.energy_low)}-{_num(r.energy_high)})"
            )
        )
        keys = entry.param_keys()
        trace = "reg " + ", ".join(keys) if keys else f"file params/ratchets.yaml ({r.source})"
        rows.append([f"`{r.id}`", r.mechanism.replace("_", " "), r.host, r.agent, tol, es,
                     str(r.energy_basis or "-"), "yes" if r.modelled else "no", trace])  # fmt: skip
    md.append(_md_table(rows, ["Ratchet", "Mechanism", "Host", "Agent", "Tolerance",
                               "Energy step", "Per", "In model", "trace"]))  # fmt: skip
    md.append(
        "\nEnergy steps are the energy paid each time one unit goes through the ratchet: for "
        "tramp elements and route switches, primary minus secondary production energy of the host; "
        "for dissipation, the cradle-to-gate energy of replacing the lost metal from primary "
        "(Nuss and Eckelman 2014, Table S38). Ranges come from the parameters' own ranges.\n"
    )
    return md


def _interaction_section(cat: catalogue.Catalogue, runs: Runs, cmp: dict) -> list[str]:
    g = cat.interactions()
    md = ["\n## 2. How do they interact?\n",
          "Declared interactions (catalogue `interactions`):\n"]  # fmt: skip
    md.append(_md_table([[f"`{r['from']}`", r["kind"], f"`{r['to']}`", r["note"],
                          "file params/ratchets.yaml"] for _, r in g.iterrows()],
                        ["From", "Kind", "To", "Note", "trace"]))  # fmt: skip
    b, c = runs.t("baseline"), runs.t("carbon_cup")
    md.append(
        "\nThe one interaction the model can already measure end to end is copper's: copper that "
        "is collected but not separated (`cu_to_other_scrap`) becomes the copper load of steel "
        "scrap (`cu_in_steel`), and copper replaced from primary drives `cu_ore_grade` and "
        "`cu_depth`. Pulling copper out of that loop with the carbon cup moves all three at "
        f"once: the steel LP crossover moves from {_year(cmp['st_crossover_year_lp_base'][0])} "
        f"[{b}] to {_year(cmp['st_crossover_year_lp_cup'][0])} [{c}], the cumulative copper "
        f"penalty in steel from {_fmt(cmp['st_cum_cu_penalty_base'][0])} EJ to "
        f"{_fmt(cmp['st_cum_cu_penalty_cup'][0])} EJ, and the Harmsen bracket's first supply "
        f"shortfall from {_year(cmp['harmsen_first_shortfall_year_base'][0])} to "
        f"{_year(cmp['harmsen_first_shortfall_year_cup'][0])} [{c}]. A lever on one ratchet "
        "acts on its neighbours; levers should be compared on the whole chain.\n"
    )  # fmt: skip
    md.append(
        "The Metal Wheel makes the same point from the metallurgy side: whether an element is a "
        "ratchet depends on the carrier route it lands in. Nickel is a benign alloying element in "
        "steel but lost to the nickel cycle; gold and the PGMs are recovered in the copper route "
        "and lost (and harmful) in the steel route; zinc volatilises in the steel route and is "
        "the carrier of its own. The routes are the infrastructure, so the ratchet count depends "
        "on which infrastructures exist.\n"
    )
    return md


def _wheel_section(wheel: mwm.MetalWheel) -> list[str]:
    counts = wheel.counts()
    md = ["\n## 3. Metal Wheel compatibility matrix\n",
          f"{len(wheel.elements)} elements x {len(wheel.routes)} carrier routes, from the UNEP "
          "(2013) Metal Wheel (Fig. 15) and element radar chart after Nakajima et al. (Fig. 64), "
          "combined by the rules in `metalwheels.ratchets.metal_wheel` [file "
          "params/metal_wheel.yaml]. Cells neither figure resolves are `unknown`, "
          "explicitly.\n"]  # fmt: skip
    rows = [[r, *[str(int(counts.loc[r].get(f, 0))) for f in mwm.FATES],
             "file params/metal_wheel.yaml"] for r in wheel.routes]  # fmt: skip
    md.append(_md_table(rows, ["Route", *[f.replace("_", " ") for f in mwm.FATES], "trace"]))
    md.append("\n<details><summary>Full matrix</summary>\n\n" + wheel.to_markdown()
              + "\n\n</details>\n")  # fmt: skip
    return md


def _energy_section(runs: Runs, cat: catalogue.Catalogue) -> list[str]:
    md = ["\n## 4. What do they do to energy demand?\n",
          "Maintenance energy is the primary energy needed to deliver 2025 final demand with "
          "year-t technical coefficients (Phase 3), defined until the bracket's first copper "
          "supply shortfall.\n"]  # fmt: skip
    rows = []
    for name, r in runs.out.items():
        s = r.summary
        for br in scenarios.BRACKETS:
            p = f"cas_{br}_"
            vals = [_fmt(s.get(f"{p}maintenance_energy_{y}", (float("nan"),))[0]) for y in YEARS]
            rows.append([name, br, *vals, _year(s[f"{p}valid_until"][0]),
                         f"{s[f'{p}rho_2100'][0]:.4f}", runs.t(name)])  # fmt: skip
    md.append(_md_table(rows, ["Scenario", "Bracket", *[f"E {y} (EJ/yr)" for y in YEARS],
                               "Defined to", "rho 2100", "trace"]))  # fmt: skip
    md.append("\n![maintenance energy](figures/p4_maintenance_energy.png)\n")
    rows = []
    for name, r in runs.out.items():
        s = r.summary
        rows.append([name, _year(s["st_crossover_year_lp"][0]),
                     _year(s["st_crossover_year_aggregate"][0]),
                     _num(s["st_cum_cu_penalty_ej"][0]), f"{s['st_scrap_cu_2100'][0] * 100:.3f}",
                     f"{s['eol_recycling_rate_end'][0]:.3f}",
                     _year(s["harmsen_first_shortfall_year"][0]), runs.t(name)])  # fmt: skip
    md.append("\nSteel and copper ratchets by scenario:\n")
    md.append(
        _md_table(
            rows,
            [
                "Scenario",
                "Steel crossover (LP)",
                "Crossover (aggregate)",
                "Cum. Cu penalty in steel (EJ)",
                "Scrap Cu 2100 (wt%)",
                "Cu EoL recycling rate, end",
                "Harmsen shortfall",
                "trace",
            ],
        )
    )
    md.append("\n![steel copper and CO2](figures/p4_steel_cu_and_co2.png)\n")
    b = runs.out["baseline"].summary
    bt = runs.t("baseline")
    e = runs.out["baseline"].timeseries["cas_harmsen_maintenance_energy_ej"].loc[2025:].dropna()
    dev = float((e / e.iloc[0] - 1).abs().max() * 100)
    al = cat.energy_step("al_dilution")[0]
    st = cat.energy_step("cu_in_steel")[0]
    pt = cat.energy_step("pt_dissipation")[0]
    md.append(
        "\n**Reading.** Inside the agreed cascade boundary the modelled ratchets move maintenance "
        f"energy by at most {dev:.1f} % from its 2025 level in the Harmsen bracket [{bt}]: it is "
        f"{_fmt(b['cas_harmsen_maintenance_energy_2025'][0])} EJ/yr in 2025 and "
        f"{_fmt(b['cas_harmsen_maintenance_energy_2200'][0])} EJ/yr in 2200 [{bt}], with "
        f"rho(A) near {b['cas_harmsen_rho_2100'][0]:.3f} [{bt}]. Two things keep the signal "
        "small, and neither means the ratchets are small: the energy sector's own use dominates "
        "rho(A) (Phase 3 sensitivity), and only copper's ratchets and copper-in-steel are coupled "
        "into A(t). The per-unit energy steps in section 1 show where the larger effects sit once "
        f"coupled: aluminium route switches cost {_num(al)} MJ per kg diverted [reg "
        f"rt.al.energy_primary, rt.al.energy_recycling] against {_num(st)} MJ/kg for steel [reg "
        f"st.energy.bf_bof, st.energy.scrap_eaf], and dissipated platinum {_num(pt)} MJ/kg "
        "[reg rt.ced.pt]. The second clear effect is timing: the ratchets decide *when* the "
        "copper and steel constraints bind (crossover and shortfall years above), which a "
        "fixed-final-demand cascade shows as a change of date more than of level.\n"
    )  # fmt: skip
    return md


def _carbon_cup_section(runs: Runs, sens: Runs, cmp: dict, thermo, reg) -> list[str]:
    t_max = reg.value("cc.t_max")
    tab = thermo.table(t_max).sort_values("onset_K")
    md = ["\n## 5. Design lever: the carbon cup\n",
          "Carbon-reducible metals housed in burnable, separable carbon sub-matrices; at end of "
          "life the product is burnt, the metal reports to the ash as oxide and carbon reduces "
          "it. Eligibility is Ellingham-based: carbon must reduce the metal's most stable oxide "
          f"at or below cc.t_max = {t_max:.0f} K [reg cc.t_max].\n"]  # fmt: skip
    rows = [[r.element, r.oxide, r.method.replace("_", " "), f"{r.onset_K:.0f}",
             "yes" if r.eligible else "no", "file data/processed/oxide_thermo.csv"]
            for r in tab.itertuples()]  # fmt: skip
    md.append(_md_table(rows, ["Element", "Most stable oxide", "Data", "Onset (K)",
                               f"Eligible at {t_max:.0f} K", "trace"]))  # fmt: skip
    md.append("\n![Ellingham](figures/p4_ellingham.png)\n")
    c = runs.t("carbon_cup")
    rows = []
    for br in scenarios.BRACKETS:
        for y in (2050, 2100, 2200, 2500):
            de = cmp[f"d_maintenance_energy_{br}_{y}"][0]
            rows.append([br, str(y), "undefined (after a shortfall)" if de != de else
                         _num(de, signed=True), _num(cmp[f"d_rho_{br}_{y}"][0], signed=True),
                         f"{runs.t('baseline')}; {c}"])  # fmt: skip
        rows.append([br, f"cumulative to {_year(cmp[f'common_valid_until_{br}'][0])}",
                     f"{_num(cmp[f'd_maintenance_energy_cum_{br}'][0], signed=True)} EJ", "-",
                     f"{runs.t('baseline')}; {c}"])  # fmt: skip
    md.append("\nScenario `carbon_cup` against `baseline` (Delta = cup minus baseline):\n")
    md.append(_md_table(rows, ["Bracket", "Year", "Delta maintenance energy (EJ/yr)",
                               "Delta rho", "trace"]))  # fmt: skip
    rows = [[lab, *[_num(cmp[f"{k}_{y}"][0]) for y in (2050, 2100, 2200, 2500)],
             _num(cmp[f"{k}_cum"][0]), c]
            for lab, k in (("captured", "cc_co2_captured"),
                           ("emitted, fossil", "cc_co2_emitted_fossil"),
                           ("emitted, biogenic", "cc_co2_emitted_biogenic"),
                           ("net to atmosphere", "cc_co2_net"))]  # fmt: skip
    md.append("\nCO2 stream (Mt CO2/yr; cumulative Mt CO2):\n")
    md.append(_md_table(rows, ["Stream", "2050", "2100", "2200", "2500", "Cumulative", "trace"]))
    keys = sorted(k for k in reg if k.startswith("cc."))
    base = runs.out["baseline"]
    rows, signs = [], []
    for label, r in sens.out.items():
        _d, o = carbon_cup.compare(base, r)
        signs.append(o["d_maintenance_energy_kesler_2100"][0] < 0)
        rows.append([label, _num(o["d_maintenance_energy_harmsen_2100"][0], signed=True),
                     _num(o["d_maintenance_energy_kesler_2100"][0], signed=True),
                     _num(o["d_maintenance_energy_cum_kesler"][0], signed=True),
                     _num(o["cc_co2_net_cum"][0], signed=True),
                     _year(o["st_crossover_year_lp_cup"][0]),
                     f"{runs.t('baseline')}; {sens.t(label)}"])  # fmt: skip
    md.append(
        "\nEvery carbon-cup parameter is an explicit assumption with a wide range ["
        + "reg "
        + ", ".join(keys)
        + "]. Sensitivity of the result to them (each row its own "
        "run, against the baseline):\n"
    )
    md.append(_md_table(rows, ["Variant", "Delta E 2100, Harmsen (EJ/yr)",
                               "Delta E 2100, Kesler (EJ/yr)", "Cum. Delta E, Kesler (EJ)",
                               "Cum. net CO2 (Mt)", "Steel crossover (LP)", "trace"]))  # fmt: skip
    rr = {
        n: runs.out[n].summary["eol_recycling_rate_end"][0]
        for n in ("baseline", "recycling_85", "carbon_cup")
    }
    md.append(
        f"\nMaintenance energy falls in 2100 (Kesler) in {sum(signs)} of {len(signs)} variants. "
        "The direction holds while the matrix is cheap in energy; a carbon-heavy matrix that "
        "costs more energy to make than burning it returns can reverse it, so the net matrix "
        "energy is the parameter to pin down first. Separation and timing effects do not depend "
        "on it. For scale, the default cup (half of end-of-life copper by 2059) lifts copper's "
        f"end-of-run EoL recycling rate from {rr['baseline']:.3f} [{runs.t('baseline')}] to "
        f"{rr['carbon_cup']:.3f} [{c}], less than raising collection and processing to 0.85 "
        f"for all copper ({rr['recycling_85']:.3f} [{runs.t('recycling_85')}]); its distinct "
        "contribution is that it targets the copper that would otherwise reach steel scrap and "
        "turns the carbon into a capturable stream.\n"
    )  # fmt: skip
    return md


def _gaps_section(reg) -> list[str]:
    pend = sorted(k for k, p in reg.items() if p.status == "assumption")
    md = ["\n## 6. Open items\n",
          f"Parameters still resting on assumptions ({len(pend)}): "
          + ", ".join(f"`{k}`" for k in pend) + ".\n",
          "Not yet coupled into the model: the aluminium, glass, polymer and specialty-metal "
          "ratchets (characterised only); the Metal Wheel's `unknown` cells (mostly the Ni/Cr and "
          "Sn routes, which no radar sector covers); energy-sector quality (falling EROI) as a "
          "ratchet of its own.\n"]  # fmt: skip
    return md


# ------------------------------------------------------------------------------- build
def build(out_dir: Path = REPORTS, runs_dir: Path | None = None) -> Path:
    logging.getLogger("metalwheels").setLevel(logging.ERROR)
    reg = registry.load()
    snap = snapshot(reg)
    runs, sens = _run_all(reg, runs_dir or export.RUNS_DIR)
    cat = catalogue.load(reg=reg)
    wheel = mwm.MetalWheel.load()
    thermo = carbon_cup.Thermo()
    _df, cmp = carbon_cup.compare(runs.out["baseline"], runs.out["carbon_cup"])
    figures(runs, thermo, reg.value("cc.t_max"), out_dir / "figures")
    mismatches = catalogue.check_wheel(cat, wheel)

    md = ["# Synthesis — ratchets, interactions and maintenance energy\n",
          f"*Generated by `metalwheels report synthesis` (metalwheels {__version__}, commit "
          f"`{snap['git_commit']}`). Do not edit by hand; regenerate instead.*\n",
          f"Registry snapshot: `{SNAPSHOT_NAME}` (registry SHA-256 "
          f"`{snap['registry_sha256'][:16]}`; {len(snap['parameters'])} parameters; "
          f"{len(snap['files'])} hashed input files). Every numeric table row ends with a "
          "`trace` cell: `run <id>` (run table below; outputs in `runs/<scenario>/`), "
          "`reg <key>` (parameter in the snapshot) or `file <path>` (hashed input).\n",
          "## Runs\n", _run_table(runs, snap), "\n",
          "Carbon-cup sensitivity runs (scenario `carbon_cup` with overrides):\n",
          _run_table(sens, snap), "\n",
          f"Catalogue / Metal Wheel cross-check: "
          f"{'all expectations met' if not mismatches else '; '.join(mismatches)}.\n"]  # fmt: skip
    md += _catalogue_section(cat, wheel)
    md += _interaction_section(cat, runs, cmp)
    md += _wheel_section(wheel)
    md += _energy_section(runs, cat)
    md += _carbon_cup_section(runs, sens, cmp, thermo, reg)
    md += _gaps_section(reg)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SNAPSHOT_NAME).write_text(json.dumps(snap, indent=1) + "\n", encoding="utf-8")
    path = out_dir / "synthesis.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path
