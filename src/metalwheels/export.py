"""Run outputs: Parquet time series, summary CSV and run metadata (Task 1.8).

Layout: ``runs/<scenario>/timeseries.parquet``, ``summary.csv``, ``meta.json``.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from metalwheels import __version__
from metalwheels.copper.stockflow import FLOW_COLUMNS
from metalwheels.registry import Registry
from metalwheels.scenarios import BRACKETS, STEEL_COLUMNS, RunOutput

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "runs"

_MEDIA = [
    "media_ore_cu",
    "media_tailings",
    "media_tailings_and_slag",
    "media_other_scrap_markets",
    "media_landfill",
    "media_environment",
    "media_cum_tailings_and_slag",
    "media_cum_other_scrap_markets",
    "media_cum_landfill",
    "media_cum_environment",
    "media_cum_ore_cu",
]
_BRACKET = [
    "grade_pct",
    "depth_m",
    "discovered_fraction",
    "extracted_cu",
    "shortfall_cu",
    "primary_cu",
    "energy_intensity_mj_per_kg",
    "primary_energy_ej",
    "ore",
    "cum_ore",
    "waste_rock",
    "cum_waste_rock",
    "tailings",
    "cum_tailings",
    "tailings_volume_km3",
    "cum_tailings_volume_km3",
    "rock_moved",
    "cum_rock_moved",
    "remaining_total",
]
_LAYERS = {
    "harmsen": ["0_1km", "1_2km", "2_3km", "3_4km", "4_5km"],
    "kesler": ["0_1km", "1_2km", "2_3km", "3_4km"],
}


_STEEL_EXTRA = ("stock_per_capita", "cu_to_other_sinks", "energy_ej", "cu_penalty_ej")


def schema(steel: bool = True) -> dict[str, str]:
    """Declared time-series columns and dtypes (index ``year``, int64). ``steel`` adds the
    Phase 2 ``st_*`` columns."""
    cols = {c: "float64" for c in FLOW_COLUMNS if c not in ("target_stock", "demand_driven")}
    cols["demand_driven"] = "bool"
    cols["target_stock"] = "float64"
    cols.update(
        {"population": "float64", "stock_per_capita": "float64", "ore_cu_demand": "float64"}
    )
    cols.update({c: "float64" for c in _MEDIA})
    for br in BRACKETS:
        cols.update({f"{br}_{c}": "float64" for c in _BRACKET})
        cols.update({f"{br}_remaining_{layer}": "float64" for layer in _LAYERS[br]})
    if steel:
        cols.update({f"st_{c}": "float64" for c in (*STEEL_COLUMNS, *_STEEL_EXTRA)})
    return cols


def validate(ts: pd.DataFrame) -> None:
    """Raise ``ValueError`` unless ``ts`` matches :func:`schema` exactly (with or without the
    steel columns, inferred from their presence)."""
    want = schema(steel=any(c.startswith("st_") for c in ts.columns))
    missing, extra = set(want) - set(ts.columns), set(ts.columns) - set(want)
    if missing or extra:
        raise ValueError(f"schema mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}")
    wrong = {c: str(ts[c].dtype) for c, d in want.items() if str(ts[c].dtype) != d}
    if wrong:
        raise ValueError(f"wrong dtypes: {wrong}")
    if ts.index.name != "year" or str(ts.index.dtype) != "int64":
        raise ValueError("index must be 'year' (int64)")


def registry_hash(reg: Registry) -> str:
    payload = json.dumps({k: p.model_dump() for k, p in sorted(reg.items())}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write(run: RunOutput, reg: Registry, out_dir: Path | None = None) -> Path:
    """Write a run to ``out_dir`` (default ``runs/<scenario>``); returns the directory."""
    validate(run.timeseries)
    d = Path(out_dir) if out_dir else RUNS_DIR / run.scenario.name
    d.mkdir(parents=True, exist_ok=True)
    run.timeseries.to_parquet(d / "timeseries.parquet")
    with (d / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["metric", "value", "unit"])
        for k, (v, unit) in run.summary.items():
            w.writerow([k, f"{v:.6g}", unit])
    meta = {
        "scenario": run.scenario.name,
        "description": run.scenario.description,
        "overrides": run.scenario.overrides,
        "policy": run.scenario.policy,
        "end_year": run.scenario.end_year,
        "energy_option": run.scenario.energy_option,
        "steel": run.scenario.steel,
        "metalwheels_version": __version__,
        "git_commit": git_commit(),
        "registry_sha256": registry_hash(reg),
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "runtime_s": round(run.runtime_s, 3),
        "calibration": run.calibration,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return d
