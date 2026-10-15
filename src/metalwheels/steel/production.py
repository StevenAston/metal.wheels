"""World crude (raw) steel and pig iron production, 1900-2025 (Phase 2 input).

Sources (raw files under ``data/raw/``, hashes in ``data/raw/MANIFEST.csv``):

* 1943-2021 raw steel and 1910-2021 pig iron: USGS Data Series 140, iron and steel statistics
  (2023 revision), "World production" columns.
* 2022-2025: USGS Mineral Commodity Summaries, iron and steel chapter, "World total (rounded)"
  rows, hand-transcribed into ``data/raw/usgs/mcs_world_iron_steel.csv`` (the most recent edition
  that reports each year as a non-estimate; 2025 is the MCS 2026 estimate).

DS140 has no world raw steel before 1943 and no world pig iron before 1910, so earlier years are
reconstructed from the US series in the same workbook (``status = reconstructed``):

* world pig iron 1900-1909 = US pig iron x the mean world/US pig iron ratio of 1910-1913;
* world raw steel 1900-1942 = world pig iron x r(t), a world steel/pig iron ratio that rises
  linearly from r0 (centred on 1911.5) to r1 (centred on 1951) and is held at r0 before 1911.5.
  r1 is the mean reported world ratio over 1947-1955; r0 = k x the mean US ratio over 1910-1913,
  where k is the mean of (world ratio / US ratio) over 1947-1955. Wartime 1943-1946 is excluded.
  The year-by-year US ratio is not used directly because it swings with US blast-furnace
  utilisation in the 1930s (2.1 in 1940), which would overstate world steel.

The reconstruction gives about 33 Mt for 1900, 74 Mt for 1913 and 138 Mt for 1940. These are
close to commonly quoted figures (roughly 28, 76 and 140 Mt), but that comparison is from memory
and has not been checked against a source. The series steps up from 132 Mt (1942,
reconstructed) to 172 Mt (1943, reported); the step is real wartime steelmaking that a smooth
ratio cannot capture. Pre-1950 steel is a small share of today's in-use stock.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import openpyxl

from metalwheels.copper.production import RAW_DIR, IngestError, verify_manifest
from metalwheels.logging_setup import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[3]
PROCESSED = REPO / "data" / "processed" / "steel_production.csv"
DS140_FILE = "usgs/ds140-iron-steel-2021.xlsx"
MCS_FILE = "usgs/mcs_world_iron_steel.csv"
FIRST_YEAR, SPLICE_YEAR, LAST_YEAR = 1900, 2021, 2025
PIG_SHARE_YEARS = range(1910, 1914)
RATIO_YEARS = range(1947, 1956)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def _sheet(wb, name: str, wanted: tuple[str, ...]) -> dict[int, dict[str, float | None]]:
    rows = list(wb[name].iter_rows(values_only=True))
    header = next((r for r in rows if r and r[0] == "Year"), None)
    if header is None:
        raise IngestError(f"{name}: no 'Year' header row")
    names = [" ".join(str(h).split()) for h in header if h]
    cols = {}
    for want in wanted:
        hits = [i for i, h in enumerate(header) if h and " ".join(str(h).split()) == want]
        if not hits:
            raise IngestError(f"{name}: no column {want!r} (have {names})")
        cols[want] = hits[0]
    out: dict[int, dict[str, float | None]] = {}
    for r in rows:
        if isinstance(r[0], int):
            out[r[0]] = {
                k: (float(r[i]) if isinstance(r[i], (int, float)) else None)
                for k, i in cols.items()
            }
    return out


def read_ds140(path: Path) -> tuple[dict, dict]:
    """``(pig, steel)``: per-year dicts with US and world production (t)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    pig = _sheet(wb, "Pig Iron", ("Primary production", "World production"))
    steel = _sheet(wb, "Steel", ("Raw steel production", "World production"))
    return pig, steel


def read_mcs(path: Path) -> dict[int, dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return {
            int(r["year"]): {
                "steel": float(r["raw_steel_mt"]) * 1e6,
                "pig": float(r["pig_iron_mt"]) * 1e6,
                "status": r["status"],
                "edition": r["mcs_edition"],
            }
            for r in csv.DictReader(fh)
        }


def _ratio(steel: dict, pig: dict, y: int, scope: str) -> float:
    if scope == "world":
        return steel[y]["World production"] / pig[y]["World production"]
    return steel[y]["Raw steel production"] / pig[y]["Primary production"]


def reconstruction_factors(pig: dict, steel: dict) -> dict[str, float]:
    """Factors for pre-1943 years: world/US pig share, k, ratio anchors r0 and r1."""
    pig_share = statistics.mean(
        pig[y]["World production"] / pig[y]["Primary production"] for y in PIG_SHARE_YEARS
    )
    k = statistics.mean(
        _ratio(steel, pig, y, "world") / _ratio(steel, pig, y, "us") for y in RATIO_YEARS
    )
    r0 = k * statistics.mean(_ratio(steel, pig, y, "us") for y in PIG_SHARE_YEARS)
    r1 = statistics.mean(_ratio(steel, pig, y, "world") for y in RATIO_YEARS)
    return {"pig_share": pig_share, "k": k, "r0": r0, "r1": r1}


def world_ratio(year: int, f: dict[str, float]) -> float:
    """Reconstructed world steel/pig iron ratio r(year)."""
    t0 = statistics.mean(PIG_SHARE_YEARS)
    t1 = statistics.mean(RATIO_YEARS)
    frac = min(max((year - t0) / (t1 - t0), 0.0), 1.0)
    return f["r0"] + frac * (f["r1"] - f["r0"])


def build(raw_dir: Path = RAW_DIR) -> list[dict]:
    verify_manifest(raw_dir)
    pig, steel = read_ds140(raw_dir / DS140_FILE)
    mcs = read_mcs(raw_dir / MCS_FILE)
    f = reconstruction_factors(pig, steel)
    log.info(
        "steel reconstruction: world/US pig %.4f, k %.4f, r0 %.4f, r1 %.4f",
        f["pig_share"], f["k"], f["r0"], f["r1"],
    )  # fmt: skip
    rows = []
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        if y <= SPLICE_YEAR:
            p, s = pig[y], steel[y]
            world_pig = p["World production"]
            pig_status = "reported"
            if world_pig is None:
                world_pig, pig_status = p["Primary production"] * f["pig_share"], "reconstructed"
            world_steel = s["World production"]
            status, source = "reported", "USGS DS140 iron and steel (2023 rev.)"
            if world_steel is None:
                world_steel, status = world_pig * world_ratio(y, f), "reconstructed"
                source = "Reconstructed from USGS DS140 US and world series"
        else:
            m = mcs[y]
            world_steel, world_pig, status = m["steel"], m["pig"], m["status"]
            pig_status, source = m["status"], f"USGS MCS {m['edition']}"
        rows.append(
            {
                "year": y,
                "crude_steel_mt": world_steel / 1e6,
                "pig_iron_mt": world_pig / 1e6,
                "source": source,
                "status": status,
                "pig_iron_status": pig_status,
            }
        )
    return rows


def write(rows: list[dict], path: Path = PROCESSED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **r,
                    "crude_steel_mt": f"{r['crude_steel_mt']:.3f}",
                    "pig_iron_mt": f"{r['pig_iron_mt']:.3f}",
                }
            )
    log.info("wrote %d rows to %s", len(rows), path)
    return path


def load(path: Path = PROCESSED, column: str = "crude_steel_mt") -> dict[int, float]:
    """``{year: Mt}`` for ``column`` (``crude_steel_mt`` or ``pig_iron_mt``)."""
    with path.open(newline="", encoding="utf-8") as fh:
        return {int(r["year"]): float(r[column]) for r in csv.DictReader(fh)}
