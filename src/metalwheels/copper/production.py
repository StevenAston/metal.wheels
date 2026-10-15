"""Task 1.1: world copper mine production, 1900-2025.

Sources (raw files under ``data/raw/``, hashes in ``data/raw/MANIFEST.csv``):

* 1900-2020: USGS Data Series 140, copper statistics (2023 revision), column
  "World production" (world mine production, t Cu content).
* 2021-2025: USGS Mineral Commodity Summaries, copper chapter, "World total (rounded)"
  mine production. For each year the most recent edition that reports it as a non-estimate
  is used; 2025 is the MCS 2026 estimate. Values are hand-transcribed into
  ``data/raw/usgs/mcs_world_mine_copper.csv`` and checked against the PDFs by the tests.

Output: ``data/processed/cu_mine_production.csv`` with columns
``year, mine_production_mt, source, status``.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import openpyxl

from metalwheels.logging_setup import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[3]
RAW_DIR = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed" / "cu_mine_production.csv"

DS140_FILE = "usgs/ds140-copper-2020.xlsx"
MCS_FILE = "usgs/mcs_world_mine_copper.csv"
FIRST_YEAR, SPLICE_YEAR, LAST_YEAR = 1900, 2020, 2025
#: Maximum relative disagreement tolerated where DS140 and MCS overlap.
OVERLAP_TOLERANCE = 0.01


class IngestError(RuntimeError):
    """Raised when raw inputs are missing, altered or inconsistent."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(raw_dir: Path = RAW_DIR) -> int:
    """Check every hashed file in MANIFEST.csv. Returns the number verified."""
    n = 0
    with (raw_dir / "MANIFEST.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["sha256"]:
                continue
            path = raw_dir / row["path"]
            if not path.is_file():
                raise IngestError(f"raw file missing: {path}")
            got = sha256(path)
            if got != row["sha256"]:
                raise IngestError(f"hash mismatch for {row['path']}: {got} != {row['sha256']}")
            n += 1
    return n


def read_ds140_world(path: Path) -> dict[int, float]:
    """World production (t Cu) by year from a DS140 copper workbook."""
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = next((r for r in rows if r and r[0] == "Year"), None)
    if header is None or "World production" not in header:
        raise IngestError(f"{path.name}: no 'Year' header row with a 'World production' column")
    col = header.index("World production")
    out: dict[int, float] = {}
    for r in rows:
        if isinstance(r[0], int) and isinstance(r[col], (int, float)):
            out[r[0]] = float(r[col])
    return out


def read_mcs(path: Path) -> dict[int, tuple[float, str, str]]:
    """``{year: (t Cu, status, edition)}`` from the MCS transcription CSV."""
    out: dict[int, tuple[float, str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["year"])] = (
                float(row["world_mine_kt"]) * 1e3,
                row["status"],
                row["mcs_edition"],
            )
    return out


def build(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Assemble the 1900-2025 series. Raises if any year is missing."""
    verify_manifest(raw_dir)
    ds140 = read_ds140_world(raw_dir / DS140_FILE)
    mcs = read_mcs(raw_dir / MCS_FILE)

    if SPLICE_YEAR in ds140 and SPLICE_YEAR in mcs:
        a, b = ds140[SPLICE_YEAR], mcs[SPLICE_YEAR][0]
        rel = abs(a - b) / b
        log.info(
            "splice check %d: DS140 %.0f t vs MCS %.0f t (%.2f %%)", SPLICE_YEAR, a, b, 100 * rel
        )
        if rel > OVERLAP_TOLERANCE:
            raise IngestError(f"DS140 and MCS disagree by {rel:.1%} in {SPLICE_YEAR}")

    rows: list[dict] = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        if year <= SPLICE_YEAR:
            if year not in ds140:
                raise IngestError(f"DS140 has no world production for {year}")
            t, source, status = ds140[year], "USGS DS140 copper (2023 rev.)", "reported"
        else:
            if year not in mcs:
                raise IngestError(f"MCS transcription has no value for {year}")
            t, status, edition = mcs[year]
            source = f"USGS MCS {edition}"
        rows.append(
            {"year": year, "mine_production_mt": t / 1e6, "source": source, "status": status}
        )
    return rows


def write(rows: list[dict], path: Path = PROCESSED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "mine_production_mt": f"{r['mine_production_mt']:.4f}"})
    log.info("wrote %d rows to %s", len(rows), path)
    return path


def load(path: Path = PROCESSED) -> dict[int, float]:
    """``{year: Mt Cu}`` from the processed CSV."""
    with path.open(newline="", encoding="utf-8") as fh:
        return {int(r["year"]): float(r["mine_production_mt"]) for r in csv.DictReader(fh)}
