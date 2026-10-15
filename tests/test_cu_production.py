"""Task 1.1 done condition: 126 annual rows 1900-2025, no gaps; cumulative 1900-2015 within
+/-10 % of USGS's ~600 Mt past production; 2025 value 23 +/- 0.5 Mt; raw files hashed."""

import csv
import re
import shutil
import subprocess

import pytest

from metalwheels.copper import production


@pytest.fixture(scope="module")
def series():
    return production.load()


def test_126_rows_no_gaps(series):
    assert sorted(series) == list(range(1900, 2026))


def test_cumulative_1900_2015_near_600_mt(series):
    cum = sum(v for y, v in series.items() if y <= 2015)
    assert cum == pytest.approx(600, rel=0.10)


def test_2025_value(series):
    assert series[2025] == pytest.approx(23.0, abs=0.5)


def test_all_values_positive_and_plausible(series):
    assert all(0.4 < v < 30 for v in series.values())


def test_processed_csv_matches_rebuild_from_raw():
    rebuilt = production.build()
    with production.PROCESSED.open(newline="", encoding="utf-8") as fh:
        on_disk = list(csv.DictReader(fh))
    assert len(rebuilt) == len(on_disk)
    for r, d in zip(rebuilt, on_disk, strict=True):
        assert int(d["year"]) == r["year"]
        assert float(d["mine_production_mt"]) == pytest.approx(r["mine_production_mt"], abs=5e-5)
        assert d["source"] == r["source"] and d["status"] == r["status"]


def test_raw_hashes_verified():
    assert production.verify_manifest() >= 6


def test_tampered_raw_file_detected(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(production.RAW_DIR, raw)
    (raw / production.DS140_FILE).write_bytes(b"not a spreadsheet")
    with pytest.raises(production.IngestError, match="hash mismatch"):
        production.build(raw)


def test_splice_year_agrees():
    ds = production.read_ds140_world(production.RAW_DIR / production.DS140_FILE)
    mcs = production.read_mcs(production.RAW_DIR / production.MCS_FILE)
    assert ds[2020] == pytest.approx(mcs[2020][0], rel=production.OVERLAP_TOLERANCE)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_mcs_transcription_matches_pdfs():
    """Each transcribed value appears in its edition's 'World total (rounded)' row."""
    mcs = production.read_mcs(production.RAW_DIR / production.MCS_FILE)
    for year, (t, _status, edition) in mcs.items():
        pdf = production.RAW_DIR / "usgs" / f"mcs{edition}-copper.pdf"
        text = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
        ).stdout
        line = next(ln for ln in text.splitlines() if "World total" in ln)
        mine = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", line)[:2]]
        # MCS edition N reports years N-2 (reported) and N-1 (estimated).
        idx = year - (int(edition) - 2)
        assert mine[idx] == round(t / 1e3), (year, edition, line)
