"""World crude steel / pig iron series 1900-2025 (Phase 2 input)."""

import csv
import re
import shutil
import subprocess

import pytest

from metalwheels.copper.production import RAW_DIR
from metalwheels.steel import production


@pytest.fixture(scope="module")
def steel():
    return production.load()


def test_126_rows_no_gaps(steel):
    assert sorted(steel) == list(range(1900, 2026))


def test_reported_anchor_years(steel):
    assert steel[1950] == pytest.approx(189.0)
    assert steel[2008] == pytest.approx(1330.0)
    assert steel[2025] == pytest.approx(1900.0)


def test_reconstruction_is_smooth_and_plausible(steel):
    pre = [steel[y] for y in range(1900, 1943)]
    assert all(15 < v < 180 for v in pre)
    assert 25 < steel[1900] < 40 and 60 < steel[1913] < 90
    # no 1930s spike from US blast-furnace utilisation (the year-by-year US ratio gives ~213 Mt)
    assert steel[1940] < 160


def test_ratio_anchors():
    pig, st = production.read_ds140(RAW_DIR / production.DS140_FILE)
    f = production.reconstruction_factors(pig, st)
    assert 0.85 < f["k"] < 1.0
    assert 1.3 < f["r1"] < 1.45
    assert f["r0"] < f["r1"]
    assert production.world_ratio(1900, f) == pytest.approx(f["r0"])
    assert production.world_ratio(1960, f) == pytest.approx(f["r1"])


def test_processed_csv_matches_rebuild():
    rebuilt = production.build()
    with production.PROCESSED.open(newline="", encoding="utf-8") as fh:
        on_disk = list(csv.DictReader(fh))
    assert len(rebuilt) == len(on_disk)
    for r, d in zip(rebuilt, on_disk, strict=True):
        assert float(d["crude_steel_mt"]) == pytest.approx(r["crude_steel_mt"], abs=5e-4)
        assert float(d["pig_iron_mt"]) == pytest.approx(r["pig_iron_mt"], abs=5e-4)
        assert d["status"] == r["status"]


def test_splice_agrees_with_mcs_2023():
    """MCS 2023 reports 2021 world raw steel as 1,950 Mt, the same as DS140."""
    assert production.load()[2021] == pytest.approx(1950.0)


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_mcs_transcription_matches_pdfs():
    mcs = production.read_mcs(RAW_DIR / production.MCS_FILE)
    for year, m in mcs.items():
        pdf = RAW_DIR / "usgs" / f"mcs{m['edition']}-iron-steel.pdf"
        text = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
        ).stdout
        line = next(ln for ln in text.splitlines() if "World total" in ln)
        nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", line)[:4]]
        idx = year - (int(m["edition"]) - 2)
        assert nums[idx] == round(m["pig"] / 1e6), (year, line)
        assert nums[2 + idx] == round(m["steel"] / 1e6), (year, line)
