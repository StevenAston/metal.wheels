"""Task 4.4 done condition: the synthesis report regenerates from one CLI command and every
number in it traces to a run ID and the parameter registry snapshot."""

import json
import re

import pytest
from typer.testing import CliRunner

from metalwheels import cli, export, registry, report_synthesis

TRACE = re.compile(r"^(run [0-9a-f]{12}(; run [0-9a-f]{12})*|reg [\w.,\s]+|file \S+.*)$")
NUMERIC_PROSE = re.compile(r"\d\.\d|\d,\d{3}|\d\s*%|\d\s*(EJ|MJ|Mt)\b")
MARKER = re.compile(r"\[(run [0-9a-f]{12}|reg [^\]]+|file [^\]]+)\]")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthesis")
    path = report_synthesis.build(out_dir=d, runs_dir=d / "runs")
    snap = json.loads((d / report_synthesis.SNAPSHOT_NAME).read_text())
    return d, path.read_text(encoding="utf-8"), snap


def _tables(md: str):
    lines, out, cur = md.splitlines(), [], []
    for ln in [*lines, ""]:
        if ln.startswith("|"):
            cur.append([c.strip() for c in ln.strip("|").split("|")])
        elif cur:
            out.append(cur)
            cur = []
    return out


def _run_ids(md):
    ids = set()
    for t in _tables(md):
        if t[0][0] == "Run ID":
            ids |= {row[0].strip("`") for row in t[2:]}
    return ids


def test_snapshot_matches_registry(built):
    _, md, snap = built
    reg = registry.load()
    assert snap["registry_sha256"] == export.registry_hash(reg)
    assert set(snap["parameters"]) == set(reg)
    assert snap["registry_sha256"][:16] in md


def test_every_numeric_table_row_has_a_valid_trace(built):
    _d, md, snap = built
    ids = _run_ids(md)
    assert len(ids) == len(report_synthesis.SCENARIOS) + len(report_synthesis.SENSITIVITY)
    for t in _tables(md):
        head, rows = t[0], t[2:]
        if head[0] == "Run ID":
            continue
        if "trace" not in head:
            assert not any(re.search(r"\d", c) for row in rows for c in row), head
            continue
        k = head.index("trace")
        for row in rows:
            tr = row[k]
            assert TRACE.match(tr), (head, row)
            for rid in re.findall(r"run ([0-9a-f]{12})", tr):
                assert rid in ids
            if tr.startswith("reg "):
                for key in tr[4:].split(","):
                    assert key.strip() in snap["parameters"], key
            if tr.startswith("file "):
                assert tr.split()[1] in snap["files"], tr


def test_numeric_prose_carries_markers(built):
    _, md, snap = built
    ids = _run_ids(md)
    body = md.splitlines()[6:]
    for ln in body:
        if ln.startswith(("|", "#", "!", "<", "Legend", "Routes")) or not ln.strip():
            continue
        if NUMERIC_PROSE.search(ln):
            marks = MARKER.findall(ln)
            assert marks, ln
            for m in marks:
                if m.startswith("run "):
                    assert m[4:] in ids
                elif m.startswith("reg "):
                    for key in m[4:].split(","):
                        assert key.strip() in snap["parameters"], key


def test_run_ids_match_written_runs(built):
    d, md, _ = built
    written = {json.loads(p.read_text())["run_id"] for p in (d / "runs").glob("*/meta.json")}
    assert _run_ids(md) == written


def test_figures_written(built):
    d, md, _ = built
    for name in re.findall(r"\(figures/([\w.]+)\)", md):
        assert (d / "figures" / name).is_file()


def test_cli_command_is_registered():
    res = CliRunner().invoke(cli.app, ["report", "--help"])
    assert res.exit_code == 0 and "synthesis" in res.output
