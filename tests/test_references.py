"""Task 0.3 done condition: every params source resolves; every entry has a DOI or URL."""

import json

import pytest

from metalwheels import references, registry


@pytest.fixture(scope="module")
def bib():
    return references.load()


def test_every_param_source_resolves(bib):
    unresolved = references.unresolved_sources(registry.load(), bib)
    assert unresolved == {}, f"unresolved bibliography keys: {unresolved}"


def test_every_entry_has_doi_or_url(bib):
    assert references.missing_links(bib) == []


def test_entries_have_title_and_year(bib):
    for key, e in bib.items():
        assert e.get("title"), key
        assert e.get("issued", {}).get("date-parts", [[None]])[0][0], key


def test_planned_sources_present(bib):
    """The Task 0.3 list in PLAN.md (CEEC 2021 and IEA pending identification)."""
    expected = {
        "gloser_2013",
        "gloser_chahoud_2017",
        "helbig_2020",
        "charpentier_poncelet_2022",
        "lifset_2012",
        "gerst_2008",
        "mudd_jowitt_2018",
        "northey_2014",
        "koppelaar_2016",
        "norgate_jahanshahi_2010",
        "harmsen_2013",
        "kesler_wilkinson_2008",
        "usgs_fs_2014_3004",
        "usgs_mcs_2026",
        "daehn_2017",
        "daehn_2019",
        "ciacci_2015",
        "unep_2013_metal_recycling",
        "valero_valero_2014",
    }
    assert expected <= set(bib)


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"id": "a", "title": "x"}, {"id": "a", "title": "y"}]))
    with pytest.raises(references.BibliographyError, match="duplicate"):
        references.load(p)


def test_link_prefers_doi():
    assert references.link({"DOI": "10.1/x", "URL": "https://e.org"}) == "https://doi.org/10.1/x"
    assert references.link({"URL": "https://e.org"}) == "https://e.org"
