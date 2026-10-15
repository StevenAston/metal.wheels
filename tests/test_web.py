"""Task 1.11 done condition (Playwright): page loads, every control changes the charts, three
spot-check values equal the Python outputs within 0.5 %, and the page renders at 375 px width
without horizontal scroll."""

import re

import pytest

pw = pytest.importorskip("playwright.sync_api")

from metalwheels import scenarios, webexport  # noqa: E402


@pytest.fixture(scope="module")
def page_file(tmp_path_factory):
    out = tmp_path_factory.mktemp("web")
    data = webexport.build_grid(webexport.GRID_SMALL)
    local, _ = webexport.write_pages(data, out)
    return local


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # browser binaries not installed
            pytest.skip(f"chromium unavailable: {exc}")
        yield b
        b.close()


def _open(browser, page_file, width=1200, scheme="light"):
    page = browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(page_file.as_uri())
    page.wait_for_selector("body[data-ready='1']", timeout=15000)
    return page, errors


def _num(text: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", text.replace(",", "")))


def _paths(page, chart_id):
    return page.eval_on_selector_all(
        f"#{chart_id} svg path", "els => els.map(e => e.getAttribute('d'))"
    )


def test_page_loads_without_errors(browser, page_file):
    page, errors = _open(browser, page_file)
    assert page.title() == "Copper to 3025"
    assert len(page.query_selector_all(".chart svg")) == 8
    page.close()
    assert [e for e in errors if "fonts.g" not in e] == []


def test_every_control_changes_charts(browser, page_file):
    page, _ = _open(browser, page_file)
    before = _paths(page, "ch-mine")
    page.click("#pre-95")
    after_preset = _paths(page, "ch-mine")
    assert after_preset != before
    page.locator("#c").fill("0")  # collection slider back to 0.67, processing stays 0.95
    assert _paths(page, "ch-mine") not in (before, after_preset)
    page.locator("#p").fill("0")
    assert _paths(page, "ch-mine") == before
    page.click("#win-2100")
    assert _paths(page, "ch-mine") != before
    lay = _paths(page, "ch-layers")
    page.click("#lay-kesler")
    assert _paths(page, "ch-layers") != lay
    page.close()


def test_spot_checks_match_python(browser, page_file):
    page, _ = _open(browser, page_file)
    page.click("#pre-95")
    out = scenarios.run_scenario(webexport.grid_scenario(0.95, 0.95, 220.0)).summary
    checks = {
        "#ro-cum-mine": out["cum_mine_output_future"][0] / 1000,
        "#kv-harmsen-grade_2100": out["harmsen_grade_2100"][0],
        "#kv-kesler-energy_intensity_2100": out["kesler_energy_intensity_2100"][0],
    }
    for sel, expected in checks.items():
        shown = _num(page.inner_text(sel))
        assert shown == pytest.approx(expected, rel=0.005), sel
    page.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_no_horizontal_scroll_at_375px(browser, page_file, scheme):
    page, _ = _open(browser, page_file, width=375, scheme=scheme)
    sw = page.evaluate("document.documentElement.scrollWidth")
    assert sw <= 375
    page.close()
