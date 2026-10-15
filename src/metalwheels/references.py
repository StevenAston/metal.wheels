"""Bibliography (CSL-JSON) and cross-checks against the parameter registry.

``references/references.json`` is a CSL-JSON array; each entry's ``id`` is the key that
``source`` / ``range_source`` fields in ``params/*.yaml`` refer to. DOIs were resolved through
doi.org content negotiation (Crossref / DataCite) rather than typed by hand.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from metalwheels.registry import Registry


class BibliographyError(ValueError):
    """Raised when the bibliography is malformed."""


def default_path() -> Path:
    """``$METALWHEELS_REFERENCES`` if set, else the repository's ``references/references.json``."""
    env = os.environ.get("METALWHEELS_REFERENCES")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "references" / "references.json"


def load(path: Path | str | None = None) -> dict[str, dict]:
    """Return ``{id: entry}``. Raises on duplicate ids or entries without an id."""
    p = Path(path) if path is not None else default_path()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise BibliographyError(f"{p.name}: expected a CSL-JSON array")
    out: dict[str, dict] = {}
    for entry in data:
        key = entry.get("id")
        if not key:
            raise BibliographyError(f"{p.name}: entry without id: {entry.get('title')!r}")
        if key in out:
            raise BibliographyError(f"{p.name}: duplicate id {key!r}")
        out[key] = entry
    return out


def missing_links(bib: dict[str, dict]) -> list[str]:
    """Ids of entries with neither a DOI nor a URL."""
    return sorted(k for k, e in bib.items() if not (e.get("DOI") or e.get("URL")))


def unresolved_sources(reg: Registry, bib: dict[str, dict]) -> dict[str, list[str]]:
    """``{bibliography key: [parameter keys citing it]}`` for keys absent from ``bib``."""
    out: dict[str, list[str]] = {}
    for pkey, p in reg.items():
        for src in (p.source, p.range_source):
            if src and src not in bib:
                out.setdefault(src, []).append(pkey)
    return out


def link(entry: dict) -> str:
    """Best link for an entry: DOI URL if present, else URL."""
    doi = entry.get("DOI")
    return f"https://doi.org/{doi}" if doi else entry.get("URL", "")
