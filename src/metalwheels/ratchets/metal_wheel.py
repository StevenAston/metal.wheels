"""Task 4.2: Metal Wheel compatibility matrix (element x carrier route -> fate).

The transcriptions in ``params/metal_wheel.yaml`` (UNEP 2013 Fig. 15 "Metal Wheel" and Fig. 64
"element radar chart") are combined cell by cell with fixed, documented rules. Each cell gets one
primary fate, any secondary fates the figures show, the rule that decided it, and its locators.

Fates:

* ``recovered`` - the route's carrier metal, a wheel symbol coloured "mainly recovered", or a
  radar symbol in the metal phase marked "recoverable";
* ``dissolved_benign`` / ``dissolved_harmful`` - ends up in the carrier metal and cannot be
  removed; benign or harmful by the wheel colour (orange "not detrimental" / red "detrimental"),
  or benign when the radar marks it an alloying element;
* ``dissolved_unrated`` - the radar puts it in the metal phase but neither figure rates its
  effect. This category is an addition to the four in PLAN.md, so that a known phase is not
  reported as ``unknown``;
* ``lost_to_slag`` - radar slag phase, or wheel oxide ring (``sink``);
* ``volatilised`` - radar gas phase, or a wheel ``compound`` symbol on a pyrometallurgical route
  for an element in the sourced volatile set (EAF dust and radar gas-phase elements);
* ``unknown`` - neither figure places the element on that route, or the wheel puts it in the
  "dust, slime, speiss, slag" ring and no source says which of these.

Decision order (first match wins): carrier -> wheel "recovered" -> radar phase -> wheel ring.
The wheel's BAT infrastructure recovers green elements from by-products (dusts, slimes, speiss),
which is why a green wheel symbol outranks a radar slag or gas phase.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from metalwheels.registry import default_params_dir

FATES = (
    "recovered",
    "dissolved_benign",
    "dissolved_harmful",
    "dissolved_unrated",
    "lost_to_slag",
    "volatilised",
    "unknown",
)
DISSOLVED = frozenset({"dissolved_benign", "dissolved_harmful", "dissolved_unrated"})
RINGS = ("metal", "compound", "sink")
PHASES = ("metal", "slag", "gas")
COLOURS = {"G": "recovered", "O": "lost, not detrimental", "R": "lost, detrimental"}
MARKERS = {"R": "recoverable", "A": "alloying", "D": "deoxidation agent", "-": None}
_PHASE_FATE = {"slag": "lost_to_slag", "gas": "volatilised"}
_RING_PHASE = {"metal": "metal", "sink": "slag"}

#: Short symbols for the rendered table.
SYMBOL = {
    "recovered": "R",
    "dissolved_benign": "d",
    "dissolved_harmful": "D!",
    "dissolved_unrated": "d?",
    "lost_to_slag": "S",
    "volatilised": "V",
    "unknown": "?",
}


class MetalWheelError(ValueError):
    """Raised when the transcription file is inconsistent."""


@dataclass(frozen=True)
class Cell:
    route: str
    element: str
    fate: str
    secondary: tuple[str, ...]
    rule: str
    wheel: tuple[tuple[str, str], ...]  # (ring, colour) in the order rings are listed
    radar: tuple[str, ...]  # phases, primary first
    radar_marker: str | None
    locator: str
    note: str = ""

    @property
    def sourced(self) -> bool:
        return self.fate != "unknown" and bool(self.locator)


@dataclass
class MetalWheel:
    routes: dict[str, dict]
    elements: list[str]
    wheel: dict[str, dict]
    radar: dict[str, dict]
    volatile: frozenset[str]
    pyro_routes: frozenset[str]
    source: str
    wheel_locator: str
    radar_locator: str
    _cells: dict[tuple[str, str], Cell] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path | str | None = None) -> MetalWheel:
        p = Path(path) if path is not None else default_params_dir() / "metal_wheel.yaml"
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        mw = cls(
            routes=doc["routes"],
            elements=list(doc["elements"]),
            wheel=doc["wheel"],
            radar=doc["radar"],
            volatile=frozenset(doc["volatile"]["members"]),
            pyro_routes=frozenset(doc["volatile"]["pyrometallurgical_routes"]),
            source=doc["source"],
            wheel_locator=doc["wheel_locator"],
            radar_locator=doc["radar_locator"],
        )
        mw.validate()
        return mw

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        if len(set(self.elements)) != len(self.elements):
            raise MetalWheelError("duplicate elements")
        for r in self.routes:
            if r not in self.wheel:
                raise MetalWheelError(f"route {r!r} has no wheel slice")
        for r, sl in self.wheel.items():
            if r not in self.routes:
                raise MetalWheelError(f"wheel slice for unknown route {r!r}")
            for ring in RINGS:
                for el, col in (sl.get(ring) or {}).items():
                    if col not in COLOURS:
                        raise MetalWheelError(f"wheel {r}.{ring}.{el}: bad colour {col!r}")
            for el, order in (sl.get("boundary") or {}).items():
                # the first ring listed must be the one the symbol is transcribed in
                if not set(order) <= set(RINGS) or el not in (sl.get(order[0]) or {}):
                    raise MetalWheelError(f"wheel {r}.boundary.{el}: bad rings {order}")
        for r, sec in self.radar.items():
            if r not in self.routes:
                raise MetalWheelError(f"radar sector for unknown route {r!r}")
            seen: dict[str, str] = {}
            for ph in PHASES:
                for el, mk in (sec.get(ph) or {}).items():
                    if mk not in MARKERS:
                        raise MetalWheelError(f"radar {r}.{ph}.{el}: bad marker {mk!r}")
                    if el in seen:
                        raise MetalWheelError(f"radar {r}.{el} in both {seen[el]} and {ph}")
                    seen[el] = ph
            for el, order in (sec.get("boundary") or {}).items():
                if not set(order) <= set(PHASES) or el not in seen:
                    raise MetalWheelError(f"radar {r}.boundary.{el}: bad phases {order}")

    # ------------------------------------------------------------------ cell logic
    def _wheel_entries(self, route: str, el: str) -> tuple[tuple[str, str], ...]:
        sl = self.wheel[route]
        entries = [(ring, sl[ring][el]) for ring in RINGS if el in (sl.get(ring) or {})]
        order = (sl.get("boundary") or {}).get(el)
        if order:  # the ring holding most of the symbol comes first
            entries.sort(key=lambda e: order.index(e[0]) if e[0] in order else len(order))
        return tuple(entries)

    def _radar(self, route: str, el: str) -> tuple[tuple[str, ...], str | None]:
        sec = self.radar.get(route)
        if not sec:
            return (), None
        for ph in PHASES:
            if el in (sec.get(ph) or {}):
                marker = MARKERS[sec[ph][el]]
                order = (sec.get("boundary") or {}).get(el)
                return (tuple(order) if order else (ph,)), marker
        return (), None

    def cell(self, route: str, el: str) -> Cell:
        key = (route, el)
        if key not in self._cells:
            self._cells[key] = self._resolve(route, el)
        return self._cells[key]

    def _resolve(self, route: str, el: str) -> Cell:
        if route not in self.routes:
            raise KeyError(f"unknown route {route!r}")
        w = self._wheel_entries(route, el)
        phases, marker = self._radar(route, el)
        loc_w = f"{self.source}: {self.wheel_locator}, {self.routes[route]['wheel_slice']}"
        loc_r = (
            f"{self.source}: {self.radar_locator}, {self.routes[route]['radar_sector']}"
            if phases
            else ""
        )
        locator = "; ".join(x for x in ((loc_w if w else ""), loc_r) if x)
        colours = [c for _, c in w]

        def done(fate: str, rule: str, note: str = "", secondary: tuple[str, ...] = ()) -> Cell:
            sec = tuple(dict.fromkeys(s for s in secondary if s != fate))
            return Cell(route, el, fate, sec, rule, w, phases, marker, locator, note)

        phase_fates = tuple(self._phase_fate(p, w, marker) for p in phases)

        if el in self.routes[route]["carriers"]:
            c = done(
                "recovered", "carrier", "carrier metal of the route", secondary=phase_fates[1:]
            )
            return c if c.locator else dataclasses.replace(c, locator=loc_w)
        if "G" in colours:
            ring = next(r for r, c in w if c == "G")
            return done(
                "recovered",
                "wheel-green",
                f"wheel {ring} ring, recovered at BAT",
                secondary=phase_fates,
            )
        if phases:
            return done(phase_fates[0], "radar-phase", secondary=phase_fates[1:])
        if not w:
            return done("unknown", "absent", "on neither figure for this route")
        ring_fates = [(r, self._ring_fate(r, c, route, el)) for r, c in w]
        resolved = [(r, f) for r, f in ring_fates if f != "unknown"]
        if not resolved:
            return done(
                "unknown",
                f"wheel-{w[0][0]}",
                "wheel: to dust, slime, speiss or slag; no source resolves which",
            )
        ring, fate = resolved[0]
        note = (
            ""
            if ring == w[0][0]
            else (
                f"wheel {w[0][0]} ring unresolved (dust, slime, speiss or slag); {ring} ring used"
            )
        )
        return done(fate, f"wheel-{ring}", note, secondary=tuple(f for _, f in resolved[1:]))

    def _phase_fate(self, phase: str, w: tuple[tuple[str, str], ...], marker: str | None) -> str:
        if phase in _PHASE_FATE:
            return _PHASE_FATE[phase]
        if marker == "recoverable":
            return "recovered"
        # rate by the metal ring, else the compound ring; oxide-ring colours rate the slag
        metal_cols = [c for r, c in w if r == "metal"] or [c for r, c in w if r == "compound"]
        if "R" in metal_cols:
            return "dissolved_harmful"
        if "O" in metal_cols or marker == "alloying":
            return "dissolved_benign"
        return "dissolved_unrated"

    def _ring_fate(self, ring: str, colour: str, route: str, el: str) -> str:
        if colour == "G":
            return "recovered"
        if ring == "metal":
            return "dissolved_harmful" if colour == "R" else "dissolved_benign"
        if ring == "sink":
            return "lost_to_slag"
        if route in self.pyro_routes and el in self.volatile:
            return "volatilised"
        return "unknown"

    # ------------------------------------------------------------------ views
    def cells(self) -> list[Cell]:
        return [self.cell(r, e) for e in self.elements for r in self.routes]

    def frame(self) -> pd.DataFrame:
        """Long table, one row per (element, route)."""
        rows = [
            {
                "element": c.element,
                "route": c.route,
                "fate": c.fate,
                "secondary": ", ".join(c.secondary),
                "rule": c.rule,
                "wheel": ", ".join(f"{r}:{col}" for r, col in c.wheel),
                "radar": ", ".join(c.radar) + (f" ({c.radar_marker})" if c.radar_marker else ""),
                "locator": c.locator,
                "note": c.note,
            }
            for c in self.cells()
        ]
        return pd.DataFrame(rows)

    def matrix(self, symbols: bool = False) -> pd.DataFrame:
        """Wide table: elements x routes."""
        df = self.frame().pivot(index="element", columns="route", values="fate")
        df = df.loc[self.elements, list(self.routes)]
        return df.map(SYMBOL.get) if symbols else df

    def counts(self) -> pd.DataFrame:
        return self.frame().groupby(["route", "fate"]).size().unstack(fill_value=0)

    def to_markdown(self) -> str:
        """Rendered matrix with legend (used by the reports and the CLI)."""
        m = self.matrix(symbols=True)
        head = "| Element | " + " | ".join(m.columns) + " |"
        sep = "|---|" + "---|" * len(m.columns)
        body = [f"| {el} | " + " | ".join(m.loc[el]) + " |" for el in m.index]
        legend = ", ".join(f"`{s}` {f.replace('_', ' ')}" for f, s in SYMBOL.items())
        routes = "; ".join(f"`{k}` {v['name']}" for k, v in self.routes.items())
        return "\n".join([head, sep, *body, "", f"Legend: {legend}.", "", f"Routes: {routes}."])
