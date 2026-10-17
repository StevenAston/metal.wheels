"""Task 4.1: ratchet catalogue (schema, validation, energy steps, interaction graph).

The catalogue lives in ``params/ratchets.yaml`` under ``ratchets:``; its numbers live under
``parameters:`` (or in other registry files) and are referenced by key. :func:`load` validates the
entries against :class:`Ratchet` and checks every reference: parameter keys exist in the registry,
bibliography keys resolve, interaction targets exist, and Metal Wheel expectations name real
routes, elements and fates. :func:`check_wheel` compares those expectations with the matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from metalwheels import references
from metalwheels import registry as _registry
from metalwheels.ratchets import metal_wheel as mw
from metalwheels.registry import Registry

Mechanism = Literal["tramp_contamination", "dissipation", "grade", "depth", "route_switch"]
InteractionKind = Literal["amplifies", "feeds", "fed_by", "parallel", "competes"]
MECHANISMS: tuple[str, ...] = Mechanism.__args__  # type: ignore[attr-defined]


class CatalogueError(ValueError):
    """Raised when the catalogue fails validation."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Tolerance(_Strict):
    param: str
    range_params: list[str] = Field(default_factory=list)


class EnergyStep(_Strict):
    formula: Literal["a", "a - b", "a * b"]
    params: dict[str, str]
    basis: str = Field(min_length=3)

    @model_validator(mode="after")
    def _check(self) -> EnergyStep:
        need = {"a"} if self.formula == "a" else {"a", "b"}
        if set(self.params) != need:
            raise ValueError(f"formula {self.formula!r} needs params {sorted(need)}")
        return self


class WheelCheck(_Strict):
    route: str
    element: str
    expect: list[str] = Field(min_length=1)

    @field_validator("expect")
    @classmethod
    def _fates(cls, v: list[str]) -> list[str]:
        bad = set(v) - set(mw.FATES)
        if bad:
            raise ValueError(f"unknown fates {sorted(bad)}")
        return v


class Interaction(_Strict):
    with_: str = Field(alias="with")
    kind: InteractionKind
    note: str = Field(min_length=5)


class Ratchet(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str = Field(min_length=5)
    mechanism: Mechanism
    host: str
    agent: str
    route: str | None
    removal: str = Field(min_length=5)
    tolerance: Tolerance | None
    tolerance_note: str | None = None
    energy_step: EnergyStep | None
    energy_note: str | None = None
    modelled_in: list[str] = Field(default_factory=list)
    source: str
    locator: str = Field(min_length=3)
    secondary_sources: list[str] = Field(default_factory=list)
    wheel_check: list[WheelCheck] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _notes(self) -> Ratchet:
        if self.energy_step is None and not self.energy_note:
            raise ValueError("energy_step is null: give energy_note saying why")
        return self

    @property
    def sourced(self) -> bool:
        return bool(self.source and self.locator)

    def param_keys(self) -> list[str]:
        keys: list[str] = []
        if self.tolerance:
            keys += [self.tolerance.param, *self.tolerance.range_params]
        if self.energy_step:
            keys += list(self.energy_step.params.values())
        return keys


class Catalogue:
    def __init__(self, entries: list[Ratchet], reg: Registry):
        self.entries = entries
        self.reg = reg
        self.by_id = {r.id: r for r in entries}

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    # ---------------------------------------------------------------- values
    def tolerance(self, rid: str) -> tuple[float, float, float] | None:
        """(value, low, high) as fractions, or None."""
        t = self.by_id[rid].tolerance
        if t is None:
            return None
        p = self.reg[t.param]
        vals = [p.value, *(self.reg.value(k) for k in t.range_params)]
        lows = [p.low if p.low is not None else p.value, *vals[1:]]
        highs = [p.high if p.high is not None else p.value, *vals[1:]]
        return p.value, min(lows), max(highs)

    def energy_step(self, rid: str) -> tuple[float, float, float, str] | None:
        """(value, low, high, unit) of the entry's energy step, by interval arithmetic."""
        es = self.by_id[rid].energy_step
        if es is None:
            return None
        pa = self.reg[es.params["a"]]

        def rng(p):
            return (
                p.value,
                p.low if p.low is not None else p.value,
                p.high if p.high is not None else p.value,
            )

        a, alo, ahi = rng(pa)
        if es.formula == "a":
            return a, alo, ahi, pa.unit
        pb = self.reg[es.params["b"]]
        b, blo, bhi = rng(pb)
        if es.formula == "a - b":
            if pa.unit != pb.unit:
                raise CatalogueError(f"{rid}: a - b with units {pa.unit} and {pb.unit}")
            return a - b, alo - bhi, ahi - blo, pa.unit
        # a * b, b dimensionless
        if pb.unit != "1":
            raise CatalogueError(f"{rid}: a * b needs a dimensionless b, got {pb.unit}")
        prods = [alo * blo, alo * bhi, ahi * blo, ahi * bhi]
        return a * b, min(prods), max(prods), pa.unit

    # ---------------------------------------------------------------- views
    def frame(self) -> pd.DataFrame:
        rows = []
        for r in self.entries:
            tol = self.tolerance(r.id)
            es = self.energy_step(r.id)
            rows.append(
                {
                    "id": r.id,
                    "mechanism": r.mechanism,
                    "host": r.host,
                    "agent": r.agent,
                    "route": r.route or "",
                    "tolerance": tol[0] if tol else float("nan"),
                    "tolerance_low": tol[1] if tol else float("nan"),
                    "tolerance_high": tol[2] if tol else float("nan"),
                    "energy_step": es[0] if es else float("nan"),
                    "energy_low": es[1] if es else float("nan"),
                    "energy_high": es[2] if es else float("nan"),
                    "energy_unit": es[3] if es else "",
                    "energy_basis": r.energy_step.basis if r.energy_step else r.energy_note,
                    "modelled": bool(r.modelled_in),
                    "source": r.source,
                    "locator": r.locator,
                }
            )
        return pd.DataFrame(rows)

    def interactions(self) -> pd.DataFrame:
        rows = [
            {"from": r.id, "to": i.with_, "kind": i.kind, "note": i.note}
            for r in self.entries
            for i in r.interactions
        ]
        return pd.DataFrame(rows, columns=["from", "to", "kind", "note"])


def _raw_entries(path: Path) -> list[dict]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = doc.get("ratchets")
    if not isinstance(raw, list) or not raw:
        raise CatalogueError(f"{path.name}: expected a non-empty 'ratchets' list")
    return raw


def load(
    path: Path | str | None = None,
    reg: Registry | None = None,
    bib: dict | None = None,
    wheel: mw.MetalWheel | None = None,
) -> Catalogue:
    """Parse, validate and cross-check the catalogue."""
    p = Path(path) if path is not None else _registry.default_params_dir() / "ratchets.yaml"
    reg = reg if reg is not None else _registry.load()
    bib = bib if bib is not None else references.load()
    wheel = wheel if wheel is not None else mw.MetalWheel.load()
    entries: list[Ratchet] = []
    for i, d in enumerate(_raw_entries(p)):
        try:
            entries.append(Ratchet(**d))
        except Exception as exc:  # pydantic.ValidationError or TypeError
            raise CatalogueError(f"ratchet #{i} ({d.get('id')!r}): {exc}") from exc
    ids = [r.id for r in entries]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise CatalogueError(f"duplicate ratchet ids {sorted(dup)}")
    problems: list[str] = []
    for r in entries:
        for k in r.param_keys():
            if k not in reg:
                problems.append(f"{r.id}: unknown parameter {k!r}")
        if r.source not in bib:
            problems.append(f"{r.id}: source {r.source!r} not in bibliography")
        for s in r.secondary_sources:
            key = s.split(" ")[0]
            if key not in bib:
                problems.append(f"{r.id}: secondary source {key!r} not in bibliography")
        if r.route is not None and r.route not in wheel.routes:
            problems.append(f"{r.id}: unknown route {r.route!r}")
        for wc in r.wheel_check:
            if wc.route not in wheel.routes or wc.element not in wheel.elements:
                problems.append(f"{r.id}: wheel_check {wc.route}/{wc.element} not in the matrix")
        for it in r.interactions:
            if it.with_ not in ids:
                problems.append(f"{r.id}: interaction with unknown ratchet {it.with_!r}")
    if problems:
        raise CatalogueError("; ".join(problems))
    return Catalogue(entries, reg)


def check_wheel(cat: Catalogue, wheel: mw.MetalWheel) -> list[str]:
    """Mismatches between each entry's expected Metal Wheel fates and the matrix."""
    out = []
    for r in cat:
        for wc in r.wheel_check:
            got = wheel.cell(wc.route, wc.element).fate
            if got not in wc.expect:
                out.append(f"{r.id}: {wc.route}/{wc.element} is {got}, expected {wc.expect}")
    return out
