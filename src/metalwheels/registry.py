"""Parameter registry with provenance.

Every number the model uses lives in a YAML file under ``params/`` together with
where it came from. A parameter file looks like::

    domain: energy
    parameters:
      norgate.coarse.a:
        value: 13.0
        unit: MJ/kg
        status: derived
        source: norgate_jahanshahi_2010
        locator: "Values tabulated in Koppelaar & Koppelaar 2016, Table 1"
        note: "..."

Rules enforced on load (Task 0.2 done condition):

* ``unit`` is required and must be one of :data:`ALLOWED_UNITS` (SI or SI-accepted;
  dimensionless quantities use ``"1"`` and are expressed as fractions, never percent).
* ``source`` (a bibliography key, resolved in Task 0.3) is required unless
  ``status: assumption``. ``status: sourced`` also requires a ``locator``.
* ``low <= value <= high`` whenever bounds are given.
* A non-``fixed`` distribution requires both ``low`` and ``high``.

Distribution semantics (used by the Monte Carlo, Task 1.9):

* ``fixed``      - always ``value``.
* ``uniform``    - U(low, high).
* ``triangular`` - mode ``value``, support [low, high].
* ``normal``     - mean ``value``; [low, high] is the central 95 % interval; samples are
  clipped to [low, high].
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from metalwheels.logging_setup import get_logger

log = get_logger(__name__)

#: Units accepted in the registry. Extend deliberately; a typo should fail loudly.
ALLOWED_UNITS: frozenset[str] = frozenset(
    {
        "1",  # dimensionless (fractions, ratios, log-space coefficients)
        "a",  # years
        "m",
        "km",
        "1/m",
        "kg",
        "t",
        "kt",
        "Mt",
        "Gt",
        "t/m3",
        "kg/person",
        "MJ/kg",
        "MJ/kg/m",
        "EJ",
    }
)

Status = Literal["sourced", "assumption", "derived"]
Distribution = Literal["fixed", "uniform", "triangular", "normal"]


class RegistryError(ValueError):
    """Raised when a parameter file is malformed or a parameter fails validation."""


class Param(BaseModel):
    """One parameter and its provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    domain: str
    value: float
    unit: str = Field(min_length=1)
    status: Status
    low: float | None = None
    high: float | None = None
    distribution: Distribution = "fixed"
    source: str | None = None
    locator: str | None = None
    range_source: str | None = None  # bibliography key for the bounds, if different
    note: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Param:
        if self.unit not in ALLOWED_UNITS:
            raise ValueError(
                f"unit {self.unit!r} is not in ALLOWED_UNITS ({', '.join(sorted(ALLOWED_UNITS))})"
            )
        if self.status != "assumption" and not self.source:
            raise ValueError(f"status {self.status!r} requires a source (bibliography key)")
        if self.status == "sourced" and not self.locator:
            raise ValueError("status 'sourced' requires a locator (table/page/figure)")
        if self.low is not None and self.value < self.low:
            raise ValueError(f"value {self.value} < low {self.low}")
        if self.high is not None and self.value > self.high:
            raise ValueError(f"value {self.value} > high {self.high}")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low {self.low} > high {self.high}")
        if self.distribution != "fixed" and (self.low is None or self.high is None):
            raise ValueError(f"distribution {self.distribution!r} requires both low and high")
        return self


class Registry(Mapping[str, Param]):
    """Read-only mapping of parameter key -> :class:`Param`."""

    def __init__(self, params: Mapping[str, Param]):
        self._params = dict(params)

    def __getitem__(self, key: str) -> Param:
        try:
            return self._params[key]
        except KeyError:
            raise KeyError(f"unknown parameter {key!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def value(self, key: str) -> float:
        """Shortcut for ``registry[key].value``."""
        return self[key].value

    def domain(self, name: str) -> dict[str, Param]:
        return {k: p for k, p in self._params.items() if p.domain == name}

    @property
    def domains(self) -> list[str]:
        return sorted({p.domain for p in self._params.values()})

    def with_overrides(self, overrides: Mapping[str, float], label: str = "override") -> Registry:
        """Copy with some values replaced (e.g. by a scenario or a Monte Carlo draw).

        Overridden parameters become ``fixed`` with no bounds, because scenario values are
        allowed to lie outside a parameter's sourced range (that is often the point).
        Unknown keys raise ``KeyError``.
        """
        unknown = set(overrides) - set(self._params)
        if unknown:
            raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
        new = dict(self._params)
        for key, value in overrides.items():
            old = self._params[key]
            new[key] = old.model_copy(
                update={
                    "value": float(value),
                    "low": None,
                    "high": None,
                    "distribution": "fixed",
                    "note": f"[{label}: {value!r}; was {old.value!r}] {old.note or ''}".strip(),
                }
            )
        return Registry(new)

    def sources(self) -> set[str]:
        """Every bibliography key referenced (``source`` and ``range_source``)."""
        keys: set[str] = set()
        for p in self._params.values():
            keys.update(k for k in (p.source, p.range_source) if k)
        return keys


def default_params_dir() -> Path:
    """``$METALWHEELS_PARAMS`` if set, else the repository's ``params/`` directory."""
    env = os.environ.get("METALWHEELS_PARAMS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "params"


def parse_file(path: Path) -> dict[str, Param]:
    """Parse and validate one YAML parameter file."""
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(doc, dict) or "domain" not in doc or "parameters" not in doc:
        raise RegistryError(f"{path.name}: expected top-level keys 'domain' and 'parameters'")
    domain = doc["domain"]
    raw = doc["parameters"] or {}
    if not isinstance(raw, dict):
        raise RegistryError(f"{path.name}: 'parameters' must be a mapping")
    out: dict[str, Param] = {}
    for key, fields in raw.items():
        if not isinstance(fields, dict):
            raise RegistryError(f"{path.name}: {key}: expected a mapping of fields")
        try:
            out[key] = Param(key=key, domain=domain, **fields)
        except (ValidationError, TypeError) as exc:
            raise RegistryError(f"{path.name}: {key}: {exc}") from exc
    return out


def load(params_dir: Path | str | None = None) -> Registry:
    """Load and validate every ``*.yaml`` file in ``params_dir``."""
    directory = Path(params_dir) if params_dir is not None else default_params_dir()
    if not directory.is_dir():
        raise RegistryError(f"parameter directory not found: {directory}")
    params: dict[str, Param] = {}
    origin: dict[str, str] = {}
    files = sorted(directory.glob("*.yaml"))
    for path in files:
        for key, param in parse_file(path).items():
            if key in params:
                raise RegistryError(f"duplicate key {key!r} in {path.name} and {origin[key]}")
            params[key] = param
            origin[key] = path.name
    log.debug("loaded %d parameters from %d files in %s", len(params), len(files), directory)
    return Registry(params)
