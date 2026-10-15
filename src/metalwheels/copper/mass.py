"""Task 1.7: mass outputs - ore, waste rock, tailings mass and volume.

* ore (Mt)           = copper contained in ore (Mt) / ore grade (mass fraction)
* waste rock (Mt)    = strip ratio x ore (waste-to-ore ratio; Nassar et al. 2022)
* concentrate (Mt)   = copper recovered to concentrate / concentrate grade (Harmsen 2013: 27-30 %)
* tailings (Mt)      = ore - concentrate
* tailings volume    = tailings mass / dry density (Morales et al. 2021), reported in km^3

Notes: heap-leach / SX-EW ore ends on leach pads rather than in tailings dams; it is counted as
tailings here (a simplification). Mass and volume use Mt and t/m^3, so Mt / (t/m^3) = 10^6 m^3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metalwheels.registry import Registry


def ore_mass(cu_in_ore: np.ndarray | float, grade: np.ndarray | float) -> np.ndarray:
    """Ore tonnage (same mass unit as ``cu_in_ore``) at ore grade ``grade`` (mass fraction)."""
    g = np.asarray(grade, dtype=float)
    if np.any(g <= 0):
        raise ValueError("ore grade must be positive")
    return np.asarray(cu_in_ore, dtype=float) / g


def volume_km3(mass_mt: np.ndarray | float, density_t_per_m3: float) -> np.ndarray:
    """Mt at a dry density (t/m^3) -> km^3 (1 km^3 = 10^9 m^3)."""
    if density_t_per_m3 <= 0:
        raise ValueError("density must be positive")
    return np.asarray(mass_mt, dtype=float) * 1e6 / density_t_per_m3 / 1e9


@dataclass(frozen=True)
class MassParams:
    strip_ratio: float
    concentrate_grade: float
    tailings_density: float  # t/m^3
    tailings_share: float  # share of Cu in ore lost to tailings

    @classmethod
    def from_registry(cls, reg: Registry, **overrides: float) -> MassParams:
        vals = {
            "strip_ratio": reg.value("geo.strip_ratio"),
            "concentrate_grade": reg.value("geo.concentrate_grade"),
            "tailings_density": reg.value("geo.tailings_dry_density"),
            "tailings_share": reg.value("cu.tailings_share_of_ore_cu"),
        }
        bad = set(overrides) - set(vals)
        if bad:
            raise ValueError(f"unknown overrides: {sorted(bad)}")
        vals.update(overrides)
        return cls(**vals)


def mass_flows(
    years: np.ndarray, cu_in_ore: np.ndarray, grade_pct: np.ndarray, prm: MassParams
) -> pd.DataFrame:
    """Annual and cumulative masses (Mt) and tailings volume (km^3).

    ``grade_pct`` is the mined grade in wt% (as returned by the geology module).
    """
    g = np.asarray(grade_pct, dtype=float) / 100
    cu = np.asarray(cu_in_ore, dtype=float)
    ok = np.isfinite(g) & (g > 0)
    ore = np.zeros_like(cu)
    ore[ok] = ore_mass(cu[ok], g[ok])
    conc = cu * (1 - prm.tailings_share) / prm.concentrate_grade
    tail = np.where(ore > 0, ore - conc, 0.0)
    df = pd.DataFrame(
        {
            "ore": ore,
            "waste_rock": prm.strip_ratio * ore,
            "concentrate": np.where(ore > 0, conc, 0.0),
            "tailings": tail,
            "tailings_volume_km3": volume_km3(tail, prm.tailings_density),
        },
        index=pd.Index(np.asarray(years), name="year"),
    )
    df["rock_moved"] = df["ore"] + df["waste_rock"]
    for col in ("ore", "waste_rock", "tailings", "tailings_volume_km3", "rock_moved"):
        df[f"cum_{col}"] = df[col].cumsum()
    return df
