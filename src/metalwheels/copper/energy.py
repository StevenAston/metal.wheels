"""Task 1.6: primary energy per kg of copper from ore, as a function of ore grade and depth.

Options (grade g as a mass fraction, depth D in m):

* ``norgate_coarse`` / ``norgate_fine`` - Norgate & Jahanshahi 2010 LCA values (beneficiation to
  75 um / 5 um plus smelting and refining, including embodied energy), fitted exactly by
  E = a + b / g, plus a depth term k_D * max(0, D - D_0) beyond the depth already embodied in the
  grade studies (Harmsen 2013: D_0 = 497 m). Source grade range 0.1-3.0 %.
* ``koppelaar_surface`` / ``koppelaar_underground`` - Koppelaar & Koppelaar 2016 Eq. (4),
  ln E = a + c D + b / g, fitted to mine-site data for MINING AND CONCENTRATING ONLY (no smelting
  or refining), so it is not like-for-like with Norgate. Source range: head grade 0.22-4.8 %,
  depth 50-1200 m.

Every evaluation outside an option's source range logs a WARNING (extrapolation).
Units: Mt x MJ/kg = PJ; annual totals are reported in EJ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metalwheels.logging_setup import get_logger
from metalwheels.registry import Registry

log = get_logger(__name__)

OPTIONS = ("norgate_coarse", "norgate_fine", "koppelaar_surface", "koppelaar_underground")


@dataclass(frozen=True)
class EnergyModel:
    option: str
    a: float
    b: float
    c: float = 0.0  # Koppelaar depth coefficient (1/m)
    k_depth: float = 0.0  # Norgate depth term (MJ/kg/m)
    d0: float = 0.0
    grade_range: tuple[float, float] = (0.0, np.inf)
    depth_range: tuple[float, float] | None = None

    @property
    def log_linear(self) -> bool:
        return self.option.startswith("koppelaar")

    @classmethod
    def from_registry(
        cls, reg: Registry, option: str = "norgate_coarse", k_depth: float | None = None
    ) -> EnergyModel:
        if option not in OPTIONS:
            raise ValueError(f"unknown energy option {option!r}; choose from {OPTIONS}")
        if option.startswith("norgate"):
            kind = option.split("_")[1]
            return cls(
                option,
                a=reg.value(f"en.norgate.{kind}.a"),
                b=reg.value(f"en.norgate.{kind}.b"),
                k_depth=reg.value("en.depth.energy_per_m") if k_depth is None else k_depth,
                d0=reg.value("en.depth.reference_depth"),
                grade_range=(reg.value("en.norgate.grade_min"), reg.value("en.norgate.grade_max")),
            )
        kind = option.split("_")[1]
        return cls(
            option,
            a=reg.value(f"en.koppelaar.{kind}.a"),
            b=reg.value(f"en.koppelaar.{kind}.b"),
            c=reg.value(f"en.koppelaar.{kind}.c"),
            grade_range=(reg.value("en.koppelaar.grade_min"), reg.value("en.koppelaar.grade_max")),
            depth_range=(reg.value("en.koppelaar.depth_min"), reg.value("en.koppelaar.depth_max")),
        )

    def _check_range(self, g: np.ndarray, d: np.ndarray) -> None:
        lo, hi = self.grade_range
        bad_g = np.isfinite(g) & ((g < lo) | (g > hi))
        if bad_g.any():
            log.warning(
                "%s evaluated outside its source grade range %.4g-%.4g for %d value(s) "
                "(min %.4g, max %.4g): extrapolation",
                self.option, lo, hi, int(bad_g.sum()), np.nanmin(g[bad_g]), np.nanmax(g[bad_g]),
            )  # fmt: skip
        if self.depth_range is not None:
            dlo, dhi = self.depth_range
            bad_d = np.isfinite(d) & ((d < dlo) | (d > dhi))
            if bad_d.any():
                log.warning(
                    "%s evaluated outside its source depth range %.0f-%.0f m for %d value(s) "
                    "(min %.0f, max %.0f): extrapolation",
                    self.option,
                    dlo,
                    dhi,
                    int(bad_d.sum()),
                    np.nanmin(d[bad_d]),
                    np.nanmax(d[bad_d]),
                )

    def intensity(self, grade: np.ndarray | float, depth: np.ndarray | float = 0.0) -> np.ndarray:
        """Primary energy (MJ/kg Cu) at ore grade ``grade`` (mass fraction), ``depth`` (m)."""
        g = np.asarray(grade, dtype=float)
        d = np.broadcast_to(np.asarray(depth, dtype=float), g.shape)
        self._check_range(g, d)
        if self.log_linear:
            return np.exp(self.a + self.c * d + self.b / g)
        return self.a + self.b / g + self.k_depth * np.maximum(0.0, d - self.d0)


def annual_energy_ej(primary_cu_mt: np.ndarray, intensity_mj_per_kg: np.ndarray) -> np.ndarray:
    """Mt Cu x MJ/kg = PJ -> EJ."""
    return np.asarray(primary_cu_mt) * np.asarray(intensity_mj_per_kg) / 1e3
