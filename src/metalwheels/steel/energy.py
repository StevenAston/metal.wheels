"""Task 2.4: steel route energies and the copper-induced energy penalty.

Steel made from primary iron uses E_primary per kg; steel made from scrap uses E_scrap. E_primary
mixes BF-BOF and DRI-EAF by the DRI share. Each tonne of primary iron that copper forces into the
charge, because scrap that would otherwise be melted is too contaminated, costs
(E_primary - E_scrap). Over a year this gives

    penalty (EJ) = copper-induced primary iron (Mt) x (E_primary - E_scrap) (MJ/kg) / 1e3,

which is the Phase 2 energy charged back to copper. With full processing of end-of-life copper
(p = 1) no copper reaches the steel pool through the Task 2.2 link, so the penalty is zero.

**Scrap vs ore, reconciled.** On a primary-energy basis (electricity counted with 67 % generation
and grid losses) Worrell et al. 2008 give scrap-EAF 8.0 and BF-BOF 20.6 MJ/kg, so scrap steel
needs 61 % less energy (63 % on the thin-slab variant). That matches the IEA (2021) statement of
"60-70 % less". The IEA (2020) roadmap's "roughly one-eighth" does not follow from the same table
on either basis: final energy gives 4.3 / 18.0 = 0.24 (2.6 / 14.8 = 0.18 thin slab). It is only
reachable on a final-energy basis that compares the electric melting step alone with the whole
ore-based chain. That reading is an inference; the roadmap does not state its basis. The model
uses the primary-energy figures throughout, consistent with the copper energy module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metalwheels.registry import Registry


@dataclass(frozen=True)
class SteelEnergy:
    bf_bof: float  # MJ/kg
    dri_eaf: float
    scrap_eaf: float
    dri_share: float

    @classmethod
    def from_registry(cls, reg: Registry) -> SteelEnergy:
        return cls(
            bf_bof=reg.value("st.energy.bf_bof"),
            dri_eaf=reg.value("st.energy.dri_eaf"),
            scrap_eaf=reg.value("st.energy.scrap_eaf"),
            dri_share=reg.value("st.dri_share_of_primary"),
        )

    @property
    def primary(self) -> float:
        return (1 - self.dri_share) * self.bf_bof + self.dri_share * self.dri_eaf

    @property
    def penalty_per_kg(self) -> float:
        """Extra MJ per kg of steel made from primary iron instead of scrap."""
        return self.primary - self.scrap_eaf


def ej(mass_mt, mj_per_kg):
    """Mt x MJ/kg -> EJ."""
    return np.asarray(mass_mt, dtype=float) * mj_per_kg / 1e3


def penalty_ej(copper_induced_primary_mt, energy: SteelEnergy):
    """Annual copper-induced steel energy penalty (EJ/yr)."""
    return ej(copper_induced_primary_mt, energy.penalty_per_kg)


def steel_energy_ej(primary_iron_mt, scrap_used_mt, energy: SteelEnergy):
    """Annual steelmaking primary energy (EJ/yr) by route mix."""
    return ej(primary_iron_mt, energy.primary) + ej(scrap_used_mt, energy.scrap_eaf)
