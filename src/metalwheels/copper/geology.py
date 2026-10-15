"""Task 1.5: geology - lognormal grade-tonnage within depth slices, discovery and extraction order.

Grade-tonnage (Gerst 2008). Ore tonnage is lognormally distributed in ln(grade, wt%) with
parameters mu, sigma^2. For copper endowment E (Mt Cu):

* mean grade (wt%)          m = exp(mu + sigma^2 / 2);  total ore T = E / (m / 100)
* ore above cutoff g        T_>(g) = T * (1 - Phi((ln g - mu) / sigma))
* copper above cutoff g     C_>(g) = E * (1 - Phi((ln g - mu - sigma^2) / sigma))
* richest-first cutoff after removing X:  ln g = mu + sigma^2 + sigma * Phi^-1(1 - X / E)

Depth slices. Each endowment bracket is cut into slices (default 100 m). Every slice holds
*discovered* copper (Harmsen 2013 Table 1 "analysed") and *undiscovered* copper; both share the
bracket's grade distribution. Within the first km, discovered copper is weighted towards the
surface (Harmsen: 85 % of depth-to-top-of-ore within 500 m, from Long 2011); deeper layers are
uniform within each layer.

Extraction order. Each slice i has an ordering cost offset P_i = m * k_D * max(0, D_i - D_0),
where k_D is the depth-energy term (Harmsen 2013) and m a multiplier calibrated so that the share
of 1900-2010 extraction from the top 500 m matches Harmsen's statement (0.85 x 526 / 580 Mt).
At cost level lam each slice is worked down to grade g_i = 100 b / (lam - P_i), with b the Norgate
grade coefficient, so extraction always takes the cheapest remaining copper across slices.

Paced discovery (decided by Steve, 2026-10-15). Up to the calibration year only discovered
copper is minable (everything produced by 2010 came from deposits known by 2010). Afterwards, a
uniform fraction phi of every slice's undiscovered copper becomes discovered whenever needed to
keep discovered-but-unmined copper at least ``lead_years`` x the demand trend (trailing mean over
``smoothing_years``, so single-year demand steps do not trigger discovery pulses);
``lead_years`` is the 2010 ratio unless given. Discovery in any year is also capped at
``max_discovery_rate`` x the copper still undiscovered (diminishing discovery, decided by Steve
2026-10-15): the last deposits are the hardest to find, so discovered reserves run down
gradually instead of discovery stopping abruptly when everything has been found.

The state in year t is fully described by (lam_t, phi_t): cumulative extraction
X_t = sum_i (A_i + phi_t U_i) C_i(g_i(lam_t)) / E_i, solved for lam_t each year.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from metalwheels.logging_setup import get_logger
from metalwheels.registry import Registry

log = get_logger(__name__)

G_MAX, G_MIN = 1e3, 1e-9  # cutoff bounds, wt%
HARMSEN_LAYERS = (("0_1km", 0, 1000), ("1_2km", 1000, 2000), ("2_3km", 2000, 3000),
                  ("3_4km", 3000, 4000), ("4_5km", 4000, 5000))  # fmt: skip
#: Copper fully extracted from a slice/layer once less than this share remains.
EXHAUSTED = 1e-3


@dataclass(frozen=True)
class GradeTonnage:
    """Lognormal grade-tonnage pool (grades in wt%)."""

    mu: float
    sigma2: float
    endowment: float  # Mt Cu

    @property
    def sigma(self) -> float:
        return math.sqrt(self.sigma2)

    @property
    def median_grade(self) -> float:
        return math.exp(self.mu)

    @property
    def mean_grade(self) -> float:
        return math.exp(self.mu + self.sigma2 / 2)

    @property
    def total_ore(self) -> float:
        return self.endowment / (self.mean_grade / 100)

    def copper_above(self, g: np.ndarray | float) -> np.ndarray:
        g = np.clip(np.asarray(g, dtype=float), G_MIN, G_MAX)
        return self.endowment * (1 - ndtr((np.log(g) - self.mu - self.sigma2) / self.sigma))

    def ore_above(self, g: np.ndarray | float) -> np.ndarray:
        g = np.clip(np.asarray(g, dtype=float), G_MIN, G_MAX)
        return self.total_ore * (1 - ndtr((np.log(g) - self.mu) / self.sigma))

    def cutoff_for_extracted(self, x: float) -> float:
        """Cutoff grade (wt%) once copper ``x`` (Mt) has been taken richest-first."""
        if not 0 <= x <= self.endowment:
            raise ValueError(f"extracted {x} outside [0, {self.endowment}]")
        if x == 0:
            return G_MAX
        if x == self.endowment:
            return G_MIN
        return math.exp(self.mu + self.sigma2 + self.sigma * ndtri(1 - x / self.endowment))


@dataclass(frozen=True)
class DepthProfile:
    """Discovered (A) and undiscovered (U) copper by depth slice, plus grade distribution."""

    name: str
    mids: np.ndarray  # slice mid-depth (m)
    layer: np.ndarray  # layer label per slice
    discovered: np.ndarray  # Mt Cu
    undiscovered: np.ndarray  # Mt Cu
    mu: float
    sigma2: float

    @property
    def endowment(self) -> float:
        return float(self.discovered.sum() + self.undiscovered.sum())

    def layer_totals(self) -> pd.Series:
        return (
            pd.Series(self.discovered + self.undiscovered, index=self.layer)
            .groupby(level=0, sort=False)
            .sum()
        )

    @classmethod
    def from_registry(cls, reg: Registry, bracket: str, thickness: float = 100.0) -> DepthProfile:
        """``bracket`` is "harmsen" (Table 1, 0-5 km) or "kesler" (Harmsen's depth profile scaled
        to Kesler & Wilkinson's 89 Bt above 3.3 km, truncated at 3.3 km)."""
        top_share = reg.value("geo.discovered_top500_share_first_km")
        mids, layer, disc, undisc = [], [], [], []
        for key, top, bot in HARMSEN_LAYERS:
            orig = reg.value(f"geo.endow.harmsen.layer_{key}")
            anal = reg.value(f"geo.endow.harmsen.analysed_{key}")
            n = round((bot - top) / thickness)
            for k in range(n):
                mid = top + (k + 0.5) * thickness
                if top == 0:
                    shallow = mid < 500
                    w = (top_share if shallow else 1 - top_share) / (n / 2)
                else:
                    w = 1 / n
                mids.append(mid)
                layer.append(key)
                disc.append(anal * w)
                undisc.append((orig - anal) / n)
        mids, disc, undisc = np.array(mids), np.array(disc), np.array(undisc)
        layer = np.array(layer)
        if bracket == "harmsen":
            mu, s2 = reg.value("geo.grade.gerst.mu_w"), reg.value("geo.grade.gerst.sigma2_w")
        elif bracket == "kesler":
            limit = reg.value("geo.endow.kesler.depth_limit")
            keep = mids < limit
            mids, layer, disc, undisc = mids[keep], layer[keep], disc[keep], undisc[keep]
            target = reg.value("geo.endow.kesler.above_3_3km")
            scale = (target - disc.sum()) / undisc.sum()
            undisc = undisc * scale
            mu, s2 = (
                reg.value("geo.grade.kesler_ext.mu_w"),
                reg.value("geo.grade.kesler_ext.sigma2_w"),
            )
        else:
            raise ValueError(f"unknown bracket {bracket!r}")
        return cls(bracket, mids, layer, disc, undisc, mu, s2)


@dataclass
class ExtractionResult:
    profile: str
    years: np.ndarray
    grade: np.ndarray  # mined grade of the year's slice (wt%)
    ore: np.ndarray  # ore mined (Mt)
    depth: np.ndarray  # copper-weighted mean depth of the year's extraction (m)
    discovered_fraction: np.ndarray  # phi
    extracted: np.ndarray  # copper actually extracted (Mt); < demand once exhausted
    shortfall: np.ndarray  # annual demand not met because the endowment is exhausted (Mt)
    remaining_by_layer: pd.DataFrame  # Mt Cu, index year
    multiplier: float
    lead_years: float

    @property
    def first_shortfall_year(self) -> int | None:
        idx = np.flatnonzero(self.shortfall > 1e-9)
        return int(self.years[idx[0]]) if idx.size else None

    def layer_exhaustion_years(self, totals: pd.Series) -> dict[str, int | None]:
        out = {}
        for layer, total in totals.items():
            frac = self.remaining_by_layer[layer] / total
            hit = frac.index[frac < EXHAUSTED]
            out[layer] = int(hit[0]) if len(hit) else None
        return out


class Extraction:
    """Marginal-cost-ordered extraction from a depth profile with paced discovery."""

    def __init__(
        self,
        profile: DepthProfile,
        b: float,
        k_depth: float,
        d0: float,
        multiplier: float,
        calibration_year: int = 2010,
        lead_years: float | None = None,
        smoothing_years: int = 1,
        max_discovery_rate: float = 1.0,
    ):
        self.p = profile
        self.b = b
        self.offset = multiplier * k_depth * np.maximum(0.0, profile.mids - d0)
        self.multiplier = multiplier
        self.calibration_year = calibration_year
        self.lead_years = lead_years
        if smoothing_years < 1:
            raise ValueError("smoothing_years must be >= 1")
        self.smoothing_years = int(smoothing_years)
        if not 0 < max_discovery_rate <= 1:
            raise ValueError("max_discovery_rate must be in (0, 1]")
        self.max_discovery_rate = float(max_discovery_rate)
        self.sigma = math.sqrt(profile.sigma2)
        self.mean_grade = math.exp(profile.mu + profile.sigma2 / 2)

    def _cut(self, lam: float) -> np.ndarray:
        slack = lam - self.offset
        with np.errstate(divide="ignore"):
            g = np.where(slack > 0, 100 * self.b / np.maximum(slack, 1e-300), G_MAX)
        return np.clip(g, G_MIN, G_MAX)

    def _cu_frac(self, g: np.ndarray) -> np.ndarray:
        return 1 - ndtr((np.log(g) - self.p.mu - self.p.sigma2) / self.sigma)

    def _ore_frac(self, g: np.ndarray) -> np.ndarray:
        return 1 - ndtr((np.log(g) - self.p.mu) / self.sigma)

    def _solve(self, avail: np.ndarray, x: float) -> np.ndarray:
        """Cutoff grades per slice at which cumulative extraction from ``avail`` equals x."""
        if x <= 0:
            return np.full(len(avail), G_MAX)
        lo = float(self.offset.min())
        hi = lo + 1.0
        while (avail * self._cu_frac(self._cut(hi))).sum() < x:
            hi = lo + (hi - lo) * 4
            if hi > 1e15:
                raise RuntimeError("cost level search diverged")
        lam = brentq(
            lambda lv: (avail * self._cu_frac(self._cut(lv))).sum() - x,
            lo,
            hi,
            xtol=1e-12,
            rtol=1e-13,
        )
        return self._cut(lam)

    def run(self, years: np.ndarray, demand: np.ndarray) -> ExtractionResult:
        """Replay annual copper-in-ore demand (Mt/yr) from a pristine state."""
        years = np.asarray(years)
        demand = np.asarray(demand, dtype=float)
        n, A, U = len(years), self.p.discovered, self.p.undiscovered
        grade, ore, depth = (np.full(n, np.nan) for _ in range(3))
        phi_out, extracted, shortfall = np.zeros(n), np.zeros(n), np.zeros(n)
        remaining = np.zeros((n, len(A)))
        prev_cu, prev_ore = np.zeros(len(A)), np.zeros(len(A))
        phi, cum, lead = 0.0, 0.0, self.lead_years
        cap = 1 - EXHAUSTED * 1e-3  # never solve exactly at the endowment

        # discovery responds to the trailing-mean demand trend, not single-year swings
        k = self.smoothing_years
        csum = np.concatenate([[0.0], np.cumsum(demand)])
        lo_idx = np.maximum(0, np.arange(n) - k + 1)
        trend = (csum[1:] - csum[lo_idx]) / (np.arange(n) + 1 - lo_idx)

        for t in range(n):
            want = cum + demand[t]
            if years[t] > self.calibration_year:
                if lead is None:
                    raise RuntimeError("lead_years not set: calibration year not in the run")
                need = lead * trend[t]
                if (A + phi * U).sum() - want < need and phi < 1:
                    wanted = min(1.0, (need + want - A.sum()) / U.sum())
                    # diminishing discovery: at most a fixed share of what remains undiscovered
                    phi = min(wanted, phi + self.max_discovery_rate * (1 - phi))
            avail = A + phi * U
            x = min(want, avail.sum() * cap)
            shortfall[t] = want - x
            cut = self._solve(avail, x)
            cu = avail * self._cu_frac(cut)
            o = avail / (self.mean_grade / 100) * self._ore_frac(cut)
            d_cu, d_ore = cu - prev_cu, o - prev_ore
            extracted[t] = x - cum
            if d_ore.sum() > 0 and d_cu.sum() > 0:
                grade[t] = 100 * d_cu.sum() / d_ore.sum()
                depth[t] = (d_cu * self.p.mids).sum() / d_cu.sum()
            ore[t] = d_ore.sum()
            remaining[t] = A + U - cu
            phi_out[t] = phi
            prev_cu, prev_ore, cum = cu, o, x
            if years[t] == self.calibration_year and lead is None:
                lead = (avail.sum() - cum) / trend[t]

        rem = pd.DataFrame(remaining, index=pd.Index(years, name="year"), columns=self.p.layer)
        rem = rem.T.groupby(level=0, sort=False).sum().T
        return ExtractionResult(
            self.p.name, years, grade, ore, depth, phi_out, extracted, shortfall, rem,
            self.multiplier, float(lead) if lead is not None else float("nan"),
        )  # fmt: skip


def calibrate_multiplier(
    profile: DepthProfile,
    b: float,
    k_depth: float,
    d0: float,
    years: np.ndarray,
    demand: np.ndarray,
    target_share: float,
    year: int = 2010,
    shallow: float = 500.0,
) -> float:
    """Multiplier m such that the share of extraction up to ``year`` from above ``shallow`` m
    equals ``target_share`` (discovered copper only, as in the replay)."""
    cum = float(np.asarray(demand)[np.asarray(years) <= year].sum())

    def share(m: float) -> float:
        ex = Extraction(profile, b, k_depth, d0, m)
        cut = ex._solve(profile.discovered, cum)
        cu = profile.discovered * ex._cu_frac(cut)
        return cu[profile.mids < shallow].sum() / cu.sum()

    lo, hi = 0.05, 200.0
    if not share(lo) <= target_share <= share(hi):
        raise ValueError(
            f"target share {target_share:.3f} outside reachable [{share(lo):.3f}, {share(hi):.3f}]"
        )
    return brentq(lambda m: share(m) - target_share, lo, hi, xtol=1e-6)


def extraction_for(
    reg: Registry, bracket: str, years: np.ndarray, ore_cu: np.ndarray, thickness: float = 100.0
) -> tuple[DepthProfile, ExtractionResult]:
    """Build a bracket's profile, calibrate the depth multiplier on history, run extraction."""
    prof = DepthProfile.from_registry(reg, bracket, thickness)
    b = reg.value("en.norgate.coarse.b")
    k_depth = reg.value("en.depth.energy_per_m")
    d0 = reg.value("en.depth.reference_depth")
    m = calibrate_multiplier(
        prof, b, k_depth, d0, years, ore_cu, reg.value("geo.calib.produced_share_top500_2010")
    )
    log.info("%s: depth-order multiplier calibrated to %.3f", bracket, m)
    ex = Extraction(
        prof, b, k_depth, d0, m,
        smoothing_years=int(reg.value("geo.discovery_smoothing_years")),
        max_discovery_rate=reg.value("geo.max_discovery_rate"),
    )  # fmt: skip
    return prof, ex.run(years, ore_cu)
