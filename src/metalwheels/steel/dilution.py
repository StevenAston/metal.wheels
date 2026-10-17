"""Task 2.3: copper tolerances, dilution and scrap allocation.

Concentrations are mass fractions (0.004 = 0.4 wt%). Masses are Mt.

**Dilution share.** A melt of scrap at concentration c_s must be diluted with primary iron at c_p
to reach a product tolerance c_t. The primary share of the charge is

    x = (c_s - c_t) / (c_s - c_p),   clipped to [0, 1]  (x = 0 when c_s <= c_t).

**Allocation.** Each year, end-of-life scrap categories (mass s_i, concentration c_i) and
unlimited primary iron (c_p) are allocated to product sinks (mass m_j, tolerance tau_j). This is
the pinch analysis in Daehn et al. (2017), with the same assumption of perfectly coordinated
global trade and free blending. It is solved as a linear programme:

    minimise   sum_j p_j                                    (primary iron)
    subject to sum_i x_ij + p_j = m_j                       (each sink filled)
               sum_i c_i x_ij + c_p p_j <= tau_j m_j        (copper tolerance)
               sum_j x_ij <= s_i,   x_ij, p_j >= 0          (scrap available)

Among allocations with the least primary iron, the one putting the least copper into products
is chosen, so when some scrap cannot be used it is the dirtiest scrap that is left over. Where
freedom remains, copper is spread as evenly as tolerances allow ("mixed", default) or steered
into the most tolerant products ("pinch"); see :func:`allocate`. Scrap that is not used
is split into scrap that is surplus by mass (more scrap than total demand) and scrap that is
*unusable because of copper*. Primary iron above the mass-only requirement max(0, sum m - sum s)
is the copper-induced primary requirement that Task 2.4 charges back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

#: Relative weight of the "copper into the most tolerant products" tie-break in the LP.
TIE_BREAK = 1e-3


def dilution_share(c_scrap, c_tolerance, c_primary):
    """Primary-iron share of the charge needed to bring scrap at ``c_scrap`` to ``c_tolerance``.

    Works element-wise on arrays. Raises if the primary source is not cleaner than the
    tolerance (no amount of dilution would then suffice)."""
    c_s, c_t, c_p = (np.asarray(v, dtype=float) for v in (c_scrap, c_tolerance, c_primary))
    if np.any(c_p >= c_t):
        raise ValueError("primary concentration must be below the tolerance")
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(c_s > c_t, (c_s - c_t) / (c_s - c_p), 0.0)
    x = np.clip(x, 0.0, 1.0)
    return float(x) if x.ndim == 0 else x


@dataclass(frozen=True)
class Allocation:
    """Result of one year's allocation. Arrays follow the order of the inputs."""

    scrap_used: np.ndarray  # per source (Mt)
    scrap_unused: np.ndarray  # per source (Mt)
    primary: np.ndarray  # per sink (Mt)
    sink_copper: np.ndarray  # Cu delivered to each sink (Mt)
    flows: np.ndarray  # x_ij, sources x sinks (Mt)
    primary_concentration: float

    @property
    def primary_total(self) -> float:
        return float(self.primary.sum())

    @property
    def sink_mass(self) -> np.ndarray:
        return self.flows.sum(axis=0) + self.primary

    @property
    def sink_concentration(self) -> np.ndarray:
        return self.sink_copper / self.sink_mass

    @property
    def mass_only_primary(self) -> float:
        return max(0.0, float(self.sink_mass.sum() - (self.scrap_used + self.scrap_unused).sum()))

    @property
    def copper_induced_primary(self) -> float:
        """Primary iron required only because of copper (Mt)."""
        return self._clean(max(0.0, self.primary_total - self.mass_only_primary))

    def _clean(self, v: float) -> float:
        """Zero out solver noise (below 1e-9 of throughput)."""
        return 0.0 if v < 1e-9 * max(1.0, float(self.sink_mass.sum())) else v

    @property
    def surplus_scrap(self) -> float:
        """Scrap left over because total scrap exceeds total demand (Mt)."""
        total = (self.scrap_used + self.scrap_unused).sum()
        return self._clean(max(0.0, float(total - self.sink_mass.sum())))

    @property
    def unusable_scrap(self) -> float:
        """Scrap left over because of copper, beyond any mass surplus (Mt)."""
        return self._clean(max(0.0, float(self.scrap_unused.sum()) - self.surplus_scrap))


def allocate(
    scrap_mass: Sequence[float],
    scrap_conc: Sequence[float],
    sink_mass: Sequence[float],
    sink_tolerance: Sequence[float],
    primary_conc: float,
    mode: str = "mixed",
) -> Allocation:
    """Allocate scrap and primary iron to product sinks (see module docstring).

    ``mode`` settles how copper is spread when tolerances leave freedom:

    * ``"mixed"`` (default): copper is spread as evenly as the tolerances allow
      (water-filling: every product reaches min(tau_j, z) with the lowest common level z).
      This resembles Daehn's 2008 Sankey, where EAF steel goes to products pro rata.
    * ``"pinch"``: copper is steered into the most tolerant products first, as in Daehn's
      perfect-allocation pinch analysis. It concentrates copper in construction steel.

    Both give the same primary iron and unusable scrap; only where the copper sits differs."""
    if mode not in ("mixed", "pinch"):
        raise ValueError(f"unknown allocation mode {mode!r}")
    s = np.asarray(scrap_mass, dtype=float)
    c = np.asarray(scrap_conc, dtype=float)
    m = np.asarray(sink_mass, dtype=float)
    tau = np.asarray(sink_tolerance, dtype=float)
    if s.shape != c.shape or m.shape != tau.shape:
        raise ValueError("mass and concentration arrays must align")
    if np.any(s < 0) or np.any(m < 0) or np.any(c < 0):
        raise ValueError("masses and concentrations must be non-negative")
    if np.any(tau <= primary_conc):
        raise ValueError("every tolerance must exceed the primary concentration")
    n_i, n_j = len(s), len(m)
    nx = n_i * n_j  # x_ij flattened row-major (source-major), then p_j, then z (mixed mode)
    mixed = mode == "mixed"
    n = nx + n_j + (1 if mixed else 0)

    def xi(i, j):
        return i * n_j + j

    a_eq = np.zeros((n_j, n))
    a_ub = np.zeros((n_j + n_i, n))
    b_ub = np.zeros(n_j + n_i)
    for j in range(n_j):
        a_eq[j, nx + j] = 1.0
        a_ub[j, nx + j] = primary_conc
        b_ub[j] = tau[j] * m[j]
        for i in range(n_i):
            a_eq[j, xi(i, j)] = 1.0
            a_ub[j, xi(i, j)] = c[i]
    for i in range(n_i):
        for j in range(n_j):
            a_ub[n_j + i, xi(i, j)] = 1.0
        b_ub[n_j + i] = s[i]
    if mixed:  # sink copper <= z * m_j for every sink
        lvl = np.zeros((n_j, n))
        lvl[:, :nx] = a_ub[:n_j, :nx]
        lvl[:, nx : nx + n_j] = a_ub[:n_j, nx : nx + n_j]
        lvl[:, -1] = -m
        a_ub = np.vstack([a_ub, lvl])
        b_ub = np.append(b_ub, np.zeros(n_j))

    # One LP gives the lexicographic optimum: each Mt of primary costs 1 + c_p and each Mt of
    # scrap i costs c_i < 1. Swapping scrap for primary always raises the cost, so the solution
    # minimises primary first and, among those solutions, the copper put into products.
    # A third, much smaller term settles the remaining freedom so the allocation is unique and
    # does not flip between equally good solver vertices from year to year: in "pinch" mode a
    # TIE_BREAK weight steers copper to the most tolerant sinks; in "mixed" mode a small cost
    # on the common level z spreads it evenly. The z weight is kept below
    # m_min / c_max, so lowering z can never pay for leaving usable scrap behind.
    if np.any(c >= 1.0):
        raise ValueError("concentrations must be mass fractions below 1")
    cost = np.empty(n)
    tight = (tau.max() - tau) / tau.max() if (tau.max() > 0 and not mixed) else np.zeros(n_j)
    for i in range(n_i):
        cost[i * n_j : (i + 1) * n_j] = c[i] * (1.0 + TIE_BREAK * tight)
    cost[nx : nx + n_j] = 1.0 + primary_conc
    if mixed:
        m_pos = m[m > 0]
        c_top = max(float(c.max()) if c.size else 0.0, primary_conc, 1e-9)
        cost[-1] = 1e-3 * (float(m_pos.min()) if m_pos.size else 1.0) / c_top
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=m, method="highs")
    if not res.success:
        raise RuntimeError(f"allocation LP failed: {res.message}")
    sol = res.x

    flows = np.clip(sol[:nx].reshape(n_i, n_j), 0.0, None)
    primary = np.clip(sol[nx : nx + n_j], 0.0, None)
    used = flows.sum(axis=1)
    unused = np.clip(s - used, 0.0, None)
    sink_cu = c @ flows + primary_conc * primary
    return Allocation(used, unused, primary, sink_cu, flows, primary_conc)


def copper_balance(scrap_mass, scrap_conc, sink_mass, sink_tolerance, primary_conc=None):
    """Daehn et al. (2017) Fig. 2 aggregate test: ``(copper in scrap supply, copper tolerable)``
    (Mt). Copper in supply counts scrap only, as in Daehn (0.7 Mt in 2008 against 1.8 Mt
    tolerable, p. 6602); tolerable is sum_j tau_j m_j. The aggregate crossover is the first year
    supply exceeds tolerable. ``primary_conc`` is accepted for symmetry with :func:`allocate`
    and ignored. The LP in :func:`allocate` is the stricter test, since primary iron also uses
    up some tolerance."""
    s, c = np.asarray(scrap_mass, float), np.asarray(scrap_conc, float)
    m, tau = np.asarray(sink_mass, float), np.asarray(sink_tolerance, float)
    return float(s @ c), float(m @ tau)
