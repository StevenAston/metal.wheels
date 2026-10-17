"""Task 3.2-3.4: Leontief inverse, spectral radius, maintenance energy and sensitivities.

For a physical input-output matrix A (inputs of sector i per unit output of sector j) and a final
demand vector y, gross output is x = (I - A)^-1 y = L y. The series sum_k A^k converges, and
every element of L is finite and non-negative, only when the spectral radius rho(A) < 1. rho
measures how much each unit of final demand echoes back through the system: as rho -> 1 the
system needs ever more gross output to deliver the same final demand.

Sensitivities (Task 3.4). For a non-negative irreducible A, rho is the Perron eigenvalue with
right and left eigenvectors u and v, and

    d rho / d a_ij = v_i u_j / (v . u),        elasticity = (a_ij / rho) d rho / d a_ij.

Maintenance energy E = e_E^T L y (energy output needed to deliver y), with dL/da_ij = L e_i e_j^T L:

    d E / d a_ij = (e_E^T L)_i (L y)_j,        elasticity = (a_ij / E) dE / d a_ij.
"""

from __future__ import annotations

import numpy as np

from metalwheels.logging_setup import get_logger

log = get_logger(__name__)

RHO_WARNING = 0.9


def leontief_inverse(a: np.ndarray) -> np.ndarray:
    """(I - A)^-1 for one matrix (n, n) or a stack (T, n, n)."""
    a = np.asarray(a, dtype=float)
    eye = np.eye(a.shape[-1])
    return np.linalg.inv(eye - a)


def spectral_radius(a: np.ndarray, label: str = "") -> np.ndarray | float:
    """max |eigenvalue| of A (or of each matrix in a stack). Logs a WARNING when rho >= 0.9 and
    an ERROR when rho >= 1 (the Leontief series diverges)."""
    a = np.asarray(a, dtype=float)
    rho = np.max(np.abs(np.linalg.eigvals(a)), axis=-1)
    rmax = float(np.max(rho))
    where = f" ({label})" if label else ""
    if rmax >= 1.0:
        log.error("spectral radius %.4f >= 1%s: the cascade does not converge", rmax, where)
    elif rmax >= RHO_WARNING:
        log.warning(
            "spectral radius %.4f >= %.1f%s: cascade near divergence", rmax, RHO_WARNING, where
        )
    return float(rho) if np.ndim(rho) == 0 else rho


def perron_vectors(a: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(rho, right eigenvector u, left eigenvector v) for the eigenvalue of largest modulus."""
    a = np.asarray(a, dtype=float)
    w, vr = np.linalg.eig(a)
    k = int(np.argmax(np.abs(w)))
    wl, vl = np.linalg.eig(a.T)
    kl = int(np.argmin(np.abs(wl - w[k])))
    u = np.real(vr[:, k])
    v = np.real(vl[:, kl])
    u = u * np.sign(u.sum()) if u.sum() != 0 else u
    v = v * np.sign(v.sum()) if v.sum() != 0 else v
    return float(np.real(w[k])), u, v


def rho_gradient(a: np.ndarray) -> np.ndarray:
    """Analytic d rho / d a_ij (matrix)."""
    _, u, v = perron_vectors(a)
    return np.outer(v, u) / float(v @ u)


def rho_elasticities(a: np.ndarray) -> np.ndarray:
    rho, _, _ = perron_vectors(a)
    return np.asarray(a, dtype=float) * rho_gradient(a) / rho


def maintenance_energy(a: np.ndarray, y: np.ndarray, energy_index: int) -> np.ndarray | float:
    """Energy-sector gross output needed to deliver final demand ``y`` (one matrix or a stack)."""
    x = np.einsum("...ij,j->...i", leontief_inverse(a), np.asarray(y, dtype=float))
    out = x[..., energy_index]
    return float(out) if np.ndim(out) == 0 else out


def energy_gradient(a: np.ndarray, y: np.ndarray, energy_index: int) -> np.ndarray:
    """Analytic dE / d a_ij for E = e_E^T (I - A)^-1 y."""
    lmat = leontief_inverse(a)
    return np.outer(lmat[energy_index, :], lmat @ np.asarray(y, dtype=float))


def energy_elasticities(a: np.ndarray, y: np.ndarray, energy_index: int) -> np.ndarray:
    e = maintenance_energy(a, y, energy_index)
    return np.asarray(a, dtype=float) * energy_gradient(a, y, energy_index) / e


def finite_difference_elasticity(f, a: np.ndarray, i: int, j: int, rel: float = 1e-6) -> float:
    """Central-difference elasticity of scalar f(A) with respect to a_ij."""
    a = np.asarray(a, dtype=float)
    h = rel * a[i, j]
    if h == 0:
        return 0.0
    up, dn = a.copy(), a.copy()
    up[i, j] += h
    dn[i, j] -= h
    return (f(up) - f(dn)) / (2 * h) * a[i, j] / f(a)
