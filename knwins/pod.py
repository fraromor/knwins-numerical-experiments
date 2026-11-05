r"""POD with an SVD weighted by the Gram matrix of the error norm.

The error norm is :math:`H^1` for the velocity and :math:`L^2` for the
pressure; the module also fits the decay of the errors.

Snapshots :math:`U = [u_1, \dots, u_N]` (dof vectors as columns), Gram matrix
:math:`H = R^T R`. The SVD :math:`R U = Q S Z^T` gives the
:math:`H`-orthonormal POD basis :math:`\Phi = R^{-1} Q`, and with the
:math:`H`-orthogonal projector :math:`\Pi_r` onto the first :math:`r` modes

.. math::

    \|u_j - \Pi_r u_j\|_H^2 = \sum_{k \ge r} (S_k Z_{jk})^2.

The SVD of :math:`R U` avoids squaring the condition number (method of
snapshots).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def gram_factor(H: sp.spmatrix) -> sp.csr_matrix:
    r"""Return the sparse :math:`R` with :math:`R^T R = H`.

    :math:`H` is symmetric positive definite. :math:`R` comes from the
    :math:`LDL^T` factorization given by a symmetric sparse LU without
    pivoting: :math:`P H P^T = L U`, :math:`U = D L^T`,
    :math:`R = D^{-1/2} U P`.

    Args:
        H: symmetric positive definite Gram matrix.
    """
    lu = splu(sp.csc_matrix(H), permc_spec="MMD_AT_PLUS_A",
              diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    assert np.all(lu.perm_r == lu.perm_c), "LU used non-symmetric pivoting"
    n = H.shape[0]
    P = sp.csr_matrix((np.ones(n), (lu.perm_r, np.arange(n))), shape=(n, n))
    return (sp.diags(1 / np.sqrt(lu.U.diagonal())) @ lu.U @ P).tocsr()


def max_projection_errors(U: np.ndarray, H: sp.spmatrix) -> np.ndarray:
    r"""Return the maximum POD projection errors.

    :math:`\varepsilon(r) = \max_j \|u_j - \Pi_r u_j\|_H` for
    :math:`r = 1, \dots, N-1`.

    Args:
        U: snapshots as columns, shape ``(ndofs, N)``.
        H: Gram matrix of the error norm.

    Returns:
        The errors, shape ``(N - 1,)``.
    """
    _, S, Zt = np.linalg.svd(gram_factor(H) @ U, full_matrices=False)
    squared = (S[:, None] * Zt)**2  # row k: contributions of mode k
    tails = np.cumsum(squared[::-1], axis=0)[::-1]  # row r: sum_{k >= r}
    return np.sqrt(tails[1:U.shape[1]]).max(axis=1)


def above_roundoff(errors: np.ndarray, relative: float = 2e-14,
                   absolute: float = 1e-16) -> np.ndarray:
    r"""Return the errors before the first one below the round-off level.

    Args:
        errors: :math:`\varepsilon(n)` for :math:`n = 1, 2, \dots`.
        relative: round-off level relative to :math:`\varepsilon(1)`.
        absolute: absolute round-off level.
    """
    below = errors < max(relative * errors[0], absolute)
    return errors[:np.argmax(below)] if below.any() else errors


def fit_decay(errors: np.ndarray, exponent: float,
              window: Tuple[float, float]) -> Tuple[float, float, float]:
    r"""Fit :math:`\varepsilon(n) \approx \exp(-b n^a - c)`.

    Least-squares fit of the logarithm of the errors.

    Args:
        errors: :math:`\varepsilon(n)` for :math:`n = 1, 2, \dots`.
        exponent: :math:`a`, e.g. 1/3 or 1/2.
        window: ``(n_min, n_max)``, the fit uses ``n_min <= n <= n_max``.

    Returns:
        ``(b, c, rms)``, with ``rms`` the root mean square of the residual of
        :math:`\log \varepsilon(n)`.
    """
    n = np.arange(1, len(errors) + 1)
    m = (n >= window[0]) & (n <= window[1])
    A = np.vstack([-n[m]**exponent, -np.ones(m.sum())]).T
    y = np.log(errors[m])
    (b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return b, c, float(np.sqrt(np.mean((y - A @ (b, c))**2)))


def decay_end(errors: np.ndarray, fit_start: int = 100, start: int = 450,
              band: Tuple[float, float] = (0.8, 1.2), width: float = 0.25,
              persist: int = 15, back: int = 10) -> int:
    r"""Return the last n before the decay of the errors clearly changes.

    The change is the drop of the errors of a finite training set to zero, a
    slowdown or a plateau. The local slope of :math:`\log \varepsilon(n)`
    against :math:`n^{1/3}` (fitted on :math:`n^{1/3} - ` ``width``
    :math:`< x \le n^{1/3}`) is compared with its median on
    ``fit_start`` :math:`\le n \le` ``start``: the end is the first
    :math:`n \ge` ``start`` from which it stays outside ``band`` times the
    median for ``persist`` modes, moved back by ``back`` modes, or the last
    mode.

    Args:
        errors: :math:`\varepsilon(n)` for :math:`n = 1, 2, \dots`.
        fit_start: first n of the fit.
        start: first n where the change is looked for.
        band: allowed range of the local slope over the median.
        width: width of the local fit in :math:`n^{1/3}`.
        persist: number of modes the slope must stay outside the band.
        back: number of modes the end is moved back.
    """
    n = np.arange(1, len(errors) + 1)
    x, y = n**(1 / 3), np.log(errors)

    def slope(m: int) -> float:
        w = (x > m**(1 / 3) - width) & (x <= m**(1 / 3))
        return -np.polyfit(x[w], y[w], 1)[0]

    ref = np.median([slope(m) for m in
                     range(fit_start + 25, min(start, len(errors)) + 1, 5)])
    out = [not band[0] <= slope(m) / ref <= band[1]
           for m in range(start, len(errors) + 1)]
    for i in range(len(out)):
        if all(out[i:i + persist]):
            return start + i - back
    return len(errors)


def fit_window(errors: np.ndarray, equations: str, ins_start: int = 100,
               stokes_start: int = 50,
               stokes_floor: float = 1e-13) -> Tuple[int, int]:
    r"""Return the window ``(n_min, n_max)`` of the decay fit.

    Navier-Stokes: from ``ins_start`` to :func:`decay_end`. Stokes: from
    ``stokes_start`` to the round-off floor ``stokes_floor``
    :math:`\varepsilon(1)` of the :math:`K`-dimensional solution set.

    Args:
        errors: :math:`\varepsilon(n)` for :math:`n = 1, 2, \dots`.
        equations: ``'ins'`` or ``'stokes'``.
        ins_start: first n of the Navier-Stokes fit.
        stokes_start: first n of the Stokes fit.
        stokes_floor: round-off floor of Stokes, relative to
            :math:`\varepsilon(1)`.
    """
    if equations == "ins":
        return ins_start, decay_end(errors, ins_start)
    below = errors < stokes_floor * errors[0]
    return stokes_start, int(np.argmax(below)) if below.any() else len(errors)
