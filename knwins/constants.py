r"""Coercivity and continuity constants of the small data assumption.

.. math::

    \|f\|_{W_h^*} \le \tau = \frac{\nu^2 C_\mathrm{coer}^2}{4 C_\mathrm{cont}},

computed on the discrete velocity space :math:`W_h` (see
:class:`~knwins.spaces.VelocitySpace`):

* :math:`C_\mathrm{coer} = \min_{v \in W_h} 2 \|\varepsilon(v)\|^2
  / \|v\|_{H^1}^2`: symmetric generalized eigenvalue problem.
* :math:`C_\mathrm{cont} = \sup_{u,v,w \in W_h} \int (u\cdot\nabla)v\cdot w
  \,/\, (\|u\| \|v\| \|w\|)`: lower estimate by an alternating higher-order
  power method (HOPM) with restarts: every value is attained, hence a lower
  bound of :math:`C_{\mathrm{cont},h}`.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dolfin import (TestFunction, TrialFunction, assemble, dot, dx, grad,
                    inner, sym)

from .benchmarks import Benchmark
from .spaces import VelocitySpace


def small_data_threshold(viscosity: float, C_coer: float,
                         C_cont: float) -> float:
    r"""Return :math:`\tau = \nu^2 C_\mathrm{coer}^2 / (4 C_\mathrm{cont})`.

    Args:
        viscosity: kinematic viscosity :math:`\nu`.
        C_coer: coercivity constant.
        C_cont: continuity constant.
    """
    return viscosity**2 * C_coer**2 / (4 * C_cont)


# ------------------------------------------------------------------ coercivity

def coercivity_modes(space: VelocitySpace,
                     k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    r"""Return the k smallest eigenpairs of the coercivity problem.

    The eigenproblem is :math:`2 (\varepsilon(u), \varepsilon(v))
    = \lambda (u, v)_{H^1}` on :math:`W_h`.

    Args:
        space: the discrete velocity space.
        k: number of eigenpairs.

    Returns:
        The eigenvalues in increasing order (shape ``(k,)``) and the
        eigenvectors on the free dofs (as columns).
    """
    u, v = TrialFunction(space.V), TestFunction(space.V)
    A = space.matrix(2 * inner(sym(grad(u)), sym(grad(v))) * dx)
    values, vectors = spla.eigsh(A, k=k, M=space.gram, sigma=0, which="LM",
                                 tol=1e-12)
    order = np.argsort(values)
    return values[order], vectors[:, order]


def coercivity_constant(space: VelocitySpace) -> float:
    """Return the smallest eigenvalue of :func:`coercivity_modes`.

    Args:
        space: the discrete velocity space.
    """
    return coercivity_modes(space)[0][0]


# ---------------------------------------------------------- continuity: HOPM

def _top_singular_pair(
        space: VelocitySpace, B: sp.csr_matrix,
        v0: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return max_{v,w} w^T B v / (||v|| ||w||) in the H^1 norm, and v, w.

    Largest eigenpair of B^T G^{-1} B x = sigma^2 G x. Returns sigma and the
    normalized v, w.
    """
    n = B.shape[1]
    BtGinvB = spla.LinearOperator(
        (n, n), dtype=float,
        matvec=lambda x: B.T.dot(space.gram_solve(B.dot(x))))
    Ginv = spla.LinearOperator((n, n), dtype=float, matvec=space.gram_solve)
    _, vectors = spla.eigsh(BtGinvB, k=1, M=space.gram, Minv=Ginv,
                            which="LA", v0=v0, tol=1e-10)
    v = vectors[:, 0] / space.norm(vectors[:, 0])
    w = space.gram_solve(B.dot(v))
    sigma = space.norm(w)
    return sigma, v, w / sigma


def continuity_constant_hopm(space: VelocitySpace, n_starts: int = 10,
                             max_iter: int = 150, tol: float = 1e-9,
                             seed: int = 0, verbose: bool = True
                             ) -> Tuple[float, List[float]]:
    r"""Return a lower estimate of :math:`C_{\mathrm{cont},h}`.

    Block-alternating maximization of
    :math:`t(u, v, w) = \int (u\cdot\nabla)v\cdot w` over the :math:`H^1`
    unit spheres of :math:`W_h`:

    * fixed :math:`u`: :math:`(v, w)` = top :math:`H^1` singular pair of
      :math:`B_u[w, v] = t(u, v, w)`;
    * fixed :math:`(v, w)`: :math:`u` = normalized Riesz representative of
      :math:`t(\cdot, v, w)`.

    Both updates are exact maximizations, so :math:`t` increases
    monotonically. Starts: the 4 lowest coercivity modes, then random
    :math:`H^1`-smoothed fields.

    Args:
        space: the discrete velocity space.
        n_starts: number of starts.
        max_iter: maximum number of iterations per start.
        tol: relative tolerance on the increase of :math:`t`.
        seed: seed of the random starts.
        verbose: print the value reached from every start.

    Returns:
        The best value and the value reached from every start.
    """
    V = space.V
    v_trial, w_test = TrialFunction(V), TestFunction(V)
    u_test = TestFunction(V)

    rng = np.random.default_rng(seed)
    starts = list(coercivity_modes(space, k=4)[1].T)[:n_starts]
    while len(starts) < n_starts:
        starts.append(space.riesz(rng.standard_normal(V.dim())))

    values = []
    for s, xu in enumerate(starts):
        xu = xu / space.norm(xu)
        t_old, xv = 0.0, None
        for it in range(max_iter):
            u = space.function(xu)
            B = space.matrix(dot(grad(v_trial) * u, w_test) * dx)
            _, xv, xw = _top_singular_pair(space, B, v0=xv)

            v, w = space.function(xv), space.function(xw)
            g = assemble(dot(grad(v) * u_test, w) * dx).get_local()
            y = space.riesz(g)
            t = np.sqrt(g[space.free].dot(y))
            xu = y / t
            if abs(t - t_old) < tol * t:
                break
            t_old = t
        values.append(t)
        if verbose:
            print("HOPM start {:2d}: t = {:.8e} ({} it), best = {:.8e}".format(
                s, t, it + 1, max(values)), flush=True)
    return max(values), values


# ----------------------------------------------------------------- all at once

def compute_constants(benchmark: Benchmark, n_starts: int = 10,
                      verbose: bool = True) -> Dict[str, Any]:
    """Compute the coercivity and continuity constants of a benchmark.

    The constants are computed on the base mesh of the benchmark.

    Args:
        benchmark: the test case.
        n_starts: number of HOPM starts.
        verbose: print the progress.

    Returns:
        Dictionary with keys ``benchmark``, ``C_coer``, ``C_cont``,
        ``C_cont_starts``, ``ndofs`` and ``nfree``.
    """
    # on the base mesh: the constants do not change under refinement
    # (cube_dirichlet / cube_dirichlet_fine), the graded mesh is much larger
    space = VelocitySpace(benchmark.ungraded())
    C_coer = coercivity_constant(space)
    C_cont, C_cont_starts = continuity_constant_hopm(space, n_starts=n_starts,
                                                     verbose=verbose)
    constants = dict(benchmark=benchmark.name, C_coer=C_coer, C_cont=C_cont,
                     C_cont_starts=C_cont_starts,
                     ndofs=space.V.dim(), nfree=len(space.free))
    if verbose:
        print("{}: C_coer = {:.6e}, C_cont (HOPM) = {:.6e}".format(
            benchmark.name, C_coer, C_cont), flush=True)
    return constants


def load_or_compute_constants(benchmark: Benchmark, directory: str,
                              recompute: bool = False) -> Dict[str, Any]:
    """Return the constants of the benchmark.

    The constants are cached in ``<directory>/<benchmark>.json``.

    Args:
        benchmark: the test case.
        directory: cache directory.
        recompute: recompute even if the cache file exists.

    Returns:
        The dictionary of :func:`compute_constants`.
    """
    path = os.path.join(directory, benchmark.name + ".json")
    if recompute or not os.path.exists(path):
        constants = compute_constants(benchmark)
        os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump(constants, f, indent=2, default=float)
    with open(path) as f:
        return json.load(f)
