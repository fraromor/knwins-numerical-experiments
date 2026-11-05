r"""Source terms: the corner-scale parameter space.

With :math:`\varphi_k` the Dirichlet Laplace eigenfunctions
(:math:`\lambda_1 \le \lambda_2 \le \dots`), :math:`r = |x - c_\iota|` the
distance from the re-entrant corner of the geometry and the corner scales

.. math::

    \chi_j(x) = \exp\Big(-\frac{\log_2(r_j / r)^2}{2 s^2}\Big), \qquad
    r_j = R_0 2^{-j}, \quad R_0 = 0.1 H,

(:math:`H` the channel height), the source terms are

.. math::

    f(y) = \sigma \sum_{k \le K_\varphi,\, j < J} y_{kj}\, a_{kj}
    (\varphi_k \chi_j, 0), \qquad
    a_{kj} \|(\varphi_k \chi_j, 0)\|_{W_h^*} = \frac{\tau}{K}, \quad
    y \in \{-1, 1\}^K,

with :math:`K_\varphi = 33`, :math:`J = 15`, :math:`K = K_\varphi J = 495`
and :math:`s = 0.25`. The parameters :math:`y_1, \dots, y_{750}` are random
(seed 0) and the same in every study, and
:math:`\sigma = \tau / \max_{i \le 750} \|f(y_i)\|_{W_h^*}` puts the largest
of them on the small data threshold :math:`\tau`.

Each :math:`\varphi_k` lies in the corner-weighted analytic class
:math:`B^0_\beta(\Omega)`; multiplication by :math:`\chi_j` keeps it in the
class with constants independent of :math:`j`, and so does the linear
combination: the parameter space is a compact subset of a fixed analytic class
that satisfies the small data assumption. The scales :math:`r_j \to 0` resolve
the corner singularities, so the solution set is far from the
:math:`K`-dimensional span of the sources: the Stokes solution set has
:math:`d_n = 0` for :math:`n \ge K`, the Navier-Stokes one decays as
:math:`\exp(-b n^{1/3})`.
"""
from __future__ import annotations

from typing import Callable, Iterator, Tuple

import numpy as np
import scipy.sparse.linalg as spla
from dolfin import (DirichletBC, DomainBoundary, Function, FunctionSpace,
                    Mesh, TestFunction, TrialFunction, assemble, assign, dx,
                    grad, inner)

from .benchmarks import Geometry
from .spaces import VelocitySpace, to_scipy

#: eigenfunctions, corner scales, width of the scales
K_PHI, J_SCALES, S = 33, 15, 0.25
#: dimension of the parameter space
K = K_PHI * J_SCALES
#: number of parameters y_i (sigma is the same for every study)
N_MAX = 750
SEED = 0
#: largest corner scale R_0 / H
R0 = 0.1


def laplace_eigenpairs(mesh: Mesh, n: int) -> Iterator[Tuple[float, Function]]:
    r"""Yield the n smallest Dirichlet Laplace eigenpairs.

    The eigenproblem is :math:`-\Delta \varphi = \lambda \varphi`,
    :math:`\varphi = 0` on the boundary (:math:`P_2`). The boundary dofs are
    eliminated, so no spurious eigenpairs arise. The sign of :math:`\varphi`
    is fixed by making its largest-magnitude dof positive.

    Args:
        mesh: the mesh.
        n: number of eigenpairs.

    Yields:
        ``(lambda, phi)`` in increasing order of the eigenvalues.
    """
    V = FunctionSpace(mesh, "Lagrange", 2)
    u, v = TrialFunction(V), TestFunction(V)
    K = to_scipy(assemble(inner(grad(u), grad(v)) * dx))
    M = to_scipy(assemble(u * v * dx))
    bc = DirichletBC(V, 0.0, DomainBoundary())
    boundary = list(bc.get_boundary_values().keys())
    free = np.setdiff1d(np.arange(V.dim()), boundary)

    values, vectors = spla.eigsh(K[free][:, free], k=n, M=M[free][:, free],
                                 sigma=0, which="LM", tol=1e-12)
    for j in np.argsort(values):
        phi = Function(V)
        x = np.zeros(V.dim())
        x[free] = vectors[:, j]
        # fix the arbitrary sign of the eigenvector
        x *= np.sign(x[np.argmax(np.abs(x))])
        phi.vector().set_local(x)
        phi.vector().apply("insert")
        yield values[j], phi


def parameters(n: int) -> np.ndarray:
    r"""Return :math:`y_1, \dots, y_n \in \{-1, 1\}^K` as columns.

    :math:`y_i` does not depend on ``n``.

    Args:
        n: number of parameters, at most :data:`N_MAX`.
    """
    rng = np.random.default_rng(SEED)
    return np.sign(rng.uniform(-1, 1, (N_MAX, K)).T[:, :n])


def corner_scales(r: np.ndarray, height: float) -> Iterator[np.ndarray]:
    r"""Yield :math:`\chi_j(r)`, :math:`j = 0, \dots, J - 1`.

    Args:
        r: distances from the corner.
        height: channel height :math:`H`.
    """
    for j in range(J_SCALES):
        r_j = R0 * height * 2.0**-j
        yield np.exp(-np.log2(r_j / r)**2 / (2 * S**2))


def corner_basis(space: VelocitySpace, geometry: Geometry,
                 tau: float) -> np.ndarray:
    r"""Return the columns :math:`a_{kj} (\varphi_k \chi_j, 0)`.

    Args:
        space: the discrete velocity space, for the dual norm.
        geometry: the geometry, for the corner and the height.
        tau: small data threshold :math:`\tau`.

    Returns:
        Dof vectors on ``space.V`` as columns, shape ``(ndofs, K)``, with
        dual norm :math:`\tau / K`.
    """
    mesh = space.V.mesh()
    V0 = FunctionSpace(mesh, "Lagrange", 2)
    x = V0.tabulate_dof_coordinates()[:, :2]
    r = np.maximum(np.linalg.norm(x - np.array(geometry.corner), axis=1),
                   1e-300)
    chis = list(corner_scales(r, geometry.height))
    psi, g = Function(V0), Function(space.V)
    basis = []
    for _, phi in laplace_eigenpairs(mesh, K_PHI):
        for chi in chis:
            psi.vector().set_local(phi.vector().get_local() * chi)
            psi.vector().apply("insert")
            assign(g.sub(0), psi)
            basis.append(g.vector().get_local() / space.dual_norm(g))
    return np.array(basis).T * (tau / K)


def corner_sources(space: VelocitySpace, geometry: Geometry, n: int,
                   tau: float) -> Tuple[float, Callable[[],
                                                        Iterator[Function]]]:
    r"""Return :math:`\sigma` and a generator of the source terms.

    Args:
        space: the discrete velocity space, for the dual norm.
        geometry: the geometry, for the corner and the height.
        n: number of source terms :math:`f(y_1), \dots, f(y_n)`.
        tau: small data threshold :math:`\tau`.

    Returns:
        ``(sigma, sources)``: ``sources()`` yields the source terms,
        Functions on the :math:`[P_2]^2` space ``space.V`` (the same Function,
        overwritten).
    """
    basis = corner_basis(space, geometry, tau)
    f = Function(space.V)

    def source(y: np.ndarray) -> Function:
        f.vector().set_local(basis.dot(y))
        f.vector().apply("insert")
        return f

    sigma = tau / max(space.dual_norm(source(y)) for y in parameters(N_MAX).T)
    basis *= sigma

    def sources() -> Iterator[Function]:
        for y in parameters(n).T:
            yield source(y)
    return sigma, sources
