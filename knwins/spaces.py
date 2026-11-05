r"""Discrete velocity space and Gram matrices of the H^1 and L^2 norms.

:math:`W_h = \{v_h \in [P_2(\mathcal{T}_h)]^2` satisfying the essential b.c.
of the benchmark :math:`\}`, with the full :math:`H^1` norm
:math:`\|v\|^2 = \|v\|_{L^2}^2 + \|\nabla v\|_{L^2}^2`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dolfin import (Function, FunctionSpace, Matrix, TestFunction,
                    TrialFunction, VectorFunctionSpace, as_backend_type,
                    assemble, dot, dx, grad, inner)
from ufl import Form

from .benchmarks import Benchmark


def to_scipy(A: Matrix) -> sp.csr_matrix:
    """Convert a dolfin/PETSc matrix to a scipy CSR matrix.

    Args:
        A: assembled dolfin matrix (PETSc backend).
    """
    indptr, indices, data = as_backend_type(A).mat().getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=(A.size(0), A.size(1)))


def h1_gram(V: FunctionSpace) -> sp.csr_matrix:
    """Return the Gram matrix of the full :math:`H^1` inner product.

    Args:
        V: the function space (all its dofs).
    """
    u, v = TrialFunction(V), TestFunction(V)
    return to_scipy(assemble(inner(grad(u), grad(v)) * dx + inner(u, v) * dx))


def l2_gram(V: FunctionSpace) -> sp.csr_matrix:
    """Return the Gram matrix (mass matrix) of the :math:`L^2` inner product.

    Args:
        V: the function space (all its dofs).
    """
    u, v = TrialFunction(V), TestFunction(V)
    return to_scipy(assemble(inner(u, v) * dx))


class VelocitySpace:
    """:math:`W_h`, represented by the free (unconstrained) dofs of [P_2]^2.

    Args:
        benchmark: the test case, for the mesh and the essential b.c.

    Attributes:
        V: the :math:`[P_2]^2` space.
        free: indices of the free dofs.
        gram: :math:`H^1` Gram matrix on the free dofs.
        gram_solve: solver of ``gram @ x = b`` (sparse LU).
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self.V = VectorFunctionSpace(benchmark.mesh, "Lagrange", 2)
        constrained = set()
        for bc in benchmark.velocity_bcs(self.V):
            constrained.update(bc.get_boundary_values().keys())
        self.free = np.setdiff1d(np.arange(self.V.dim()),
                                 np.fromiter(constrained, dtype=int))
        self.gram = self.restrict(h1_gram(self.V))
        self.gram_solve = spla.splu(self.gram.tocsc()).solve

    def restrict(self, A: sp.spmatrix) -> sp.csr_matrix:
        """Restrict a scipy matrix on :math:`[P_2]^2` to the free dofs.

        Args:
            A: matrix on all dofs.
        """
        return A[self.free, :][:, self.free].tocsr()

    def matrix(self, form: Form) -> sp.csr_matrix:
        """Assemble a bilinear form, restricted to the free dofs.

        Args:
            form: bilinear UFL form on ``V``.
        """
        return self.restrict(to_scipy(assemble(form)))

    def function(self, x: np.ndarray) -> Function:
        """Return the Function with free dofs x (zero on the constrained).

        Args:
            x: values of the free dofs.
        """
        f = Function(self.V)
        values = np.zeros(self.V.dim())
        values[self.free] = x
        f.vector().set_local(values)
        f.vector().apply("insert")
        return f

    def norm(self, x: np.ndarray) -> float:
        """Return the :math:`H^1` norm of a vector of free dofs.

        Args:
            x: values of the free dofs.
        """
        return np.sqrt(x.dot(self.gram.dot(x)))

    def riesz(self, g: np.ndarray) -> np.ndarray:
        """Return the :math:`H^1` Riesz representative of a functional.

        Args:
            g: the functional, a vector on all dofs.

        Returns:
            The representative, on the free dofs.
        """
        return self.gram_solve(g[self.free])

    def dual_norm(self, f: Function) -> float:
        r"""Return the dual norm :math:`\|f\|_{W_h^*}`.

        :math:`\|f\|_{W_h^*} = \sup_{v \in W_h} (f, v) / \|v\|_{H^1}`.

        Args:
            f: a Function on ``V``.
        """
        g = assemble(dot(f, TestFunction(self.V)) * dx).get_local()
        return np.sqrt(g[self.free].dot(self.riesz(g)))
