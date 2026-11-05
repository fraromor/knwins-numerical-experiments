r"""High-fidelity solvers of the steady incompressible Navier-Stokes equations.

.. math::

    -2 \nu\, \mathrm{div}\, \varepsilon(u) + (u\cdot\nabla) u + \nabla p = f,
    \qquad \mathrm{div}\, u = 0,

with Taylor-Hood :math:`P_2/P_1` elements and Newton's method, and of the
Stokes equations (the same without the convective term).
"""
from __future__ import annotations

from typing import Tuple

from dolfin import (CompiledSubDomain, Constant, DirichletBC, FiniteElement,
                    Function, FunctionAssigner, FunctionSpace, MixedElement,
                    TestFunctions, VectorElement, assemble, div, dx, grad,
                    inner, solve, split, sym)

from .benchmarks import Benchmark

NEWTON = {"relative_tolerance": 1e-8,
          # source amplitudes span many orders of magnitude
          "absolute_tolerance": 1e-300,
          "maximum_iterations": 100}


class NavierStokesSolver:
    r"""Taylor-Hood discretization of a benchmark, reused for many sources.

    Args:
        benchmark: the test case.
        viscosity: kinematic viscosity :math:`\nu`.
    """

    def __init__(self, benchmark: Benchmark, viscosity: float) -> None:
        mesh = benchmark.mesh
        cell = mesh.ufl_cell()
        W = FunctionSpace(mesh, MixedElement(
            VectorElement("Lagrange", cell, 2),
            FiniteElement("Lagrange", cell, 1)))
        self.velocity_space = W.sub(0).collapse()
        self.pressure_space = W.sub(1).collapse()
        self._split = FunctionAssigner(
            [self.velocity_space, self.pressure_space], W)
        self.zero_mean_pressure = benchmark.zero_mean_pressure
        self.area = benchmark.area

        self.w = Function(W)
        self.f = Function(self.velocity_space)
        u, p = split(self.w)
        v, q = TestFunctions(W)
        nu = Constant(viscosity)
        self.residual = (2 * nu * inner(sym(grad(u)), sym(grad(v))) * dx
                         + inner(grad(u) * u, v) * dx
                         - p * div(v) * dx + q * div(u) * dx
                         - inner(self.f, v) * dx)
        self.bcs = benchmark.velocity_bcs(W.sub(0))
        if self.zero_mean_pressure:
            # pressure unique up to a constant: pin it at one vertex (else the
            # Jacobian is singular and Newton can fail), then normalize to
            # zero mean after the solve
            x, y = mesh.coordinates()[0]
            vertex = CompiledSubDomain("near(x[0], a) && near(x[1], b)",
                                       a=x, b=y)
            self.bcs.append(DirichletBC(W.sub(1), Constant(0.0), vertex,
                                        "pointwise"))

    def solve(self, f: Function) -> Tuple[Function, Function]:
        """Solve for a source term.

        Args:
            f: source term, a Function on :math:`[P_2]^2`.

        Returns:
            The velocity ``u`` and the pressure ``p``.
        """
        self.f.interpolate(f)  # same P2 element: exact
        # zero initial guess: Newton tolerance relative to ||f||
        self.w.vector().zero()
        solve(self.residual == 0, self.w, self.bcs,
              solver_parameters={"newton_solver": NEWTON})

        u, p = Function(self.velocity_space), Function(self.pressure_space)
        self._split.assign([u, p], self.w)
        if self.zero_mean_pressure:
            p.vector()[:] -= assemble(p * dx) / self.area
        return u, p


class StokesSolver(NavierStokesSolver):
    r"""The Stokes equations: :class:`NavierStokesSolver` without convection.

    Args:
        benchmark: the test case.
        viscosity: kinematic viscosity :math:`\nu`.
    """

    def __init__(self, benchmark: Benchmark, viscosity: float) -> None:
        super().__init__(benchmark, viscosity)
        u, p = split(self.w)
        v, q = TestFunctions(self.w.function_space())
        nu = Constant(viscosity)
        self.residual = (2 * nu * inner(sym(grad(u)), sym(grad(v))) * dx
                         - p * div(v) * dx + q * div(u) * dx
                         - inner(self.f, v) * dx)
