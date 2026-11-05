"""A numerical study: snapshots of a benchmark and POD projection errors.

The snapshots are computed for the corner-scale source terms
(:mod:`knwins.sources`); the maximum POD projection errors are upper bounds of
the Kolmogorov n-widths of the solution set. A study can also solve the Stokes
equations for the same source terms, whose solution set lies in a
:math:`K`-dimensional space.
"""
from __future__ import annotations

import json
import os
import time
from typing import Tuple

import numpy as np
from dolfin import Function

from .benchmarks import Benchmark
from .constants import load_or_compute_constants, small_data_threshold
from .plotting import plot_field
from .pod import max_projection_errors
from .solvers import NavierStokesSolver, StokesSolver
from .sources import J_SCALES, K, K_PHI, N_MAX, R0, S, SEED, corner_sources
from .spaces import VelocitySpace, h1_gram, l2_gram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
PLOTTED_SNAPSHOT = 2
EQUATIONS = ("ins", "stokes")


class Study:
    r"""Snapshots and POD errors of a benchmark.

    Snapshot i depends only on :math:`f(y_i)`, so the errors for
    ``n_train < n_snapshots`` use the first ``n_train`` snapshots. The
    Navier-Stokes results are in ``results/<name>/``, the Stokes results in
    ``results/<name>/stokes/``.

    Args:
        name: output folder in ``results/`` and ``figures/``
            (e.g. ``'cube/nu_1'``).
        benchmark: the test case.
        viscosity: kinematic viscosity :math:`\nu`.
        n_snapshots: number of snapshots, at most ``sources.N_MAX``.
        n_trains: numbers of training snapshots of the POD errors.
        plot_fields: plot source term, velocity and pressure of snapshot
            ``PLOTTED_SNAPSHOT``.
        stokes: also solve the Stokes equations.
    """

    def __init__(self, name: str, benchmark: Benchmark,
                 viscosity: float = 1.0, n_snapshots: int = 750,
                 n_trains: Tuple[int, ...] = (250, 500, 750),
                 plot_fields: bool = False, stokes: bool = False) -> None:
        self.name = name
        self.benchmark = benchmark
        self.viscosity = viscosity
        self.n_snapshots = n_snapshots
        self.n_trains = n_trains
        self.plot_fields = plot_fields
        self.stokes = stokes
        self.results = os.path.join(RESULTS, name)
        self.figures = os.path.join(FIGURES, name)

    @property
    def equations(self) -> Tuple[str, ...]:
        """The equations solved: ``('ins',)`` or ``('ins', 'stokes')``."""
        return EQUATIONS if self.stokes else EQUATIONS[:1]

    def path(self, filename: str, equations: str = "ins") -> str:
        """Return the path of a result file of the study.

        Args:
            filename: file name in the results folder.
            equations: ``'ins'`` or ``'stokes'``.
        """
        folder = self.results if equations == "ins" else os.path.join(
            self.results, "stokes")
        return os.path.join(folder, filename)

    def run(self) -> None:
        """Compute the snapshots, then the POD errors."""
        self.compute_snapshots()
        self.compute_errors()

    def threshold(self) -> Tuple[float, dict]:
        r"""Return the small data threshold :math:`\tau` and the constants."""
        constants = load_or_compute_constants(
            self.benchmark, os.path.join(RESULTS, "constants"))
        return small_data_threshold(self.viscosity, constants["C_coer"],
                                    constants["C_cont"]), constants

    def sources(self, n: int):
        r"""Return :math:`\sigma` and the generator of the first n sources.

        Args:
            n: number of source terms.
        """
        tau, _ = self.threshold()
        return corner_sources(VelocitySpace(self.benchmark),
                              self.benchmark.geometry, n, tau)

    def compute_snapshots(self) -> None:
        """Solve for all source terms; save the snapshots and study.json."""
        for equations in self.equations:
            os.makedirs(os.path.dirname(self.path("x", equations)),
                        exist_ok=True)
        tau, constants = self.threshold()
        space = VelocitySpace(self.benchmark)
        sigma, sources = corner_sources(space, self.benchmark.geometry,
                                        self.n_snapshots, tau)
        mesh = self.benchmark.mesh
        print("{}: nu = {}, tau = {:.6e}, sigma = {:.4f}, cells = {}".format(
            self.name, self.viscosity, tau, sigma, mesh.num_cells()),
            flush=True)

        solvers = {"ins": NavierStokesSolver(self.benchmark, self.viscosity)}
        if self.stokes:
            solvers["stokes"] = StokesSolver(self.benchmark, self.viscosity)
        snapshots = {equations: ([], []) for equations in solvers}
        norms, start = [], time.time()
        for i, f in enumerate(sources()):
            norms.append(space.dual_norm(f))
            for equations, solver in solvers.items():
                u, p = solver.solve(f)
                snapshots[equations][0].append(u.vector().get_local())
                snapshots[equations][1].append(p.vector().get_local())
                if (equations == "ins" and self.plot_fields
                        and i == PLOTTED_SNAPSHOT):
                    self.plot_snapshot(i, f, u, p)
            if i % 50 == 0:
                print("solved {} / {} in {:.0f}s".format(
                    i + 1, self.n_snapshots, time.time() - start), flush=True)
        for equations, (U, P) in snapshots.items():
            np.save(self.path("snapshots_u.npy", equations), np.array(U).T)
            np.save(self.path("snapshots_p.npy", equations), np.array(P).T)

        with open(self.path("study.json"), "w") as file:
            json.dump(dict(study=self.name, benchmark=self.benchmark.name,
                           viscosity=self.viscosity,
                           n_snapshots=self.n_snapshots, stokes=self.stokes,
                           K_phi=K_PHI, J_scales=J_SCALES, K=K, s=S,
                           R0=R0 * self.benchmark.geometry.height,
                           corner=self.benchmark.geometry.corner,
                           seed=SEED, n_max=N_MAX, tau=tau, sigma=sigma,
                           max_dual_norm=max(norms),
                           cells=mesh.num_cells(), hmin=mesh.hmin(),
                           constants=constants),
                      file, indent=2)

    def compute_errors(self) -> None:
        """Compute the maximum POD projection errors for every n_train.

        The snapshots are loaded from the results folder.
        """
        solver = NavierStokesSolver(self.benchmark, self.viscosity)
        H = h1_gram(solver.velocity_space)
        L = l2_gram(solver.pressure_space)
        for equations in self.equations:
            U = np.load(self.path("snapshots_u.npy", equations))
            P = np.load(self.path("snapshots_p.npy", equations))
            for n in self.n_trains:
                eps_u = max_projection_errors(U[:, :n], H)
                eps_p = max_projection_errors(P[:, :n], L)
                np.save(self.path("errors_u_{}.npy".format(n), equations),
                        eps_u)
                np.save(self.path("errors_p_{}.npy".format(n), equations),
                        eps_p)
                print("{} {} N={}: eps_u(1) = {:.3e}, eps_u(N-1) = {:.3e}, "
                      "eps_p(1) = {:.3e}, eps_p(N-1) = {:.3e}".format(
                          self.name, equations, n, eps_u[0], eps_u[-1],
                          eps_p[0], eps_p[-1]), flush=True)

    def errors(self, n_train: int,
               equations: str = "ins") -> Tuple[np.ndarray, np.ndarray]:
        """Return the saved velocity and pressure errors.

        Args:
            n_train: number of training snapshots.
            equations: ``'ins'`` or ``'stokes'``.

        Returns:
            ``(eps_u, eps_p)``.
        """
        return (np.load(self.path("errors_u_{}.npy".format(n_train),
                                  equations)),
                np.load(self.path("errors_p_{}.npy".format(n_train),
                                  equations)))

    def plot_saved_snapshot(self) -> None:
        """Plot snapshot ``PLOTTED_SNAPSHOT`` from the saved snapshots."""
        solver = NavierStokesSolver(self.benchmark, self.viscosity)
        u = Function(solver.velocity_space)
        p = Function(solver.pressure_space)
        i = PLOTTED_SNAPSHOT
        for field, name in ((u, "u"), (p, "p")):
            X = np.load(self.path("snapshots_{}.npy".format(name)),
                        mmap_mode="r")
            field.vector().set_local(np.array(X[:, i]))
            field.vector().apply("insert")
        _, sources = self.sources(i + 1)
        for f in sources():
            pass
        self.plot_snapshot(i, f, u, p)

    def plot_snapshot(self, i: int, f: Function, u: Function,
                      p: Function) -> None:
        """Plot source term, velocity components and pressure of a snapshot.

        Args:
            i: snapshot index.
            f: source term.
            u: velocity.
            p: pressure.
        """
        os.makedirs(self.figures, exist_ok=True)
        plot_field(f.sub(0), r"source term $f_{{i,h}}$, i={}".format(i),
                   os.path.join(self.figures, "eig_{}".format(i)),
                   cmap="cividis")
        plot_field(u.sub(0), "velocity x component, i={}".format(i),
                   os.path.join(self.figures, "velocity_x_{}".format(i)))
        plot_field(u.sub(1), "velocity y component, i={}".format(i),
                   os.path.join(self.figures, "velocity_y_{}".format(i)))
        plot_field(p, "pressure, i={}".format(i),
                   os.path.join(self.figures, "pressure_{}".format(i)),
                   cmap="plasma")
