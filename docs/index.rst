Kolmogorov n-widths of Navier–Stokes solution sets
==================================================

Numerical evidence for the exponential decay of the Kolmogorov n-widths of the
solution set of the steady incompressible Navier–Stokes equations

.. math::

    -2\nu\,\mathrm{div}\,\varepsilon(u) + (u\cdot\nabla)u + \nabla p = f, \qquad \mathrm{div}\,u = 0,

when the source terms :math:`f` range over a compact set of analytic functions that
satisfies the small data assumption. The upper bounds of the n-widths are the
maximum POD projection errors of the computed solutions, which decay as
:math:`\exp(-b\, n^{1/3})`; for the same source terms the Stokes solution set lies
in a finite-dimensional space.

Method
------

For each benchmark (geometry, boundary conditions and mesh,
:mod:`knwins.benchmarks`):

1. **Small data constants** :math:`C_\mathrm{coer}`, :math:`C_\mathrm{cont}` on the
   discrete velocity space of the base mesh (:mod:`knwins.constants`,
   :mod:`knwins.spaces`), and the threshold
   :math:`\tau = \nu^2 C_\mathrm{coer}^2 / (4 C_\mathrm{cont})`.
2. **Graded mesh** towards the re-entrant corner of the geometry
   (:meth:`knwins.benchmarks.Geometry.graded_mesh`).
3. **Source terms**: the corner-scale parameter space, Dirichlet Laplace
   eigenfunctions multiplied by corner scales, :math:`K = 495` parameters
   (:mod:`knwins.sources`).
4. **High-fidelity snapshots**: Taylor–Hood :math:`P_2/P_1` and Newton's method,
   Navier–Stokes and Stokes (:mod:`knwins.solvers`).
5. **Weighted POD** and maximum projection errors (:mod:`knwins.pod`).
6. **Decay fits**: least-squares fits :math:`\varepsilon(n) \approx e^{-b n^a - c}`,
   :math:`a = 1/3, 1/2`, with the rms misfit (:func:`knwins.pod.fit_decay`,
   :func:`knwins.pod.fit_window`).

The studies of the paper are collected in :mod:`experiments` (:class:`knwins.study.Study`).

Usage
-----

.. code-block:: bash

    conda env create -f environment.yml
    conda activate knwins
    python run.py constants [benchmark ...]   # coercivity / continuity constants
    python run.py study <study> [...]         # snapshots and POD errors
    python run.py errors <study> [...]        # POD errors only (from saved snapshots)
    python run.py fields <study> [...]        # field plots (from saved snapshots)
    python run.py figures                     # decay plots and README table

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/knwins
   api/scripts
