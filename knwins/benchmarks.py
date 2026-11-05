r"""Benchmarks = geometry + boundary conditions + mesh.

A :class:`Benchmark` is the only object the rest of the code needs to know
about a test case. It provides

* ``benchmark.mesh``: the dolfin mesh, graded towards the re-entrant corner
  ``benchmark.geometry.corner`` (:meth:`Geometry.graded_mesh`);
* ``benchmark.velocity_bcs(V)``: essential b.c. for a :math:`[P_2]^2`
  space ``V``;
* ``benchmark.zero_mean_pressure``: True if the pressure is unique only in
  :math:`L^2_0`;
* ``benchmark.area``: :math:`|\Omega|`;
* ``benchmark.geometry.length/height``: bounding box of the channel.

The small data constants are computed on the base mesh
(``benchmark.ungraded()``), the snapshots on the graded mesh.
"""
from __future__ import annotations

import os
from functools import cached_property
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from dolfin import (Constant, DirichletBC, DomainBoundary, File,
                    FunctionSpace, Mesh, MeshFunction, SubDomain, assemble,
                    cells, dx, refine)

MESH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "meshes")
GRADED_DIR = os.path.join(MESH_DIR, "graded")
TOL = 1e-8

# boundary markers
WALLS, INLET, OUTLET, NOSLIP_FACE = 2, 1, 4, 3


# ------------------------------------------------------------------ geometries

class _BoundaryPart(SubDomain):
    def __init__(self, inside: Callable[[np.ndarray], bool]) -> None:
        super().__init__()
        self._inside = inside

    def inside(self, x: np.ndarray, on_boundary: bool) -> bool:
        return on_boundary and self._inside(x)


class Geometry:
    r"""Channel [0, length] x [0, height], possibly with a step or an obstacle.

    Inlet at x = 0, outlet at x = length, the horizontal channel walls, and the
    no-slip faces: the square for the cube, the vertical face of the step for
    the backward-facing step. The no-slip faces are separate from the walls
    because the free slip condition u_y = 0 of :class:`SlipWallsOutflow` holds
    only on horizontal walls: they always get u = 0.

    ``corner`` is the re-entrant corner :math:`c_\iota` of the corner-scale
    source terms (:mod:`knwins.sources`); the meshes are graded towards it.
    """

    name: Optional[str] = None
    title: Optional[str] = None
    length: Optional[float] = None
    height: Optional[float] = None
    corner: Optional[Tuple[float, float]] = None
    meshes: Dict[str, str] = {}      # resolution -> mesh file in meshes/
    # graded mesh: h <= max(grading r, h_min height) for r < radius height,
    # r the distance from the corner
    grading, radius, h_min = 0.12, 0.25, 5e-7

    def build_mesh(self, resolution: str) -> Mesh:
        """Return the mesh of a resolution.

        By default the mesh is read from the file ``meshes[resolution]`` in
        ``meshes/``. Override this method to generate the mesh in Python.

        Args:
            resolution: key of :attr:`meshes`.
        """
        return Mesh(os.path.join(MESH_DIR, self.meshes[resolution]))

    def graded_mesh(self, resolution: str) -> Mesh:
        r"""Return the mesh of a resolution graded towards :attr:`corner`.

        The cells of :meth:`build_mesh` with midpoint at distance
        :math:`r < ` ``radius`` :math:`\cdot` ``height`` from the corner are
        refined (bisection) until
        :math:`h \le \max(` ``grading`` :math:`r,` ``h_min`` :math:`\cdot`
        ``height`` :math:`)`. The mesh is cached in ``meshes/graded/``.

        Args:
            resolution: key of :attr:`meshes`.
        """
        path = os.path.join(GRADED_DIR,
                            "{}_{}.xml.gz".format(self.name, resolution))
        if os.path.exists(path):
            return Mesh(path)
        mesh = self.build_mesh(resolution)
        corner = np.array(self.corner)
        h_min = self.h_min * self.height
        while True:
            markers = MeshFunction("bool", mesh, 2, False)
            marked = 0
            for cell in cells(mesh):
                r = np.linalg.norm(cell.midpoint().array()[:2] - corner)
                if (r < self.radius * self.height
                        and cell.h() > max(self.grading * r, h_min)):
                    markers[cell] = True
                    marked += 1
            if marked == 0:
                break
            mesh = refine(mesh, markers)
        os.makedirs(GRADED_DIR, exist_ok=True)
        File(path) << mesh
        return mesh

    def is_noslip_face(self, x: np.ndarray) -> bool:
        """Return whether the boundary point ``x`` lies on a no-slip face.

        Args:
            x: coordinates of a boundary point.
        """
        raise NotImplementedError

    def mark_boundaries(self, mesh: Mesh) -> MeshFunction:
        """Mark the facets: WALLS, INLET, OUTLET and NOSLIP_FACE.

        Args:
            mesh: mesh of the geometry.
        """
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
        _BoundaryPart(lambda x: True).mark(boundaries, WALLS)
        _BoundaryPart(lambda x: x[0] < TOL).mark(boundaries, INLET)
        outlet = _BoundaryPart(lambda x: x[0] > self.length - TOL)
        outlet.mark(boundaries, OUTLET)
        _BoundaryPart(self.is_noslip_face).mark(boundaries, NOSLIP_FACE)
        return boundaries


class FlowPastCube(Geometry):
    """Channel [0, 2.2] x [0, 0.4] with the square obstacle [0.15, 0.25]^2.

    The corner scales are centred at the obstacle corner (0.25, 0.25).
    """

    name, title = "cube", "flow past a cube"
    length, height = 2.2, 0.4
    corner = (0.25, 0.25)
    meshes = {"standard": "cube.xml", "fine": "cube_fine.xml"}

    def is_noslip_face(self, x: np.ndarray) -> bool:
        """Return whether ``x`` lies on the square."""
        return (0.1 - TOL < x[0] < 0.3 + TOL
                and 0.1 - TOL < x[1] < 0.3 + TOL)


class BackwardFacingStep(Geometry):
    """Channel [0, 22] x [0, 5] with a step of height 2 up to x = 4.

    The corner scales are centred at the step corner (4, 2).
    """

    name, title = "backward", "backward-facing step"
    length, height = 22.0, 5.0
    corner = (4.0, 2.0)
    step_x, step_height = 4.0, 2.0
    meshes = {"standard": "backward.xml", "coarse": "backward_coarse.xml"}

    def is_noslip_face(self, x: np.ndarray) -> bool:
        """Return whether ``x`` lies on the vertical face of the step."""
        return (abs(x[0] - self.step_x) < TOL
                and x[1] < self.step_height + TOL)


# --------------------------------------------------------- boundary conditions

class BoundaryConditions:
    """Homogeneous essential boundary conditions of the velocity."""

    name: Optional[str] = None
    zero_mean_pressure: bool = False

    def velocity_bcs(self, V: FunctionSpace,
                     boundaries: MeshFunction) -> List[DirichletBC]:
        """Return the essential boundary conditions of the velocity.

        Args:
            V: :math:`[P_2]^2` velocity space (possibly a subspace of a mixed
                space).
            boundaries: facet markers of :meth:`Geometry.mark_boundaries`.
        """
        raise NotImplementedError


class NoSlip(BoundaryConditions):
    """u = 0 on the whole boundary."""

    name = "dirichlet"
    zero_mean_pressure = True

    def velocity_bcs(self, V: FunctionSpace,
                     boundaries: MeshFunction) -> List[DirichletBC]:
        """Return u = 0 on the whole boundary."""
        return [DirichletBC(V, Constant((0.0, 0.0)), DomainBoundary())]


class NoSlipOutflow(BoundaryConditions):
    """u = 0 on inlet, walls and no-slip faces; do-nothing outflow."""

    name = "mixed"
    zero_mean_pressure = False

    def velocity_bcs(self, V: FunctionSpace,
                     boundaries: MeshFunction) -> List[DirichletBC]:
        """Return u = 0 on inlet, walls and no-slip faces."""
        zero = Constant((0.0, 0.0))
        return [DirichletBC(V, zero, boundaries, marker)
                for marker in (INLET, WALLS, NOSLIP_FACE)]


class SlipWallsOutflow(BoundaryConditions):
    """Free slip on the horizontal walls, u = 0 on inlet and no-slip faces.

    Free slip is u_y = 0 on the horizontal walls; do-nothing outflow.
    """

    name = "slip"
    zero_mean_pressure = False

    def velocity_bcs(self, V: FunctionSpace,
                     boundaries: MeshFunction) -> List[DirichletBC]:
        """Return u = 0 on inlet and no-slip faces, u_y = 0 on the walls."""
        zero = Constant((0.0, 0.0))
        return [DirichletBC(V, zero, boundaries, INLET),
                DirichletBC(V.sub(1), Constant(0.0), boundaries, WALLS),
                DirichletBC(V, zero, boundaries, NOSLIP_FACE)]


# ------------------------------------------------------------------- benchmark

class Benchmark:
    """A test case: geometry, boundary conditions and mesh resolution.

    Args:
        geometry: the channel geometry.
        conditions: the boundary conditions of the velocity.
        resolution: key of ``geometry.meshes``.
        graded: grade the mesh towards ``geometry.corner``.
    """

    def __init__(self, geometry: Geometry, conditions: BoundaryConditions,
                 resolution: str = "standard", graded: bool = True) -> None:
        self.geometry = geometry
        self.conditions = conditions
        self.resolution = resolution
        self.graded = graded
        self.name = "{}_{}".format(geometry.name, conditions.name)
        if resolution != "standard":
            self.name += "_" + resolution

    def ungraded(self) -> Benchmark:
        """Return the same benchmark on the base mesh (same name)."""
        return Benchmark(self.geometry, self.conditions, self.resolution,
                         graded=False)

    @cached_property
    def mesh(self) -> Mesh:
        """The mesh: :meth:`Geometry.graded_mesh` or the base mesh."""
        if self.graded:
            return self.geometry.graded_mesh(self.resolution)
        return self.geometry.build_mesh(self.resolution)

    @cached_property
    def boundaries(self) -> MeshFunction:
        """Facet markers of the mesh."""
        return self.geometry.mark_boundaries(self.mesh)

    @cached_property
    def area(self) -> float:
        r"""Area :math:`|\Omega|` of the domain."""
        return assemble(Constant(1.0) * dx(domain=self.mesh))

    def velocity_bcs(self, V: FunctionSpace) -> List[DirichletBC]:
        """Return the essential boundary conditions of the velocity.

        Args:
            V: :math:`[P_2]^2` velocity space on :attr:`mesh`.
        """
        return self.conditions.velocity_bcs(V, self.boundaries)

    @property
    def zero_mean_pressure(self) -> bool:
        """True if the pressure is unique only up to a constant."""
        return self.conditions.zero_mean_pressure
