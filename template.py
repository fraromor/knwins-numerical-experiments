"""Template: a convergence study of the paper on a new geometry.

The new geometry is a channel with a cavity. Command line interface::

    python template.py constants [benchmark ...]  # coercivity / continuity
    python template.py study <study> [...]        # snapshots and POD errors
    python template.py errors <study> [...]       # POD errors only
    python template.py figures                    # decay plots, rate table

Everything specific to the new geometry is defined in this file, with one
benchmark, one study and one figure: copy it and change the four marked steps
to study another geometry (see the README, "Expand the benchmarks").

Geometry: channel [0, 4] x [0, 1] with the cavity [1.5, 2.5] x [-1, 0] in the
lower wall; the two upper corners of the cavity are re-entrant, and the corner
scales of the source terms are centred at the upstream one c = (1.5, 0)::

     (0,1) +----------------------------------------+ (4,1)
           |                                        |
    inlet  |                                        |  outlet
           |                                        |
     (0,0) +------------+          +----------------+ (4,0)
                        |  cavity  |
                        |          |
               (1.5,-1) +----------+ (2.5,-1)

The meshes are graded towards c (``Geometry.graded_mesh``).
"""
import sys

import numpy as np
from dolfin import Mesh, MeshFunction, Point, RectangleMesh, SubMesh

from experiments import varying_n
from knwins.benchmarks import TOL, Benchmark, Geometry, NoSlip
from knwins.study import Study
from run import main


# 1. geometry

class ChannelWithCavity(Geometry):
    """Channel [0, length] x [0, height] with a cavity [x0, x1] x [-depth, 0].

    Inherited from :class:`Geometry`: inlet at x = 0, outlet at x = length,
    every other boundary facet is a wall unless :meth:`is_noslip_face` says
    otherwise. ``corner`` is the re-entrant corner of the corner-scale source
    terms; the meshes are graded towards it.
    """

    name, title = "cavity", "channel with a cavity"
    length, height = 4.0, 1.0
    corner = (1.5, 0.0)
    x0, x1, depth = 1.5, 2.5, 1.0
    # resolution -> number of cells per unit length (mesh size 1 / h)
    cells_per_unit = {"standard": 16, "fine": 32}

    def build_mesh(self, resolution: str) -> Mesh:
        """Return a structured mesh of the domain.

        The mesh of the bounding box is restricted to the cells in the domain
        (the corners of the domain are vertices of the box mesh).
        """
        n = self.cells_per_unit[resolution]
        box = RectangleMesh(Point(0.0, -self.depth),
                            Point(self.length, self.height),
                            int(round(n * self.length)),
                            int(round(n * (self.height + self.depth))),
                            "crossed")
        # cell midpoints
        x, y = box.coordinates()[box.cells()].mean(axis=1).T
        inside = (y > 0) | ((self.x0 < x) & (x < self.x1))
        markers = MeshFunction("size_t", box, 2, 0)
        markers.set_values(inside.astype(np.uintp))
        return SubMesh(box, markers, 1)

    def is_noslip_face(self, x: np.ndarray) -> bool:
        """Return whether ``x`` lies on a vertical face of the cavity.

        The free slip condition u_y = 0 of the walls holds only on horizontal
        walls.
        """
        return x[1] < TOL and (abs(x[0] - self.x0) < TOL
                               or abs(x[0] - self.x1) < TOL)


CAVITY = ChannelWithCavity()

# 2. benchmarks: geometry + boundary conditions + mesh resolution, e.g. also
#    Benchmark(CAVITY, NoSlip(), "fine"), Benchmark(CAVITY, SlipWallsOutflow())

BENCHMARKS = {b.name: b for b in [
    Benchmark(CAVITY, NoSlip()),
]}

# 3. studies: name (folder in results/ and figures/), benchmark, viscosity,
#    number of snapshots, numbers of training snapshots, also Stokes

STUDIES = {s.name: s for s in [
    Study("cavity/nu_1", BENCHMARKS["cavity_dirichlet"], 1.0, 750,
          (250, 500, 750), plot_fields=True, stokes=True),
]}

# 4. figures: path (in figures/) -> (title, curves [(label, study, N,
#    equations)], fits [(study, N, equations)]); varying_n: N = 250, 500,
#    750 and Stokes, the Navier-Stokes N = 750 and the Stokes errors fitted

FIGURES = {
    "cavity/cavity_rec_errors": varying_n(
        CAVITY.title + ", no-slip, $\\nu = 1$", "cavity/nu_1"),
}

if __name__ == "__main__":
    main(sys.argv[1:], BENCHMARKS, STUDIES, FIGURES, usage=__doc__)
