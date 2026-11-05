"""All benchmarks, studies and figures of the paper."""
from knwins.benchmarks import (BackwardFacingStep, Benchmark, FlowPastCube,
                               NoSlip, NoSlipOutflow, SlipWallsOutflow)
from knwins.study import Study

CUBE, STEP = FlowPastCube(), BackwardFacingStep()

BENCHMARKS = {b.name: b for b in [
    Benchmark(CUBE, NoSlip()),
    Benchmark(CUBE, NoSlip(), "fine"),
    Benchmark(CUBE, NoSlipOutflow()),
    Benchmark(CUBE, SlipWallsOutflow()),
    Benchmark(STEP, NoSlip()),
    Benchmark(STEP, NoSlip(), "coarse"),
    Benchmark(STEP, NoSlipOutflow()),
    Benchmark(STEP, SlipWallsOutflow()),
]}

# name, benchmark, viscosity, number of snapshots, numbers of training
# snapshots; the studies with N = 250, 500, 750 also solve the Stokes
# equations for the same source terms
STUDIES = {s.name: s for s in [
    Study("cube/nu_1", BENCHMARKS["cube_dirichlet"], 1.0, 750,
          (250, 500, 750), plot_fields=True, stokes=True),
    Study("cube/nu_0.5", BENCHMARKS["cube_dirichlet"], 0.5, 500, (500,)),
    Study("cube/nu_0.1", BENCHMARKS["cube_dirichlet"], 0.1, 500, (500,)),
    Study("cube/fine", BENCHMARKS["cube_dirichlet_fine"], 1.0, 500, (500,)),
    Study("cube_mixed_bc", BENCHMARKS["cube_mixed"], 1.0, 750,
          (250, 500, 750), plot_fields=True, stokes=True),
    Study("cube_mixed_slip", BENCHMARKS["cube_slip"], 1.0, 750,
          (250, 500, 750), plot_fields=True, stokes=True),
    Study("backward/nu_1", BENCHMARKS["backward_dirichlet"], 1.0, 750,
          (250, 500, 750), plot_fields=True, stokes=True),
    Study("backward/nu_0.5", BENCHMARKS["backward_dirichlet"], 0.5, 500,
          (500,)),
    Study("backward/nu_0.1", BENCHMARKS["backward_dirichlet"], 0.1, 500,
          (500,)),
    Study("backward/coarse", BENCHMARKS["backward_dirichlet_coarse"], 1.0,
          750, (750,)),
    Study("backward_mixed_bc", BENCHMARKS["backward_mixed"], 1.0, 750,
          (250, 500, 750), plot_fields=True, stokes=True),
    Study("backward_mixed_slip", BENCHMARKS["backward_slip"], 1.0, 750,
          (250, 500, 750), plot_fields=True, stokes=True),
]}


def varying_n(title: str, study: str) -> tuple:
    """Return the figure of a study for N = 250, 500, 750, with Stokes.

    The Navier-Stokes N = 750 and the Stokes errors are fitted.

    Args:
        title: figure title.
        study: name of the study.
    """
    curves = [("INS, N = {}".format(n), study, n, "ins")
              for n in (250, 500, 750)]
    curves.append(("Stokes, N = 750", study, 750, "stokes"))
    return title, curves, [(study, 750, "ins"), (study, 750, "stokes")]


# figure path (in figures/) -> (title, curves [(label, study, N, equations)],
# fits [(study, N, equations)])
FIGURES = {}
for g, geometry in (("cube", CUBE), ("backward", STEP)):
    n_mesh = {"cube": 500, "backward": 750}[g]
    # (coarser, finer) mesh; the finer is fitted
    mesh_studies = {"cube": (("standard", "cube/nu_1"), ("fine", "cube/fine")),
                    "backward": (("coarse", "backward/coarse"),
                                 ("standard", "backward/nu_1"))}[g]
    FIGURES.update({
        "{0}/{0}_rec_errors".format(g): varying_n(
            geometry.title + ", no-slip, $\\nu = 1$", g + "/nu_1"),
        "{0}/{0}_mesh_rec_errors".format(g): (
            geometry.title + ", no-slip, $\\nu = 1$, N = {}".format(n_mesh),
            [("INS, {} mesh".format(mesh), study, n_mesh, "ins")
             for mesh, study in mesh_studies],
            [(mesh_studies[1][1], n_mesh, "ins")]),
        "{0}/{0}_viscosity_rec_errors".format(g): (
            geometry.title + ", no-slip, N = 500",
            [("INS, $\\nu$ = {}".format(nu), g + "/nu_" + nu, 500, "ins")
             for nu in ("1", "0.5", "0.1")],
            [(g + "/nu_1", 500, "ins")]),
        "{0}_mixed_bc/{0}_mixed_bc_rec_errors".format(g): varying_n(
            geometry.title + ", no-slip walls and outflow, $\\nu = 1$",
            g + "_mixed_bc"),
        "{0}_mixed_slip/{0}_mixed_slip_rec_errors".format(g): varying_n(
            geometry.title + ", slip walls and outflow, $\\nu = 1$",
            g + "_mixed_slip"),
    })
