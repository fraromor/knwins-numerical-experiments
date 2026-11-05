"""Command line interface.

Commands::

    python run.py constants [benchmark ...]  # coercivity / continuity
    python run.py study <study> [...]        # snapshots and POD errors
    python run.py errors <study> [...]       # POD errors (saved snapshots)
    python run.py fields <study> [...]       # field plots (saved snapshots)
    python run.py figures                    # decay plots and README table

Without names, every benchmark / study of experiments.py is run (``fields``:
the studies with ``plot_fields``). template.py has the same interface for its
own benchmarks and studies.
"""
import os
import sys
from typing import Dict, List, Tuple

from dolfin import set_log_level

from experiments import BENCHMARKS, FIGURES, STUDIES
from knwins.benchmarks import Benchmark
from knwins.constants import load_or_compute_constants
from knwins.plotting import NAMES, plot_decay
from knwins.pod import above_roundoff, fit_window
from knwins.study import FIGURES as FIGURE_DIR, RESULTS, Study

set_log_level(40)

#: figure path (in figures/) -> (title, curves [(label, study, N,
#: equations)], fits [(study, N, equations)])
FigureSpecs = Dict[str, Tuple[str, List[Tuple[str, str, int, str]],
                              List[Tuple[str, int, str]]]]


def constants(benchmarks: Dict[str, Benchmark], names: List[str]) -> None:
    """Recompute the constants of the named benchmarks (all if empty)."""
    for name in names or benchmarks:
        load_or_compute_constants(benchmarks[name],
                                  os.path.join(RESULTS, "constants"),
                                  recompute=True)


def figures(studies: Dict[str, Study], figure_specs: FigureSpecs) -> None:
    """Plot the decay of ``figure_specs``; print the table of the fits.

    The errors are plotted and fitted up to the round-off level
    (:func:`knwins.pod.above_roundoff`), on the windows of
    :func:`knwins.pod.fit_window`.
    """
    def errors(study: str, n: int, equations: str) -> tuple:
        return tuple(above_roundoff(e)
                     for e in studies[study].errors(n, equations))

    print("| figure | equations | field | fit on n in | a | b | c | rms |")
    print("|---|---|---|---|---|---|---|---|")
    for path, (title, curves, fits) in figure_specs.items():
        os.makedirs(os.path.dirname(os.path.join(FIGURE_DIR, path)),
                    exist_ok=True)
        data = [(label, *errors(study, n, equations), equations)
                for label, study, n, equations in curves]
        fitted = []
        for study, n, equations in fits:
            eps_u, eps_p = errors(study, n, equations)
            fitted.append((equations, eps_u, eps_p,
                           fit_window(eps_u, equations),
                           fit_window(eps_p, equations)))
        rows = plot_decay(data, fitted, title,
                          os.path.join(FIGURE_DIR, path))
        for equations, field, window, a, b, c, rms in rows:
            print("| {} | {} | {} | [{}, {}] | {} | {:.2f} | {:.1f} | "
                  "{:.2f} |".format(os.path.basename(path), NAMES[equations],
                                    field, *window,
                                    "1/3" if a < 0.4 else "1/2", b, c, rms))


def main(argv: List[str], benchmarks: Dict[str, Benchmark],
         studies: Dict[str, Study], figure_specs: FigureSpecs,
         usage: str = __doc__) -> None:
    """Run a command on the named benchmarks / studies.

    Args:
        argv: the command, then the names (all if none).
        benchmarks: name -> benchmark.
        studies: name -> study.
        figure_specs: the decay plots.
        usage: message of an unknown command.
    """
    command, names = (argv[0], argv[1:]) if argv else (None, [])
    if command == "constants":
        constants(benchmarks, names)
    elif command == "study":
        for name in names or studies:
            studies[name].run()
    elif command == "errors":
        for name in names or studies:
            studies[name].compute_errors()
    elif command == "fields":
        for name in names or [s for s in studies if studies[s].plot_fields]:
            studies[name].plot_saved_snapshot()
    elif command == "figures":
        figures(studies, figure_specs)
    else:
        sys.exit(usage)


if __name__ == "__main__":
    main(sys.argv[1:], BENCHMARKS, STUDIES, FIGURES)
