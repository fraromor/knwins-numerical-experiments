"""Figures: snapshot fields and decay of the POD projection errors."""
from __future__ import annotations

from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (after the backend)
from matplotlib import cm  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
from dolfin import Function, plot  # noqa: E402

from .pod import fit_decay  # noqa: E402

plt.rcParams.update({"font.size": 14})
MARKERS = "os^vp"


def save(path: str) -> None:
    """Save the current figure as pdf (for the paper) and png (for the README).

    Args:
        path: output path without extension.
    """
    plt.savefig(path + ".pdf", bbox_inches="tight")
    plt.savefig(path + ".png", dpi=80, bbox_inches="tight")
    plt.close()


def plot_field(field: Function, title: str, path: str,
               cmap: str = "viridis") -> None:
    """Plot a scalar field with a horizontal colorbar.

    Args:
        field: scalar dolfin function.
        title: figure title.
        path: output path without extension.
        cmap: matplotlib colormap.
    """
    plt.figure(figsize=(10, 4))
    p = plot(field, cmap=cmap)
    cb = plt.colorbar(p, orientation="horizontal",
                      format=ticker.FormatStrFormatter("%.2g"))
    cb.locator = ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    plt.title(title)
    plt.tight_layout()
    save(path)


#: decay fits exp(-b n^a - c)
EXPONENTS = (1 / 3, 1 / 2)
#: reference lines (linestyle, width) of the fits: n^{1/3} red, n^{1/2}
#: bright blue (drawn on top); Navier-Stokes solid / dashed, Stokes dash-dot
#: / dotted
REFERENCES = {("ins", 1 / 3): ("-", 3.2), ("ins", 1 / 2): ((0, (5, 3)), 2.4),
              ("stokes", 1 / 3): ("-.", 3.0), ("stokes", 1 / 2): (":", 2.8)}
REFERENCE_COLORS = {1 / 3: "#e8000b", 1 / 2: "#00b4ff"}
#: the reference lines are drawn shifted off the data: Navier-Stokes up,
#: Stokes down
REFERENCE_SHIFT = {"ins": 10.0, "stokes": 0.1}
NAMES = {"ins": "INS", "stokes": "Stokes"}
TICKS = (1, 10, 20, 50, 100, 200, 300, 500, 700)

#: a curve: (label, eps_u, eps_p, equations)
Curve = Tuple[str, np.ndarray, np.ndarray, str]
#: a fit: (equations, eps_u, eps_p, window_u, window_p)
Fit = Tuple[str, np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]


def _markers(n_points: int, count: int = 28) -> List[int]:
    """Return marker positions evenly spaced in n^{1/3}."""
    x = np.linspace(1, n_points**(1 / 3), count)**3
    return sorted(set(np.clip(np.round(x).astype(int) - 1, 0, n_points - 1)))


def _styles(curves: List[Curve]) -> List[Tuple[object, object, str]]:
    """Return the velocity color, pressure color and marker of the curves.

    Navier-Stokes: shades of blue (velocity) and grey (pressure), darker for
    later curves; Stokes: orange and green, diamonds.
    """
    n_ins = sum(curve[3] == "ins" for curve in curves)
    shades = iter(np.linspace(0.95 if n_ins == 1 else 0.45, 0.95, n_ins))
    markers = iter(MARKERS)
    styles = []
    for curve in curves:
        if curve[3] == "ins":
            x = next(shades)
            styles.append((cm.Blues(x), cm.Greys(x), next(markers)))
        else:
            styles.append(("tab:orange", "tab:green", "D"))
    return styles


def plot_decay(curves: List[Curve], fits: List[Fit], title: str,
               path: str) -> List[Tuple[str, str, Tuple[int, int], float,
                                        float, float, float]]:
    r"""Plot the errors against :math:`n^{1/3}` with the decay fits.

    In this scale :math:`\exp(-b n^{1/3})` is a straight line and
    :math:`\exp(-b n^{1/2})` is concave. Every fit draws the references
    :math:`\exp(-b n^a - c)`, :math:`a = 1/3, 1/2`, on its window, with
    :math:`b` and the rms misfit of :math:`\log \varepsilon` in the legend.

    Args:
        curves: the plotted errors.
        fits: the fitted errors and their windows.
        title: figure title.
        path: output path without extension.

    Returns:
        The fits: ``(equations, field, window, a, b, c, rms)``, field
        ``'u'`` or ``'p'``.
    """
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 7.8))
    fig.suptitle(title, fontsize=17)
    panels = [("velocity", r"$\varepsilon_u(n)$, $H^1$ norm"),
              ("pressure", r"$\varepsilon_p(n)$, $L^2$ norm")]
    rows, n_max, lows = [], 0, []
    for k, (ax, (field, ylabel)) in enumerate(zip(axes, panels)):
        for (label, eps_u, eps_p, _), style in zip(curves, _styles(curves)):
            eps = (eps_u, eps_p)[k]
            n = np.arange(1, len(eps) + 1)
            n_max, lows = max(n_max, len(eps)), lows + [eps.min()]
            ax.semilogy(n**(1 / 3), eps, color=style[k], linewidth=1.6,
                        marker=style[2], markevery=_markers(len(eps)),
                        markersize=7, fillstyle="none", markeredgewidth=1.4,
                        label=label, zorder=2)
        for equations, eps_u, eps_p, window_u, window_p in fits:
            eps, window = ((eps_u, window_u), (eps_p, window_p))[k]
            n = np.arange(window[0], int(window[1]) + 1)
            for a in EXPONENTS:
                b, c, rms = fit_decay(eps, a, window)
                rows.append((equations, "up"[k], window, a, b, c, rms))
                style, width = REFERENCES[equations, a]
                ax.semilogy(
                    n**(1 / 3), REFERENCE_SHIFT[equations] * np.exp(
                        -b * n**a - c),
                    color=REFERENCE_COLORS[a], linestyle=style,
                    linewidth=width, zorder=3 if a < 0.4 else 4,
                    label=r"{} fit $e^{{-b n^{{{}}} - c}}$: b={:.2f}, "
                          r"rms={:.2f}".format(NAMES[equations],
                                               "1/3" if a < 0.4 else "1/2",
                                               b, rms))
        ax.set_title(field)
        ax.set_xlabel("number of POD modes $n$")
        ax.set_ylabel(ylabel)
        ax.grid(which="major", linestyle="dotted", linewidth=1)
    ticks = np.array([t for t in TICKS if t <= n_max])
    lo = np.floor(np.log10(min(lows)))
    first = max(max(curve[1][0], curve[2][0]) for curve in curves)
    hi = np.ceil(np.log10(2 * first))
    for ax in axes:
        ax.set_xticks(ticks**(1 / 3))
        ax.set_xticklabels(ticks)
        ax.set_ylim(10**(lo - 0.3), 10**(hi + 0.2))
        ax.set_yticks(10.0**np.arange(lo, hi + 1, 2 if hi - lo > 10 else 1))
        ax.yaxis.set_tick_params(labelleft=True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1,
                  fontsize=12.5, frameon=False)
    fig.tight_layout()
    save(path)
    return rows
