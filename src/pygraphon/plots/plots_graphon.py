"""Utilities to plot graphons."""
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes

from pygraphon.graphons.Graphon import Graphon

from .utils import _add_colorbar


def plot_graphon_function(
    graphon: Graphon,
    res=0.01,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
    cmap: str = "turbo",
    origin: str = "lower",
    show_colorbar: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot a graphon function.

    Parameters
    ----------
    graphon : Graphon
        graphon to plot
    res : float
        resolution of the plot, by default 0.01
    fig : Figure
        matplotlib figure, by default None
    ax : Axes
        maplotlib ax, by default None
    figsize : Tuple[int, int]
        figure size, by default (6, 5)
    cmap : str
        matplotlib colormap, by default "turbo"
    origin : str
        origin of the plot, by default "lower"
    show_colorbar : bool
        whether to show the colorbar, by default False

    Returns
    -------
    Tuple[Figure, Axes]
        graphon function evaluated on a grid.
    """
    res = _validate_res(res)
    fig, ax = _default_fig_ax(fig, ax, figsize)
    y = _evaluate_graphon(graphon, res=res)
    fig, ax = _show_evaluated_graphon(
        y, fig, ax, cmap=cmap, origin=origin, show_colorbar=show_colorbar
    )
    return fig, ax


def plot_probabilities(
    graphon: Graphon,
    res=100,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
    cmap: str = "turbo",
    origin: str = "lower",
    show_colorbar: bool = False,
    vmin: float = 0,
    vmax: float = 1,
) -> Tuple[Figure, Axes]:
    """Plot the probabilities from a graphon.

    Parameters
    ----------
    graphon : Graphon
        graphon to plot
    res : float
        number of points to evaluate the graphon, by default 100.
    fig : Figure
        matplotlib figure, by default None
    ax : Axes
        maplotlib ax, by default None
    figsize : Tuple[int, int]
        figure size, by default (6, 5)
    cmap : str
        matplotlib colormap, by default "turbo"
    origin : str
        origin of the plot, by default "lower"
    show_colorbar : bool
        whether to show the colorbar, by default False
    vmin : float
        minimum value of the colorbar, by default 0
    vmax : float
        maximum value of the colorbar, by default 1

    Returns
    -------
    Tuple[Figure, Axes]
        graphon function evaluated on a grid and normalized.


    .. note::
        The difference between this function and :func:`plot_graphon_function`
        is that this function normalizes the graphon function by its initial
        density so that the plotted values are between 0 and 1.
    """
    res = _validate_res(res)
    fig, ax = _default_fig_ax(fig, ax, figsize)
    y = _evaluate_graphon(graphon, res=res) * graphon.initial_rho
    fig, ax = _show_evaluated_graphon(
        y,
        fig,
        ax,
        cmap=cmap,
        origin=origin,
        show_colorbar=show_colorbar,
        vmin=vmin,
        vmax=vmax,
    )
    return fig, ax


def _validate_res(res: float) -> float:
    if res <= 0:
        raise ValueError("resolution must be positive")
    elif res > 1:
        return 1 / res
    return res


def _evaluate_graphon(graphon: Graphon, res=0.01) -> np.ndarray:
    x1, x2 = np.meshgrid(np.arange(0, 1, res), np.arange(0, 1, res))
    return graphon.graphon_function(x1, x2)


def _default_fig_ax(
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (6, 5),
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _show_evaluated_graphon(
    f: np.ndarray,
    fig: Figure,
    ax: Axes,
    cmap: str = "turbo",
    origin: str = "lower",
    show_colorbar: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    extent = [0, 1, 0, 1] if origin == "lower" else [0, 1, 1, 0]
    im = ax.imshow(
        f,
        extent=extent,
        cmap=mpl.colormaps[cmap],
        origin=origin,
        vmin=vmin,
        vmax=vmax,
    )
    if show_colorbar:
        _add_colorbar(im, ax=ax)
    return fig, ax
