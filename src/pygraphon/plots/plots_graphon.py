"""Utilities to plot graphons."""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure
from mpl_toolkits import axes_grid1

from pygraphon.graphons.Graphon import Graphon
from pygraphon.graphons.StepGraphon import StepGraphon

from .utils import make_0_1


# noqa: DAR101
def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def plot_graphon_function(
    graphon: Graphon, fig: Figure = None, ax: Axes = None, figsize: Tuple[int, int] = (6, 5)
) -> Tuple[Figure, Axes]:
    """Plot a graphon function.

    Parameters
    ----------
    graphon : Graphon
        graphon defined with a function.
    fig : Figure
        matplotlib figure, by default None
    ax : Axes
        maplotlib ax, by default None
    figsize : Tuple[int, int], optional
        figure size, by default (6, 5)

    Returns
    -------
    Tuple[Figure, Axes]
        plot
    """
    x1, x2 = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    y = graphon.graphon_function(x1, x2)
    ax.imshow(y, extent=[0, 1, 0, 1], cmap=plt.cm.get_cmap("jet"), origin="lower")
    fig.colorbar()
    plt.show()


def plot_sample(
    graphon: Graphon,
    resolution=100,
    fig: Figure = None,
    ax: Axes = None,
    colorbar=False,
    integrate_to_1: bool = False,
    colormap: str = "binary",
) -> Tuple[Figure, Axes]:
    """Plot a sample of the graphon.

    Parameters
    ----------
    graphon : Graphon
        graphon
    resolution : int
        number of samples to take, by default 100
    fig : Figure
        figure, by default None
    ax : Axes
        ax, by default None
    colorbar : bool
        if yes add a colorbar, by default False
    integrate_to_1 : bool
        if False plot with original edge density, by default False

    Returns
    -------
    Tuple[Figure, Axes]
        plotted sample
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    x1, x2 = np.linspace(0, 1, resolution, endpoint=False), np.linspace(
        0, 1, resolution, endpoint=False
    )
    result = np.zeros((x1.shape[0], x2.shape[0]))
    for i, x in enumerate(x1):
        for j, y in enumerate(x2):
            result[i, j] = graphon.graphon_function(x, y)
    if integrate_to_1:
        vmin = 0
        vmax = 1
    else:
        result *= graphon.initial_rho
        vmin = None
        vmax = None

    im = ax.imshow(result, cmap=colormap, vmin=vmin, vmax=vmax)
    if colorbar:
        _add_colorbar(im, ax=ax)
    return fig, ax


def plot(
    graphon: StepGraphon,
    fig: Figure = None,
    ax: Axes = None,
    figsize: Tuple[int, int] = (6, 5),
    colormap: str = "binary",
    integrate_to_1: bool = False,
    show_colorbar: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot the graphon.

    Parameters
    ----------
    graphon : StepGraphon
         graphon to plot.
    fig : Figure
        figure, by default None
    ax : Axes
        ax, by default None
    figsize : Tuple[int, int]
        figsize, by default (6, 5)

    Returns
    -------
    Tuple[Figure, Axes]
        plot
    """
    fig, ax = plot_sample(
        graphon,
        fig=fig,
        ax=ax,
        colorbar=show_colorbar,
        integrate_to_1=integrate_to_1,
        colormap=colormap,
    )
    make_0_1(ax)
    return fig, ax
