from turtle import color
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure
from mpl_toolkits import axes_grid1

from pygraphon.graphons.GraphonFunction import Graphon
from pygraphon.graphons.StepGraphon import StepGraphon


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
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
    """Plot the graphon."""
    x1, x2 = np.meshgrid(np.arange(0, 1, 0.1), np.arange(0, 1, 0.1))
    y = graphon.graphon_function(x1, x2)
    ax.imshow(y, extent=[0, 1, 0, 1], cmap=plt.cm.get_cmap("jet"), origin="lower")
    fig.colorbar()
    plt.show()

    raise NotImplementedError


def plot_sample(
    graphon: Graphon,
    resolution=100,
    fig: Figure = None,
    ax: Axes = None,
    colorbar=False,
    integrate_to_1: bool = False,
    cmap = "binary",
) -> Tuple[Figure, Axes]:
    """Plot the graphon."""
    x1, x2 = np.linspace(0, 1, resolution, endpoint=False), np.linspace(
        0, 1, resolution, endpoint=False
    )
    result = np.zeros((x1.shape[0], x2.shape[0]))
    for i, x in enumerate(x1):
        for j, y in enumerate(x2):
            result[i, j] = graphon.graphon_function(x, y)
    if not integrate_to_1:
        result *= graphon.initial_rho

    im = ax.imshow(result, cmap=cmap, vmin=0, vmax=1)
    if colorbar:
        add_colorbar(im, ax=ax)
    _make_pretty(ax)
    return fig, ax


def plot(
    graphon: StepGraphon, fig: Figure = None, ax: Axes = None, figsize: Tuple[int, int] = (6, 5)
) -> Tuple[Figure, Axes]:
    """Plot the graphon.

    Args:
        graphon (Graphon): graphon to plot.
        fig (Figure, optional): Defaults to None.
        ax (Axes, optional): Defaults to None.
        figsize (Tuple[int, int], optional):. Defaults to (6, 5).

    Returns:
        Tuple[Figure, Axes]: figure and axis of the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_sample(graphon= graphon, fig = fig, ax = ax, colorbar = True)
    #sc = ax.imshow(graphon.graphon, cmap=plt.cm.get_cmap("jet"), aspect="auto")
    #fig.colorbar(sc, ax=ax)
    return fig, ax


def _make_pretty(ax):
    """Remove the figure frame, x- and y-ticks, and set the aspect to equal."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.get_figure().set_facecolor("w")
    ax.set_frame_on(False)
