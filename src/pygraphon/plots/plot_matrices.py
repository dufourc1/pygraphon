"""Utilities to plot matrices."""
import matplotlib.pyplot as plt
import numpy as np


def spy(A: np.ndarray, fig=None, ax=None, marker_size=0.1, **kwargs):
    """Spy function for matplotlib.pyplot.

    Parameters
    ----------
    A : np.ndarray
         matrix to plot.
    fig : Figure, optional
        figure from matplotlib, by default None
    ax : Axes, optional
        ax from matplotlib, by default None
    marker_size : float, optional
        marker size, by default 0.1

    Returns
    -------
    Figure,Axes
        figure and ax from matplotlib.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.spy(A, markersize=marker_size, **kwargs)
    return fig, ax
