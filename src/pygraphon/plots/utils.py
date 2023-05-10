import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1


def make_pretty(ax):
    """Remove the figure frame, x- and y-ticks, and set the aspect to equal.

    Parameters
    ----------
    ax :  Axes
        matplotlib ax, by default None
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.get_figure().set_facecolor("w")
    ax.set_frame_on(False)


def make_0_1(ax):
    """Set the x- and y-ticks to range from 0 to 1.

    Parameters
    ----------
    ax :  Axes
        matplotlib ax, by default None
    """
    ticks = ax.get_xticks()
    xticks = [0.01, ticks[-1] / 2, ticks[-1] - 0.01]
    ax.set_xticks(xticks)
    ax.set_yticks(xticks)
    ax.set_xticklabels([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1])


# noqa: DAR201
def _add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot.

    Parameters
    ----------
    im : matplotlib.cm.ScalarMappable
        image plot
    aspect : int, optional
         by default 20
    pad_fraction : float, optional
         by default 0.5

    Returns
    -------
    matplotlib.colorbar.Colorbar
        colorbar added to the image plot
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
