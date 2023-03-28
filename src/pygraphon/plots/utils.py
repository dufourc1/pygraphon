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
    ax.set_frame_on(False)
