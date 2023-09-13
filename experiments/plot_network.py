from typing import Optional, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Colormap, ListedColormap
import matplotlib.pyplot as plt
import scienceplots


def plot_noisy_graph(
    G: nx.Graph,
    with_noisy_nodes: bool = False,
    index_node_signal: int = 0,
    index_edge_signal: int = 0,
    drop_isolated_nodes: bool = True,
    node_cmap: Colormap = plt.get_cmap("Reds"),
    edge_cmap: Colormap = plt.get_cmap("Blues"),
    vmin_node: Optional[float] = None,
    vmax_node: Optional[float] = None,
    vmin_edge: Optional[float] = None,
    vmax_edge: Optional[float] = None,
    same_clim: bool = False,
    alpha_noisy_nodes: float = 0.5,
    alpha_noisy_edges: float = 0.5,
    node_border: str = "black",
    node_border_noisy: str = "gray",
    offset: float = 0.15,
    layout_function=nx.kamada_kawai_layout,
    layout_all_nodes: bool = True,
    axis_off: bool = True,
    annotation: Optional[str] = None,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """Plot a graph with node and edge signals.

    Parameters
    ----------
    G :
        graph to plot. Can be obtained from
        :py:func:`~xaigraphon.graphs.Graph.to_nx`
    with_noisy_nodes :
        If True, show the noisy element differently than the non-noisy
        elements. Otherwise, just show the subgraph induced by the
        non-noisy nodes. By default :py:obj:`False` shows only the
        non-noisy nodes and edges.

        The noisy node and edges are shwon with a higher transparency
        than the non-noisy ones, which can be controlled with the
        :py:obj:`alpha_` arguments. The noisy edges are shown with a
        dashed line.
    index_node_signal :
        The node signal channel, by default plot the first
        one (by default 0).
    index_edge_signal :
        The edge signal channel, by default plot the first
        one (by default 0).
    drop_isolated_nodes :
        If True, drop the isolated nodes from the plot, by default
        :py:obj:`True`.
    node_cmap :
        colormap for the node signal, by default "Reds".
        To control the range, use :py:obj:`vmin_node` and :py:obj:`vmax_node`.
    edge_cmap :
        colormap for the edge signal, by default "Blues".
        To control the range, use :py:obj:`vmin_edge` and :py:obj:`vmax_edge`.
    same_clim :
        Force the edge and node colormap to have the same range, by default
        :py:obj:`False`.
    alpha_noisy_nodes :
        transparency of the noisy nodes, by default 0.5
    alpha_noisy_edges :
        transparency of the noisy edges, by default 0.5
    node_border :
        countour of the ndoes, by default "black" for non-noisy nodes.
        The noisy nodes are shown with a :py:obj:`node_border_noisy` countour.
    offset :
        offset to the colormap, by default 0.3.
        Instead of using the colormap directly, the colormap is darkened
        by mappping the input value to the range [offset, 1]. (used to avoid
        white colors in the colormap)
    layout_function :
        :py:mod:`networkx` layout, by default :py:func:`nx.kamada_kawai_layout`
        Can be any function that takes a graph as input and returns a
        dictionary of 2D coordinates keyed by node.
    layout_all_nodes :
        If True, use the layout function on the full graph, by default
        :py:obj:`False`. Otherwise, use the layout function on the subgraph
        that is plotted.
    **kwargs :
        additional arguments passed to :py:func:`networkx.draw_networkx`

    Returns
    -------
        Figure and Axes objects


    .. warning::
        This function expects the following node attributes:

        - :py:obj:`signal_{index_node_signal}`
        - :py:obj:`noisy_node_mask`

        And the following edge attributes:

        - :py:obj:`signal_{index_edge_signal}`
        - :py:obj:`noisy_edge_mask`

    """
    fig, ax, cax1, cax2 = _fig_with_2_colorbar_bottom(
        kwargs.pop("figsize", (6.4, 4.8)),
        kwargs.pop("fig", None),
        kwargs.pop("subplot_spec", None),
    )
    # darken cmap
    node_cmap = _darken_cmap(node_cmap, offset)
    edge_cmap = _darken_cmap(edge_cmap, offset)

    g_to_plot = (
        G.copy()
        if with_noisy_nodes
        else G.subgraph([n for n, d in G.nodes(data=True) if not d["noisy_node_mask"]]).copy()
    )

    if drop_isolated_nodes:
        g_to_plot.remove_nodes_from(list(nx.isolates(g_to_plot)))

    if layout_all_nodes:
        layout = layout_function(G)
        layout = {i: layout[i] for i in g_to_plot.nodes()}
    else:
        layout = layout_function(g_to_plot)

    options_node = {
        "node_size": kwargs.pop("node_size", 80),
        "node_color": list(nx.get_node_attributes(g_to_plot, f"signal_{index_node_signal}").values())
        if index_node_signal is not None
        else "blue",
        "alpha": [alpha_noisy_nodes if d["noisy_node_mask"] else 1 for _, d in g_to_plot.nodes(data=True)]
        if with_noisy_nodes
        else None,
        "cmap": node_cmap if index_node_signal is not None else None,
        "edgecolors": [
            node_border_noisy if d["noisy_node_mask"] else node_border for _, d in g_to_plot.nodes(data=True)
        ],
    }

    options_edges = {
        "width": 2,
        "edge_color": list(nx.get_edge_attributes(g_to_plot, f"signal_{index_edge_signal}").values())
        if index_edge_signal is not None
        else "black",
        "edge_cmap": edge_cmap if index_edge_signal is not None else None,
        "alpha": [alpha_noisy_edges if d["noisy_edge_mask"] else 1 for _, _, d in g_to_plot.edges(data=True)]
        if with_noisy_nodes
        else None,
        "style": ["dashed" if d["noisy_edge_mask"] else "solid" for _, _, d in g_to_plot.edges(data=True)]
        if with_noisy_nodes
        else "solid",
    }
    if index_node_signal is not None:
        if vmin_node is None:
            if len(options_node["node_color"]) > 0:
                vmin_node = np.min(options_node["node_color"])
        if vmax_node is None:
            if len(options_node["node_color"]) > 0:
                vmax_node = np.max(options_node["node_color"])
    if index_edge_signal is not None:
        if vmin_edge is None:
            if len(options_edges["edge_color"]) > 0:
                vmin_edge = np.min(options_edges["edge_color"])
        if vmax_edge is None:
            if len(options_edges["edge_color"]) > 0:
                vmax_edge = np.max(options_edges["edge_color"])
    if index_node_signal is not None and index_edge_signal is not None:
        if same_clim:
            vmin_node, vmax_node = min(
                vmin_node, vmin_edge), max(vmax_node, vmax_edge)
            vmin_edge, vmax_edge = vmin_node, vmax_node
    options_node["vmin"] = vmin_node
    options_node["vmax"] = vmax_node
    options_edges["edge_vmin"] = vmin_edge
    options_edges["edge_vmax"] = vmax_edge

    nodes = nx.draw_networkx_nodes(
        g_to_plot,
        pos=layout,
        ax=ax,
        **options_node,
        **kwargs,
    )
    edges = nx.draw_networkx_edges(
        g_to_plot,
        pos=layout,
        ax=ax,
        **options_edges,
        **kwargs,
    )

    # bug in nx, has to be reset manually
    if not isinstance(edges, list):
        edges.cmap = edge_cmap
        if vmin_edge is not None:
            edges.set_clim(vmin_edge, vmax_edge)
    if not isinstance(nodes, list):
        nodes.cmap = node_cmap
        if vmin_node is not None:
            nodes.set_clim(vmin_node, vmax_node)

    # correctly restrict colorbar to the range of the data
    # not with the offset so that lowest values are not white

    if annotation is not None:
        ax.annotate(
            annotation,
            xy=(0.5, 1.1),
            xycoords="axes fraction",
            ha="center",
            va="center",
        )

    if index_node_signal is not None:
        fig.colorbar(
            nodes,
            cax=cax1,
            orientation="horizontal",
            label=f"node signal {index_node_signal}",
        )
    else:
        cax1.axis("off")
    if index_edge_signal is not None and not isinstance(edges, list):
        fig.colorbar(
            edges,
            cax=cax2,
            orientation="horizontal",
            label=f"edge signal {index_edge_signal}",
        )
    else:
        cax2.axis("off")
    if axis_off:
        ax.axis("off")
    return fig, ax


def sciplot_style(x: np.ndarray,
                  y: list[np.ndarray],
                  p: list[str] = None,
                  llegend: Optional[dict] = None,
                  pparam: Optional[dict] = None,
                  style: Optional[str] = 'science',
                  fname: Optional[str] = None,
                  **kwargs):
    """Set the style from github.com/garrettj403/SciencePlots/."""
    with plt.style.context(style):
        plt.rc('text', usetex=False) #to get around the RuntimeError: LaTeX was not able to process the following string:
        fig, ax = plt.subplots()
        for i,val in enumerate(y):
            if p is None:
                ax.plot(x, val, label='Order {}'.format(i+1))
            else:
                ax.plot(x, val, label=p[i])
        if llegend is not None:
            ax.legend(**llegend)
            ax.set(**pparam)
        ax.autoscale(tight=True)
        if fname is not None:
            fig.savefig(fname, dpi=600)
        #plt.show()
    return fig, ax


def _fig_with_2_colorbar_bottom(figsize=(6.4, 4.8), fig=None, subplot_spec=None):
    if subplot_spec is None:
        fig = plt.figure(constrained_layout=True, figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[
                              1, 0.05], width_ratios=[1, 0.2, 1])
    else:
        if fig is None:
            fig = plt.gcf()
        gs = subplot_spec.subgridspec(
            2,
            2,
            height_ratios=[1, 0.05],
        )
    ax = fig.add_subplot(gs[0, :])
    cax1 = fig.add_subplot(gs[1, 0])
    cax2 = fig.add_subplot(gs[1, -1])
    return fig, ax, cax1, cax2


def _darken_cmap(cmap: Colormap, offset=0.2):
    return ListedColormap(cmap(np.linspace(offset, 1, 256)))
