import networkx as nx
import matplotlib.pyplot as plt

from causality.inference import get_edges_ICstar
from causality.inference import as_digraph

def plot_marked_partially_directed_graph(graph, plot_attributes=None):
    if plot_attributes is None:
        plot_attributes = {}

    unmarked_edge_color = plot_attributes.get('unmarked_edge_color', 'black')
    marked_edge_color = plot_attributes.get('unmarked_edge_color', 'black')
    arrowsize = plot_attributes.get('arrowsize', 25)

    digraph = as_digraph(graph)
    edges_ICstar = get_edges_ICstar(digraph)

    pos = nx.spring_layout(digraph)
    nx.draw_networkx_nodes(digraph, pos)
    nx.draw_networkx_labels(digraph, pos)

    # directed edges
    nx.draw_networkx_edges(digraph, pos, arrows=True, edgelist=edges_ICstar['directed'],
                           edge_color=unmarked_edge_color,
                           arrowsize=arrowsize)
    # undirected edges
    nx.draw_networkx_edges(digraph, pos, arrows=False, edgelist=edges_ICstar['undirected'],
                           edge_color=unmarked_edge_color,
                           arrowsize=arrowsize)

    # marked edges
    nx.draw_networkx_edges(digraph, pos, arrows=True, edgelist=edges_ICstar['marked'],
                           edge_color=marked_edge_color,
                           arrowsize=arrowsize)
    nx.draw_networkx_edge_labels(digraph, pos, arrows=True, edgelist=edges_ICstar['marked'],
                                 edge_labels={e: '*' for e in edges_ICstar['marked']}, arrowsize=arrowsize)
    plt.axis('off')
    return pos, digraph, edges_ICstar
