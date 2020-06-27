import networkx as nx
import matplotlib.pyplot as plt

def _drop_dummy_edges(graph, digraph):
    edges_to_drop = []
    for edge, edge_metadata in digraph.edges.items():
        if edge not in list(graph.edges):
            edges_to_drop.append(edge)
    for edge in edges_to_drop:
        digraph.remove_edge(*edge)


def _split_marked_edges(digraph):
    marked_edges = []
    unmarked_edges = []
    for edge, edge_metadata in digraph.edges.items():
        if edge_metadata['marked']:
            marked_edges.append(edge)
        else:
            unmarked_edges.append(edge)
    return marked_edges, unmarked_edges


def plot_DAG(graph, plot_attributes=None):
    if plot_attributes is None:
        plot_attributes = {}

    unmarked_edge_color = plot_attributes.get('unmarked_edge_color', 'black')
    marked_edge_color = plot_attributes.get('unmarked_edge_color', 'red')
    arrowsize = plot_attributes.get('arrowsize', 25)

    digraph = graph.to_directed()

    _drop_dummy_edges(graph, digraph)

    marked_edges, unmarked_edges = _split_marked_edges(digraph)

    pos = nx.spring_layout(digraph)
    nx.draw_networkx_nodes(digraph, pos)
    nx.draw_networkx_labels(digraph, pos)
    nx.draw_networkx_edges(digraph, pos, arrows=True, edgelist=unmarked_edges,
                           edge_color=unmarked_edge_color,
                           arrowsize=arrowsize)
    nx.draw_networkx_edges(digraph, pos, arrows=True, edgelist=marked_edges,
                           edge_color=marked_edge_color, arrowsize=arrowsize)
    plt.axis('off')
    return pos, digraph
