def get_directed_edges(edge: tuple, arrows: list) -> list:
    assert len(arrows) >= 1
    if len(arrows) == 1:
        endVertex = arrows[0]
        startVertex = edge[int(not edge.index(endVertex))]
        return [(startVertex, endVertex)]
    else:
        return [edge, edge[::-1]]


def as_digraph(graph):
    digraph = graph.to_directed()
    drop_dummy_edges(graph, digraph)
    return digraph

def drop_dummy_edges(graph, digraph):
    edges_to_drop = []
    for edge, edge_metadata in digraph.edges.items():
        if edge not in list(graph.edges):
            edges_to_drop.append(edge)
    for edge in edges_to_drop:
        digraph.remove_edge(*edge)

def get_edges_ICstar(digraph):
    edges_ICstar = {'marked':[],
                    'undirected': [],
                    'directed': [],
                   }
    for edge, metadata in digraph.edges.items():
        # marked
        if metadata['marked']:
            edges_ICstar['marked'].append(get_directed_edges(edge, metadata['arrows'])[0])
        else:
            # undirected
            if len(metadata['arrows']) == 0:
                edges_ICstar['undirected'].append(edge)
            else:
                directed_edges = get_directed_edges(edge, metadata['arrows'])
                for e in directed_edges: edges_ICstar['directed'].append(e)
    return edges_ICstar

