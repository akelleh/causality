import numpy as np
import pandas as pd
from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

from causality.inference import drop_dummy_edges
from causality.inference import get_directed_edges
from causality.inference import get_edges_ICstar
from causality.inference import as_digraph

from networkx.classes.digraph import DiGraph

def _make_DAG():
    # generate some toy data:
    np.random.seed(1)
    SIZE = 2000
    x1 = np.random.normal(size=SIZE)
    x2 = x1 + np.random.normal(size=SIZE)
    x3 = x1 + np.random.normal(size=SIZE)
    x4 = x2 + x3 + np.random.normal(size=SIZE)
    x5 = x4 + np.random.normal(size=SIZE)

    X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

    # define the variable types: 'c' is 'continuous'.  The variables defined here
    # are the ones the search is performed over  -- NOT all the variables defined
    # in the data frame.
    variable_types = {'x1': 'c', 'x2': 'c', 'x3': 'c', 'x4': 'c', 'x5': 'c'}

    # run the search
    ic_algorithm = IC(RobustRegressionTest)
    graph = ic_algorithm.search(X, variable_types)
    return graph


def test_get_directed_edges():
    edge = ('a', 'b')
    arrows = ['a']
    directed_edges = get_directed_edges(edge, arrows)
    assert directed_edges == [('b', 'a')]

    arrows = ['b']
    directed_edges = get_directed_edges(edge, arrows)
    assert directed_edges == [('a', 'b')]

    arrows = ['a','b']
    directed_edges = get_directed_edges(edge, arrows)
    assert ('a', 'b') in directed_edges
    assert ('b', 'a') in directed_edges
    assert len(directed_edges) == 2


def test_as_digraph():
    graph = _make_DAG()
    digraph = as_digraph(graph)
    for edge, metadata in graph.edges.items():
        assert edge in digraph.edges
    assert isinstance(digraph, DiGraph)


def test_drop_dummy_edges():
    graph = _make_DAG()
    digraph = as_digraph(graph)
    drop_dummy_edges(graph, digraph)
    assert set(graph.edges) == set(digraph.edges)
    return digraph


def test_get_edges_ICstar():
    digraph = test_drop_dummy_edges()
    edges_ICstar = get_edges_ICstar(digraph)
    assert edges_ICstar['marked'] == [('x4', 'x5')]
    assert set(edges_ICstar['undirected']) == {('x1', 'x2'), ('x1', 'x3')}
    assert set(edges_ICstar['directed']) == {('x2', 'x4'), ('x3', 'x4')}
