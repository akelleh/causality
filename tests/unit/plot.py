from causality.plot import plot_marked_partially_directed_graph
from networkx.classes.digraph import DiGraph

import numpy
import pandas as pd

from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest



def _make_DAG():
    # generate some toy data:
    SIZE = 2000
    x1 = numpy.random.normal(size=SIZE)
    x2 = x1 + numpy.random.normal(size=SIZE)
    x3 = x1 + numpy.random.normal(size=SIZE)
    x4 = x2 + x3 + numpy.random.normal(size=SIZE)
    x5 = x4 + numpy.random.normal(size=SIZE)

    X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

    # define the variable types: 'c' is 'continuous'.  The variables defined here
    # are the ones the search is performed over  -- NOT all the variables defined
    # in the data frame.
    variable_types = {'x1': 'c', 'x2': 'c', 'x3': 'c', 'x4': 'c', 'x5': 'c'}

    # run the search
    ic_algorithm = IC(RobustRegressionTest)
    graph = ic_algorithm.search(X, variable_types)
    return graph

def test_plot_DAG():
    graph = _make_DAG()
    pos, digraph, edges_ICstar = plot_marked_partially_directed_graph(graph)
    assert isinstance(pos, dict)
    assert isinstance(digraph, DiGraph)
    assert isinstance(edges_ICstar, dict)
    assert 'marked' in edges_ICstar
    assert 'directed' in edges_ICstar
    assert 'undirected' in edges_ICstar
