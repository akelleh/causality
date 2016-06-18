from tests.unit import TestAPI
import itertools

import numpy.random
import pandas as pd
import networkx as nx

from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

TEST_SET_SIZE = 2000

class Test_IC(TestAPI):

    def setUp(self):
        x1 = numpy.random.normal(size=TEST_SET_SIZE)
        x2 = x1 + numpy.random.normal(size=TEST_SET_SIZE)
        x3 = x1 + numpy.random.normal(size=TEST_SET_SIZE)
        x4 = x2 + x3 + numpy.random.normal(size=TEST_SET_SIZE)
        x5 = x4 + numpy.random.normal(size=TEST_SET_SIZE)
        self.X = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3, 'x4' : x4, 'x5' : x5})
        self.variable_types = {'x1' : 'c', 'x2' : 'c', 'x3' : 'c', 'x4' : 'c', 'x5' : 'c'}
        self.true_neighbors = { 'x1' : set(['x2','x3']),
                                'x2' : set(['x1','x4']),
                                'x3' : set(['x1','x4']),
                                'x4' : set(['x2','x3','x5']),
                                'x5' : set(['x4'])}
        self.true_colliders = set([('x3','x4'), ('x2','x4')])
        self.true_marked = set([('x4','x5')])
        self.ic = IC(RobustRegressionTest)
        self.ic.search(self.X, self.variable_types)

    def test_build_g(self):
        self.ic._build_g(self.variable_types)
        V = len(self.X.columns)
        assert(len(self.ic._g.edges()) == (V-1)*V / 2) 
        assert(set(self.ic._g.nodes()) == set(self.variable_types.keys()))
        for node, variable_type in self.variable_types.items():
            assert(self.ic._g.node[node]['type'] == variable_type)
        for i, j in self.ic._g.edges():
            assert(self.ic._g.edge[i][j]['marked'] == False)

    def test_find_skeleton(self):
        self.ic._build_g(self.variable_types)
        self.ic._find_skeleton(self.X, self.variable_types)
        for node, neighbors in self.true_neighbors.items():
            assert(set(self.ic._g.neighbors(node)) == neighbors)
            
    def test_orient_colliders(self):
        self.ic._build_g(self.variable_types)
        self.ic._find_skeleton(self.X, self.variable_types)
        self.ic._orient_colliders()
        for i, j in self.ic._g.edges():
            measured_colliders = self.ic._g.edge[i][j]['arrows']
            if len(measured_colliders) > 0:
                if j in measured_colliders:
                    assert((i,j) in self.true_colliders)
                else:
                    assert((j,i) in self.true_colliders)
            else:
                assert((i,j) not in self.true_colliders and (j,i) not in self.true_colliders)

    def test_separating_set(self):
        self.ic._build_g(self.variable_types)
        self.ic._find_skeleton(self.X, self.variable_types)
        for xi, xj in itertools.combinations(self.variable_types.keys(), 2):
            if not self.ic._g.has_edge(xi,xj):
                if (xi,xj) in self.ic.separating_sets:
                    z = self.ic.separating_sets[(xi,xj)]
                else:
                    z = self.ic.separating_sets[(xj,xi)]
                test = self.ic.independence_test([xj],[xi], list(z), self.X, self.ic.alpha)
                assert(test.independent())

    def test_marked_directed_path(self):
        marked_edges = [('a','b'),('b','c'),('c','d')]
        unmarked_edges = [('a','d')]
        nodes = ['a','b','c','d']
        g = nx.Graph()
        g.add_edges_from(marked_edges, marked=True)
        g.add_edges_from(unmarked_edges, marked=False)
        for i, j in (marked_edges + unmarked_edges):
            g.edge[i][j]['arrows'] = [j]
        self.ic._g = g
        assert(self.ic._marked_directed_path('a','d'))
        assert(not self.ic._marked_directed_path('d','a'))

    def test_recursion_rule_1(self):
        pass

    def test_recursion_rule_2(self):
        pass

    def test_search(self):
        self.ic.search(self.X, self.variable_types)
        for i, j in self.ic._g.edges():
            if self.ic._g.edge[i][j]['marked']:
                assert( (i,j) in self.true_marked or (j,i) in self.true_marked)
            else:
                assert( (i,j) not in self.true_marked and (j,i) not in self.true_marked)
                 
