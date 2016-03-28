from tests.unit import TestAPI
import itertools

import numpy
import numpy.random
import pandas as pd
import networkx as nx

from causality.inference.search import ICs
from causality.inference.independence_tests import RobustRegressionTest, ChiSquaredTest

TEST_SET_SIZE = 50000
TRIALS = 2
P = 0.5

class Test_CIT(TestAPI):

    def setUp(self):
        a = numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE)
        b = (numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE) + a) % 3
        c = numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE)
        d = numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE)
        e = (numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE) + b) % 3
        self.X = pd.DataFrame({'a' : a,
                               'b' : b,
                               'c' : c, 
                               'd' : d,
                               'e' : e }) 
        self.alpha = 0.05

    def test_independent(self):
        x = ['a']
        y = ['e']
        z = []
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(not test.independent())

        x = ['a']
        y = ['e']
        z = ['b']
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(test.independent())

        x = ['a']
        y = ['b']
        z = []
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(not test.independent())

        x = ['a']
        y = ['b']
        z = ['c','d']
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(not test.independent())

        x = ['a']
        y = ['c']
        z = []
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(test.independent())

        
        x = ['a']
        y = ['c']
        z = ['b']
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(test.independent())


        x = ['a','b']
        y = ['c']
        z = ['d']
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(test.independent())


        x = ['a']
        y = ['b','c']
        z = ['d']
        test = ChiSquaredTest(y,x,z,self.X,self.alpha)
        assert(not test.independent())

    def test_discretization(self):
        x1 = numpy.random.normal(size=TEST_SET_SIZE)
        x2 = x1 + numpy.random.normal(size=TEST_SET_SIZE)
        x3 = x2 + numpy.random.normal(size=TEST_SET_SIZE)

        X = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3})

        variable_types = {'x1' : 'c', 'x2' : 'c', 'x3' : 'c'}

        ic_algorithm = ICs(ChiSquaredTest, X, variable_types, discretize=True)
        print ic_algorithm.data.head()

        x = ['x1']
        y = ['x3']
        z = []
        test = ChiSquaredTest(y,x,z,ic_algorithm.data,self.alpha)
        print (test.total_p, test.total_chi2, test.total_dof)
        assert(not test.independent())

        x = ['x1']
        y = ['x3']
        z = ['x2']
        test = ChiSquaredTest(y,x,z,ic_algorithm.data,self.alpha)
        print (test.total_p, test.total_chi2, test.total_dof)
        #assert(test.independent())

        x1_marginal = ic_algorithm.data.groupby('x1').count() / float(len(ic_algorithm.data))
        x2_marginal = ic_algorithm.data.groupby('x2').count() / float(len(ic_algorithm.data)) 
        x3_marginal = ic_algorithm.data.groupby('x3').count() / float(len(ic_algorithm.data))
        print x1_marginal
        for i in range(5):
            x1 = numpy.random.choice(range(3), p=x1_marginal['x2'].values, size=TEST_SET_SIZE)
            x2 = numpy.random.choice(range(3), p=x2_marginal['x1'].values, size=TEST_SET_SIZE)
            x3 = numpy.random.choice(range(3), p=x3_marginal['x1'].values, size=TEST_SET_SIZE)
            Y = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3})
            test = ChiSquaredTest(y,x,z,Y,self.alpha)
            print (test.total_p, test.total_chi2, test.total_dof)
        raise Exception(test.total_p, test.total_chi2, test.total_dof)
