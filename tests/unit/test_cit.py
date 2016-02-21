from tests.unit import TestAPI
import itertools

import numpy.random
import pandas as pd
import networkx as nx

from causality.inference.independence_tests import RobustRegressionTest, ChiSquaredTest

TEST_SET_SIZE = 2000
TRIALS = 2
P = 0.5

class Test_CIT(TestAPI):

    def setUp(self):
        a = numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE)
        b = (numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE) + a) % 3
        c = numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE)
        d = numpy.random.binomial(TRIALS,P,size=TEST_SET_SIZE)
        self.X = pd.DataFrame({'a' : a,
                               'b' : b,
                               'c' : c, 
                               'd' : d }) 
        self.alpha = 0.05

    def test_independent(self):
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
