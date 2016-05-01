from tests.unit import TestAPI
import itertools

import numpy.random
import pandas as pd
import networkx as nx

from causality.inference.independence_tests import RobustRegressionTest, ChiSquaredTest, MutualInformationTest

TEST_SET_SIZE = 2000
TRIALS = 2
P = 0.5

class TestChi2(TestAPI):

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

    def test_chi2(self):
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



class TestMutualInformation(TestAPI):

    def setUp(self):
        size = 1000
        x1 = numpy.random.choice(range(5), size=size)
        x2 = [round(0.7*numpy.random.rand() * xi) for xi in x1]
        x3 = [round(0.7*numpy.random.rand() * xi) for xi in x2]
        self.X = pd.DataFrame({'x1':x1,'x2':x2, 'x3':x3})       
        self.alpha = 0.05
        self.variable_types = {'x1':'d', 'x2':'d', 'x3':'d'} 

    def test_mi(self):
        y = ['x3']
        x = ['x1']
        z = ['x2']
        test = MutualInformationTest(y, x, z, self.X, self.alpha, variable_types=self.variable_types)
        assert(test.independent())

        y = ['x3']
        x = ['x1']
        z = []
        test = MutualInformationTest(y, x, z, self.X, self.alpha, variable_types=self.variable_types)
        assert(not test.independent())

        y = ['x1']
        x = ['x1']
        z = []
        test = MutualInformationTest(y, x, z, self.X, self.alpha, variable_types=self.variable_types)
        assert(not test.independent())
    
        I, dI = test.max_likelihood_information(x, y, self.X)
        z = 1.96
        assert((numpy.exp(I-z*dI) < 5) and (5 < numpy.exp(I+z*dI)))
