import pandas as pd
from scipy.integrate import nquad
import numpy as np

from causality.estimation.nonparametric import CausalEffect, BootstrapEstimator
from tests.unit import TestAPI 
from tests.unit.settings import TOL

"""
class TestCausalEffect(TestAPI): 
    def setUp(self):
        self.X = pd.read_csv('./tests/unit/data/X.csv')
        self.discrete = pd.read_csv('./tests/unit/data/discrete.csv')

    def test_pdf_discrete(self):
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'u','b': 'u','c': 'u','d' : 'u'}
        effect = CausalEffect(self.discrete,causes,effects,admissable_set,variable_types)
        p = effect.pdf(pd.DataFrame({ 'd' : [1], 'c' : [0]}))
        print p
        # p(d=1|do(c=0) = 0.45, p(d=1|c=0) = 0.40
        assert( abs( 0.45 - p ) < 0.02 )
    
    def test_pdf_no_adjustment(self):
        causes = ['c']
        effects = ['d']
        admissable_set = []
        variable_types={'a': 'u','b': 'u','c': 'u','d' : 'u'}
        effect = CausalEffect(self.discrete,causes,effects,admissable_set,variable_types)
        # p(d=1|do(c=0) = 0.45, p(d=1|b=0) = 0.40
        p = effect.pdf(pd.DataFrame({ 'd' : [1], 'c' : [0]}))
        print p
        assert( abs( 0.40 - p ) < 0.02 ) 

    def test_pdf_continuous(self):
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'c','b': 'c','c': 'c','d' : 'c'}
        effect = CausalEffect(self.X,causes,effects,admissable_set,variable_types)
        c = np.mean(effect.support['c'])
        d = np.mean(effect.support['d'])
        e1 =  effect.pdf(pd.DataFrame({ 'd' : [d], 'c' : [ 0.9 * c]}))
        e2 =  effect.pdf(pd.DataFrame({ 'd' : [d], 'c' : [ 1.1 * c]}))
        print e2, e1, e2 - e1, (e2 - e1) / e2
        assert( abs(e2 - e1) / e2 < 0.05 )


    def test_pdf_mixed(self):
        pass


    def test_densities(self):
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'c','b': 'c','c': 'c','d' : 'c'}
        effect = CausalEffect(self.X,causes,effects,admissable_set,variable_types)
        density =  lambda x: effect.density.pdf( data_predict=[x])
        integral = nquad( density, [effect.support[d_var] for d_var in admissable_set])[0]
        print integral
        assert(abs(integral - 1.) < TOL)

        x_vals = [np.mean(effect.support[var]) for var in causes]
        z_vals = [np.mean(effect.support[var]) for var in admissable_set]
        density = lambda x: effect.conditional_density.pdf(endog_predict=[x], exog_predict=x_vals + z_vals)
        integral = nquad(density, [effect.support[d_var] for d_var in effects])[0]
        print x_vals, z_vals,integral
        assert(abs(integral - 1.) < TOL)


    def test_get_support(self):
        data_ranges = {}
        for variable in self.X.columns:
            data_ranges[variable] = ( self.X[variable].min(), self.X[variable].max())
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'c','b': 'c','c': 'c','d' : 'c'}
        effect = CausalEffect(self.X,causes,effects,admissable_set,variable_types)
        for variable, (supp_min, supp_max) in effect.support.items():
            (xmin, xmax) = data_ranges[variable]
            assert(supp_min <= xmin)
            assert(supp_max >= xmax)


    def test_integration_function(self):
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'c','b': 'c','c': 'c','d' : 'c'}
        effect = CausalEffect(self.X,causes,effects,admissable_set,variable_types)



    def test_expectation_discrete(self):
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'u','b': 'u','c': 'u','d' : 'u'}
        effect = CausalEffect(self.discrete,
                    causes,
                    effects,
                    admissable_set,
                    variable_types, 
                    density=False, 
                    expectation=True)

        x = pd.DataFrame({ 'c' : [0]})
        p = effect.expected_value(x)
        print "p(d=1 | do(c = 0) ): ", p
        assert( abs( 0.40 - p ) < 0.05 ) 

        x = pd.DataFrame({ 'c' : [1]})
        p = effect.expected_value(x)
        print "p(d=1 | do(c = 1) ): ", p
        assert( abs( 0.40 - p ) < 0.05 ) 


        causes = ['b']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'u','b': 'u','c': 'u','d' : 'u'}
        effect = CausalEffect(self.discrete,
                    causes,
                    effects,
                    admissable_set,
                    variable_types, 
                    density=False, 
                    expectation=True)

        x = pd.DataFrame({ 'b' : [0]})
        p = effect.expected_value(x)
        print "p(d=1 | do(b = 0) ): ", p
        assert( abs( p - 0.75 ) < 0.05 )

        x = pd.DataFrame({ 'b' : [1]})
        p = effect.expected_value(x)
        print "p(d=1 | do(b = 1) ): ",p 
        assert( abs( p - 0.25 ) < 0.05 )


    def test_expectation_continuous(self):
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'c','b': 'c','c': 'c','d' : 'c'}
        effect = CausalEffect(self.X,
                    causes,
                    effects,
                    admissable_set,
                    variable_types, 
                    density=False, 
                    expectation=True)

        x = pd.DataFrame({ 'c' : [400]})
        p1 = effect.expected_value(x)
        print "E(d | do(c = 400) ): ", p1

        x = pd.DataFrame({ 'c' : [600]})
        p2 = effect.expected_value(x)
        print "E(d | do(c = 600) ): ", p2
        assert( abs( p2 - p1 ) / 200 < 0.5 )


        causes = ['b']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'c','b': 'c','c': 'c','d' : 'c'}
        effect = CausalEffect(self.X,
                    causes,
                    effects,
                    admissable_set,
                    variable_types, 
                    density=False, 
                    expectation=True)

        x = pd.DataFrame({ 'b' : [400]})
        p1 = effect.expected_value(x)
        print "E(d | do(b = 400) ): ", p1

        x = pd.DataFrame({ 'b' : [600]})
        p2 = effect.expected_value(x)
        print "E(d | do(b = 600) ): ",p2
        #assert( abs( p - 0.25 ) < 0.05 )
        assert( abs( ( p2 - p1 ) / 200 - 5. < 0.01 ) )
"""
class TestBootstrapEstimator(TestAPI):
    def setUp(self):
        size = 1000
        x1 = np.random.choice([0.1,0.15], size=size)
        x2 = x1 + np.random.normal(size=size)
        x3 = x2 + np.random.normal(size=size)
        self.X_close = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3})

        size = 3000
        x1 = np.random.choice([0,1,2], size=size)
        x2 = x1 + np.random.normal(size=size)
        x3 = x2 + np.random.normal(size=size)
        self.X_discrete = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3})
         

    def test_estimate(self):
        f = lambda X : X.groupby('x1')['x2'].mean()
        est = BootstrapEstimator(f=f)
        discr = est.estimate(self.X_discrete)
        assert discr[0][0.025] <= 0. <= discr[0][0.975] 
        assert discr[1][0.025] <= 1. <= discr[1][0.975]
        assert discr[2][0.025] <= 2. <= discr[2][0.975]

    def test_found_winner(self):
        size = 10
        x1 = np.random.choice([0.1,0.5], size=size)
        x2 = x1 + np.random.normal(size=size)
        x3 = x2 + np.random.normal(size=size)
        X_close = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3})

        f = lambda X : X.groupby('x1')['x2'].mean()
        est = BootstrapEstimator(f=f)
        close_result = est.found_winner(X_close)
        assert not close_result

        size=5000
        x1 = np.random.choice([0.1,0.5], size=size)
        x2 = x1 + np.random.normal(size=size)
        x3 = x2 + np.random.normal(size=size)
        X_close = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3})

        f = lambda X : X.groupby('x1')['x2'].mean()
        est = BootstrapEstimator(f=f)
        close_result = est.found_winner(X_close)
        assert close_result

    def test_chances_of_winning(self):
        size = 1000
        x1 = np.random.choice(['a','b'], size=size)
        x2 = np.random.normal(size=size)
        x3 = x2 + np.random.normal(size=size)
        X_even = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3})
        f = lambda X : X.groupby('x1')['x2'].mean()
        est = BootstrapEstimator(f=f)
        raise Exception(est.chances_of_winning(X_even))
        assert 0.95 * 0.5 <= est.chances_of_winning <= 1.05 * 0.5
