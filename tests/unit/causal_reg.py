import pandas as pd
from scipy.integrate import nquad
import numpy as np

from nonparametric.causal_reg import CausalEffect
from tests.unit import TestAPI 
from tests.unit.settings import TOL


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
        # p(d=1|do(c=0) = 0.90, p(d=1|b=0) = 0.40
        p = effect.pdf(pd.DataFrame({ 'd' : [1], 'c' : [0]}))
        print p
        assert( abs( 0.40 - p ) < 0.02 ) 

        
    def test_pdf_continuous(self):
        causes = ['c']
        effects = ['d']
        admissable_set = ['a']
        variable_types={'a': 'c','b': 'c','c': 'c','d' : 'c'}
        effect = CausalEffect(self.X,causes,effects,admissable_set,variable_types)
        e1 =  effect.pdf(pd.DataFrame({ 'd' : [3000], 'c' : [400]}))
        e2 =  effect.pdf(pd.DataFrame({ 'd' : [3000], 'c' : [600]}))
        assert( abs(e2 - e1) / e2 < TOL )


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
