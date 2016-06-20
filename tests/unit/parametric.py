import pandas as pd
import numpy as np

from causality.estimation.parametric import DifferenceInDifferences 
from tests.unit import TestAPI 
from tests.unit.settings import TOL


class TestDID(TestAPI): 
    def setUp(self):
        SIZE = 2000
        assignment = np.random.binomial(1,0.5, size=SIZE)
        pre_experiment = assignment + np.random.normal(-1, size=SIZE)
        start = assignment + np.random.normal(1, size=SIZE)
        end = start + np.random.normal(2.*assignment) + np.random.normal(2, size=SIZE)
        self.X_pre = pd.DataFrame({'Start' : pre_experiment, 'End' : start, 'Assignment' : assignment})
        self.X = pd.DataFrame({'Start' : start, 'End' : end, 'Assignment' : assignment})
        self.did = DifferenceInDifferences()

    def test_assumption_tester(self):
        assert self.did.test_parallel_trend(self.X_pre)

        self.X_pre['End'] += self.X_pre['Assignment']
        assert not self.did.test_parallel_trend(self.X_pre)

    def test_did_estimator(self):
        lower, exp, upper = self.did.average_treatment_effect(self.X)
        assert 1.8 <= exp <= 2.2
        assert lower <= exp <= upper
        
        self.did = DifferenceInDifferences(robust=True)
        lower, exp, upper = self.did.average_treatment_effect(self.X)
        assert 1.8 <= exp <= 2.2
        assert lower <= exp <= upper 
