import pandas as pd
import numpy as np

from causality.estimation.parametric import DifferenceInDifferences, PropensityScoreMatching
from tests.unit import TestAPI 


class TestDID(TestAPI): 
    def setUp(self):
        SIZE = 2000
        assignment = np.random.binomial(1,0.5, size=SIZE)
        pre_experiment = assignment + np.random.normal(-1, size=SIZE)
        start = assignment + np.random.normal(1, size=SIZE)
        end = start + np.random.normal(2.*assignment) + np.random.normal(2, size=SIZE)
        self.X_pre = pd.DataFrame({'Start' : pre_experiment, 'End' : start, 'assignment' : assignment})
        self.X = pd.DataFrame({'Start' : start, 'End' : end, 'assignment' : assignment})
        self.did = DifferenceInDifferences()

    def test_assumption_tester(self):
        assert self.did.test_parallel_trend(self.X_pre)

        self.X_pre['End'] += self.X_pre['assignment']
        assert not self.did.test_parallel_trend(self.X_pre)

    def test_did_estimator(self):
        lower, exp, upper = self.did.average_treatment_effect(self.X)
        assert 1.8 <= exp <= 2.2
        assert lower <= exp <= upper
        
        self.did = DifferenceInDifferences(robust=True)
        lower, exp, upper = self.did.average_treatment_effect(self.X)
        assert 1.8 <= exp <= 2.2
        assert lower <= exp <= upper 


class TestPropScore(TestAPI):
    def test_match(self):
        matcher = PropensityScoreMatching()
        X = pd.DataFrame({'assignment': [1, 0, 0, 0, 0, 0],
                          'propensity score': [3, 1, 2, 3, 5, 4]})

        test, control = matcher.match(X, n_neighbors=3)
        assert set(control['propensity score'].values) == set([2, 3, 4])

    def test_score(self):
        N = 5000
        z1 = np.random.normal(size=N)
        z2 = np.random.choice(['a','b','c'], size=N)
        numeric_mapping = {'a' :3, 'b' :4, 'c' :5}
        z2_numeric = [numeric_mapping[z2i] for z2i in z2]
        p_assign = np.exp(z1 + z2_numeric) / (1. + np.exp(z1 + z2_numeric))
        assignment = np.random.binomial(1, p_assign)
        outcome = np.random.normal(assignment)
        matcher = PropensityScoreMatching()
        X = pd.DataFrame({'z1': z1, 'z2': z2, 'assignment': assignment, 'outcome': outcome})
        confounder_types = {'z1': 'c', 'z2':'o'}
        matcher.score(X, confounder_types, store_model_fit=True)
        assert 0.7 <= matcher.propensity_score_model.params['z1'] <= 1.3
        assert 0.0 <= matcher.propensity_score_model.params['z2_b'] <= 2.0
        assert 1.0 <= matcher.propensity_score_model.params['z2_c'] <= 3.0
        assert 2.0 <= matcher.propensity_score_model.params['intercept'] <= 4.0

    def test_at_estimators(self):
        N = 1000  # how many data points

        z1 = 0.5 * np.random.normal(size=N)  # a few confounding variables
        z2 = 0.5 * np.random.normal(size=N)
        z3 = 0.5 * np.random.normal(size=N)

        arg = (z1 + z2 + z3 + np.random.normal(size=N))
        p = np.exp(arg) / (1. + np.exp(arg))  # propensity to receive treatment, P(d|z), taking on a logistic form
        d = np.random.binomial(1, p)

        y = (np.random.normal(size=N) + (z1 + z2 + z3 + 1.) * d)  # effect of d is confounded by z. True ATE is 1.

        X = pd.DataFrame({'d': d, 'z1': z1, 'z2': z2, 'z3': z3, 'y': y, 'p': p})

        matcher = PropensityScoreMatching()
        ATE = matcher.estimate_ATE(X, 'd', 'y', {'z1': 'c', 'z2': 'c', 'z3': 'c'})
        assert 0.9 <= ATE <= 1.1
