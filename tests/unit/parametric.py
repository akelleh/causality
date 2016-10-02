import pandas as pd
import numpy as np

from causality.estimation.parametric import DifferenceInDifferences, PropensityScoreMatching
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


class TestPropScore(TestAPI):
    def test_match(self):
        matcher = PropensityScoreMatching()
        X = pd.DataFrame({'assignment': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                          'propensity score': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]})

        test, control = matcher.match(X, n_neighbors=3)

        matches = test[test['propensity score'] == 2]['matches'].values[0][0]
        assert set(control.iloc[matches]['propensity score'].values) == set([1, 2, 3])

        matches = test[test['propensity score'] == 4]['matches'].values[0][0]
        assert set(control.iloc[matches]['propensity score'].values) == set([3, 4, 5])

    def test_score(self):
        N = 100000
        z1 = np.random.normal(size=N)
        z2 = np.random.choice([0,1], size=N)
        z3 = np.random.choice(['a','b','c'], size=N)
        numeric_mapping = {'a' :3, 'b' :4, 'c' :5}
        z3_numeric = [numeric_mapping[z3i] for z3i in z3]
        p_assign = np.exp(z1 + z2 + z3_numeric) / (1. + np.exp(z1 + z2 + z3_numeric))
        assignment = np.random.binomial(1, p_assign)
        outcome = np.random.normal(assignment)
        matcher = PropensityScoreMatching()
        X = pd.DataFrame({'z1': z1, 'z2': z2, 'z3': z3, 'assignment': assignment, 'outcome': outcome})
        confounder_types = {'z1': 'c', 'z2':'o', 'z3' : 'o'}
        matcher.score(X, confounder_types, store_model_fit=True)
        assert 0.9 <= matcher.model_fit.params['z1'] <= 1.1
        assert 0.9 <= matcher.model_fit.params['z2'] <= 1.1
        assert 0.0 <= matcher.model_fit.params['z3_b'] <= 2.0
        assert 1.0 <= matcher.model_fit.params['z3_c'] <= 3.0
        assert 2.0 <= matcher.model_fit.params['intercept'] <= 4.0

    def test_at_estimators(self):
        ates = []
        atcs = []
        atts = []
        for i in range(100):
            N = 1000
            X = np.random.choice([0.25, 0.75], size=N)
            X = pd.DataFrame(X, columns=['Z'])
            X.loc[:, 'assignment'] = np.random.binomial(1, p=X['Z'])
            X.loc[:, 'outcome'] = np.random.normal(3.1 * X['assignment'] + 2.0 * X['Z'])

            matcher = PropensityScoreMatching()
            att = matcher.estimate_ATT(X, 'assignment', 'outcome', {'Z': 'c'}, n_neighbors=10)
            X.loc[:,'inverted assignment'] = (X['assignment'] + 1) % 2
            atc = matcher.estimate_ATT(X, 'inverted assignment', 'outcome', {'Z': 'c'}, n_neighbors=10)

            ate = (att + atc) / 2.
            atts.append(att)
            atcs.append(atc)
            ates.append(ate)
        X = pd.DataFrame({'att': atts, 'ate': ates, 'atc': atcs})
        assert (3.0 <= X.mean()).all()
        assert (X.mean() <= 4.0).all()
