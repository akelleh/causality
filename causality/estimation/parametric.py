import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.discrete.discrete_model import Logit
from sklearn.neighbors import NearestNeighbors


class DifferenceInDifferences(object):
    def __init__(self, robust=True):
        """
        We will take a dataframe where each row is a user,
        and the columns are:
        (1) Assignment: 1 = test, 0 = control
        (2) Start: The value of the metric you're interested
            in at the start of the experiment.
        (3) End: The value of the metric you're interested in
            at the end of the experiment.
        """
        if robust:
            self.model = RLM
        else:
            self.model = OLS

    def average_treatment_effect(self, X, start='Start', end='End', assignment='Assignment'):
        test = X[X['Assignment']==1][['Start','End']]
        control = X[X['Assignment']==0][['Start','End']]
        del X

        test_initial = test['Start']
        test_final = test['End']
        control_initial = control['Start']
        control_final = control['End']
        del test, control

        df = pd.DataFrame({'y' : test_initial, 
                   'assignment' : [1. for i in test_initial], 
                   't' :[0. for i in test_initial] })
        df = df.append(pd.DataFrame({'y' : test_final, 
                                     'assignment' : [1. for i in test_final], 
                                     't' :[1. for i in test_final] }))

        df = df.append(pd.DataFrame({'y' : control_initial, 
                                     'assignment' : [0. for i in control_initial], 
                                     't' :[0. for i in control_initial] }))

        df = df.append(pd.DataFrame({'y' : control_final, 
                                     'assignment' : [0. for i in control_final], 
                                     't' :[1. for i in control_final] }))
        del test_initial, test_final, control_initial, control_final
        df['did'] = df['t'] * df['assignment'] 
        df['intercept'] = 1.

        model = self.model(df['y'], df[['t', 'assignment','did', 'intercept']])
        result = model.fit()
        conf_int = result.conf_int().ix['did']
        expected = result.params['did']
        return conf_int[0], expected, conf_int[1]
        
    def test_parallel_trend(self, X, start='Start', end='End', assignment='Assignment'):
        """
        This will find the average treatment effect on
        a dataset before the experiment is run, to make
        sure that it is zero.  This tests the assumption
        that the average treatment effect between the test
        and control groups when neither is treated is 0.

        The format for this dataset is the same as that 
        for the real estimation task, except that the start
        time is some time before the experiment is run, and
        the end time is the starting point for the experiment.
        """
        lower, exp, upper = self.average_treatment_effect(X,start=start, end=end, assignment=assignment)
        if lower <= 0 <= upper:
            return True
        return False


class PropensityScoreMatching(object):
    def __init__(self):
        # change the model if there are multiple matches per treated!
        pass

    def score(self, X, confounder_types, assignment='assignment', store_model_fit=False, intercept=True):
        df = X[[assignment]]
        regression_confounders = []
        for confounder, var_type in confounder_types.items():
            if var_type == 'o' or var_type == 'u':
                c_dummies = pd.get_dummies(X[[confounder]], prefix=confounder)
                if len(c_dummies.columns) == 1:
                    df[c_dummies.columns] = c_dummies[c_dummies.columns]
                    regression_confounders.extend(c_dummies.columns)
                else:
                    df[c_dummies.columns[1:]] = c_dummies[c_dummies.columns[1:]]
                    regression_confounders.extend(c_dummies.columns[1:])
            else:
                regression_confounders.append(confounder)
                df.loc[:,confounder] = X[confounder].copy() #
                df.loc[:,confounder] = X[confounder].copy() #
        if intercept:
            df.loc[:,'intercept'] = 1.
            regression_confounders.append('intercept')
        logit = Logit(df[assignment], df[regression_confounders])
        result = logit.fit()
        if store_model_fit:
            self.model_fit = result
        X.loc[:,'propensity score'] = result.predict(df[regression_confounders])
        return X

    def match(self, X, assignment='assignment', score='propensity score', n_neighbors=2):
        treatments = X[X[assignment] != 0]
        control = X[X[assignment] == 0]
        neighbor_search = NearestNeighbors(metric='euclidean', n_neighbors=n_neighbors)
        neighbor_search.fit(control[[score]].values)
        treatments.loc[:, 'matches'] = treatments[score].apply(lambda x: neighbor_search.kneighbors(x)[1])
        return treatments, control

    def estimate_treatments(self, treatments, control, outcome):
        def get_matched_outcome(matches):
            return sum([control[outcome].values[i] / float(len(matches[0])) for i in matches[0]])
        treatments.loc[:,'control outcome'] = treatments['matches'].apply(get_matched_outcome)
        return treatments

    def estimate_ATT(self, X, assignment, outcome, confounder_types, n_neighbors=5):
        X = self.score(X, confounder_types, assignment)
        treatments, control = self.match(X, assignment='assignment', score='propensity score', n_neighbors=n_neighbors)
        treatments = self.estimate_treatments(treatments, control, outcome)
        y_hat_treated = treatments[outcome].mean()
        y_hat_control = treatments['control outcome'].mean()
        return y_hat_treated - y_hat_control

    def estimate_ATC(self, X, assignment, outcome, confounder_types, n_neighbors=5):
        """
        Assumes a 1 for the test assignment, 0 for the control assignment
        :param X: The data set, with (at least) an assignment, set of confounders, and an outcome
        :param assignment: A categorical variable (currently, 0 or 1) indicating test or control group, resp.
        :param outcome: The outcome of interest.  Should be real-valued or ordinal.
        :param confounder_types: A dictionary of variable_name: variable_type pairs of strings, where
        variable_type is in {'c', 'o', 'd'}, for 'continuous', 'ordinal', and 'discrete'.
        :param n_neighbors: An integer for the number of neighbors to use with k-nearest-neighbor matching
        :return: a float representing the treatment effect
        """
        X['assignment'] = (X['assignment'] + 1) % 2
        return -self.estimate_ATT(X, assignment, outcome, confounder_types, n_neighbors=n_neighbors)

    def estimate_ATE(self, X, assignment, outcome, confounder_types, n_neighbors=5):
        att = estimate_ATT(self, X, assignment, outcome, confounder_types, n_neighbors=n_neighbors)
        atc = estimate_ATC(self, X, assignment, outcome, confounder_types, n_neighbors=n_neighbors)
        return (atc+att)/2. 
