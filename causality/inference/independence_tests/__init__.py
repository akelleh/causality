import pandas as pd
import statsmodels.api as sm
import scipy.stats
import itertools

class RobustRegressionTest():
    def __init__(self, y, x, z, data, alpha):
        self.regression = sm.RLM(data[y], data[x+z])
        self.result = self.regression.fit()
        self.coefficient = self.result.params[x][0]
        confidence_interval = self.result.conf_int(alpha=alpha/2.)
        self.upper = confidence_interval[1][x][0]
        self.lower = confidence_interval[0][x][0]

    def independent(self):
        if self.coefficient > 0.:
            if self.lower > 0.:
                return False
            else:
                return True
        else:
            if self.upper < 0.:
                return False
            else:
                return True

class DiscreteChiSquaredTest():
    def __init__(self, y, x, z, data, alpha):
        self.alpha = alpha
        self.total_chi2 = 0.
        self.total_dof = 0
        for xi, yi in itertools.product(x,y):
            tables = data[[xi]+[yi]+z].copy()
            groupby_key = tuple([zi for zi in z] + [xi])
            tables = tables.join(pd.get_dummies(data[yi],prefix=yi)).groupby(groupby_key).sum()
            del tables[yi]

            z_values = {zi : data.groupby(zi).groups.keys() for zi in z}
            x_values = {xi : data.groupby(xi).groups.keys()}
            y_values = {yi : data.groupby(yi).groups.keys()}

            contingencies = itertools.product(*[z_values[zi] for zi in z])

            for contingency in contingencies:
                contingency_table = tables.loc[contingency].values
                try:
                    chi2, _, dof, _ = scipy.stats.chi2_contingency(contingency_table)
                except ValueError:
                    raise Exception("Not enough data or entries with 0 present: Chi^2 Test not applicable.")
                self.total_dof += dof
                self.total_chi2 += chi2
        self.total_p = 1. - scipy.stats.chi2.cdf(self.total_chi2, self.total_dof)

    def independent(self):
        if self.total_p < self.alpha:
            return False
        else:
            return True
        
