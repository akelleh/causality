import pandas as pd
import statsmodels.api as sm


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
        
        
class DiscreteLogisticTest():
    def __init__(self, y, x, z, data, alpha):
        self.dummy_df = pd.get_dummies(data[x[0])
        independent_dummies = []
        for col in (x[1:] + z):
            dummies = pd.get_dummies(data[col], prefix=col)
            dummies_to_keep = dummies.columns[1:]
            independent_dummies += dummies_to_keep
            self.dummy_df[dummies_to_keep] = dummies[dummies_to_keep]
        dependent_dummies = []
        for col in y:
            dummies = pd.get_dummies(data[col], prefix=col)
            dummies_to_keep = dummies.columns[1:]
            dependent_dummies += dummies_to_keep
            self.dummy_df[dummies_to_keep] = dummies[dummies_to_keep] 
        self.dummy_df['intercept'] = 1.
        self.regression = sm.Logit(data[dependent_dummies], data[independent_dummies] ) 
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
