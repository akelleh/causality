import pandas as pd
import numpy as np
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

class ChiSquaredTest():
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
       
class MutualInformationTest(object):
    """
    Mutual information is the most general independence test, but also one of the hardest
    to estimate well.  This implementation takes a forceful approach of estimating the 
    densities, and calculating the mutual information by directly summing/integrating.

    There are almost certainly more efficient implementations, but we want to avoid 
    approaches that require discretization.  
    """
    def __init__(self, variable_types):
        self.variable_types = variable_types

    def estimate_densities(self, x, y, z, X):
        p_x_given_z = self.estimate_cond_pdf(x, z, X)
        p_y_given_z = self.estimate_cond_pdf(y, z, X)
        p_xy_given_z = self.estimate_cond_pdf(x+y, z, X)
        p_z = self.estimate_cond_pdf(z, [], X)
        return self.mutual_information(pxy_given_z, p_x_given_z, p_y_given_z, p_z, x, y, z)

    def estimate_cond_pdf(self, x, z, X):
        # normal_reference works better with mixed types
        if 'c' not in [self.variable_types[xi] for xi in x+z]
            bw = 'cv_ml'
        else:
            bw = 'normal_reference'

        # if conditioning on the empty set, return a pdf instead of cond pdf
        if len(z) == 0:
            return KDEMulivariate(X[x],
                                  var_type=''.join([self.variable_types[xi] for xi in x]),
                                  bw=bw)
        else:
            return KDEMultivariateConditional(endog=X[x],
                                              exog=X[z],
                                              dep_type=''.join([self.variable_types[xi] for xi in x]),
                                              indep_type=''.join([self.variable_types[zi] for zi in z]),
                                              bw=bw)

    def mutual_information(self, pxy_given_z, p_x_given_z, p_y_given_z, p_z, x, y, z):
        pass        
        

    def sum_or_integrate(self, function, ranges, sum_mask):
        support = self.get_support(X)


    def get_fitted_bandwidths(self, x, y, z, p_x_given_z, p_y_given_z, p_xy_given_z, p_z):
        variable_bandwidths = defaultdict(lambda x: 0.)
        for i, vi in enumerate(x):
            if variable_bandwidths[vi] < p_x_given_z.bw[i]:
                variable_bandwidths[vi] = p_x_given_z.bw[i]
            if variable_bandwdiths[vi] < p_xy_given_z.bw[i]:
                variable_bandwidths[vi] = p_xy_given_z.bw[i]
        for i, vi in enumerate(y):
            if variable_bandwidths[vi] < p_y_given_z.bw[i]:
                variable_bandwidths[vi] = p_y_given_z.bw[i]
            if variable_bandwdiths[vi] < p_xy_given_z.bw[len(x)+i]:
                variable_bandwidths[vi] = p_xy_given_z.bw[len(x)+i]
        for i, vi in enumerate(z):
            if variable_bandwidths[vi] < p_y_given_z.bw[len(y)+i]:
                variable_bandwidths[vi] = p_y_given_z.bw[len(y)i]
            if variable_bandwdiths[vi] < p_xy_given_z.bw[len(x)+len(y)+i]:
                variable_bandwidths[vi] = p_xy_given_z.bw[len(x)+len(y)+i]
            if variable_bandwidths[vi] < p_z.bw[i]:
                variable_bandwidths[vi] = p_z.bw[i]
        return variable_bandwidths

    def get_support(self,X, x, y, z, p_x_given_z, p_y_given_z, p_xy_given_z, p_z):
        """
        find the smallest cube around which the densities are supported,
        allowing a little flexibility for variables with larger bandwidths.
        """
        data_support = {vi : (X[vi].min(), X[vi].max()) for vi in self.variable_types.keys()}
        fitted_bandwidths = self.get_fitted_bandwidths(x, y, z, p_x_given_z, p_y_given_z, p_xy_given_z, p_z)

        support = {}
        for vi in x+y+z:
            if self.variable_types[vi] == 'c':
                # for continuous variables, go outside the data support to capture
                # the kernel's contribution
                lower_support = data_support[vi][0] - 10. * fitted_bandwidths[vi]
                upper_support = data_support[vi][1] + 10. * fitted_bandwidths[vi]
                support[vi] = (lower_support, upper_support)
            else:
                # for discrete data, sum over the data's values
                support[vi] = data_support[vi]
        return support

