import pandas as pd
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional, KDEMultivariate
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_regression import KernelReg
import itertools
from scipy.integrate import nquad
from scipy import stats
import numpy as np
from multiprocessing import Pool
from scipy.signal import fftconvolve
from scipy.stats import multivariate_normal
from scipy.interpolate import LinearNDInterpolator, interpn

def I_sample(args):
    x1_sample,x2_sample,x1,x2,variable_types = args
    null_X = pd.DataFrame( { x1 : x1_sample, x2 : x2_sample } )
    null_model_data = DataSet(null_X, variable_types)
    null_I = null_model_data.biased_mutual_information(x1, x2, p=False)
    print null_I
    return null_I


class MutualInformation(object):
    def get_discrete_kernel(self, edges, bw, mean):
        centers = []
        for edge in edges:
            centers.append([edge[i] for i in range(len(edge)) ])
        kernel = np.zeros([len(center) for center in centers])
        mean   = [0. for i in centers]
        index_generator = [xrange(len(center)) for center in centers]
        points = []
        for indexes, xi in zip(itertools.product(*index_generator),itertools.product(*centers)):
            points.append(np.array(xi))
        kernel = multivariate_normal.pdf(points,mean=mean,cov=bw).reshape([len(center) for center in centers])
        return kernel / float(sum(kernel.flatten()))

    def estimate_discrete_density(self, X):
        heatmap, edges = np.histogramdd(X.values, bins=50)
        bin_sizes = [ edge[1] - edge[0] for edge in edges]
        new_edges = []
        for i, (edge, bin_size) in enumerate(zip(edges, bin_sizes)):
            new_edges.append(list(edge) + [edge[-1] + bin_size])
        heatmap, new_edges= np.histogramdd(X.values, bins=new_edges)
        dA = 1.
        for bin_size in bin_sizes:
            dA *= bin_size
        bw = np.sqrt(X.var()) * 1.06 * len(X)**(-1./5.)
        kernel = self.get_discrete_kernel(edges,bw, X.mean());
        kernel_estimate = fftconvolve(heatmap, kernel, mode='same')
        return kernel_estimate / (dA*float(len(X))), edges

    def get_subset_entropy(self, X, subset):
        kernel_estimate, points= self.estimate_discrete_density(X[subset])
        p = interpn(points, kernel_estimate, X[subset], method='linear', fill_value=0.)
        entropy = 0.
        for pi in p:
            if pi > 0:
                entropy += - np.log( pi ) / float(len(X))
        return entropy
    
    def estimate(self, X, subset_1, subset_2):
        H1 = self.get_subset_entropy(X, subset_1)
        H2 = self.get_subset_entropy(X, subset_2)
        H12 = self.get_subset_entropy(X, subset_1 + subset_2)
        return H1 + H2 - H12

class DataSet(object):
    def __init__(self, X, variable_types, edgelist=None):
        """
        This object represents the dataset.  It holds the DataFrame
        and the edgelist that has been inferred for it.  It contains 
        utility functions for operating on the data, like measuring 
        mutual information.
        """
        self.X = X
        self.variable_types = variable_types
        if 'c' not in variable_types.values():
            bw = 'cv_ml'
        else:
            bw = 'normal_reference'

        self.pdf_estimate = self.density = KDEMultivariate(X, 
                                  var_type=''.join([variable_types[var] for var in X.columns]),
                                  bw=bw)
        self.support = self.__get_support()


    def __get_support(self):
        """
        find the smallest cube around which the densities are supported,
        allowing a little flexibility for variables with larger bandwidths.
        """
        data_support = { variable : (self.X[variable].min(), self.X[variable].max()) for variable in self.X.columns}
        variable_bandwidths = { variable : bw for variable, bw in zip(self.X.columns, self.pdf_estimate.bw)}
        support = {}
        for variable in self.X.columns:
            if self.variable_types[variable] == 'c':
                lower_support = data_support[variable][0] - 10. * variable_bandwidths[variable]
                upper_support = data_support[variable][1] + 10. * variable_bandwidths[variable]
                support[variable] = (lower_support, upper_support)
            else:
                support[variable] = data_support[variable]
        return support


    def biased_mutual_information(self, x1, x2, verbose=False, p=False, p_value_trials=100):
        """
        compute the mutual information I(x1,x2) between x1 and x2 for continuous variables,
        I(x1,x2) = H(x2) - H(x2 | x1) by using a kernel density estimate for the densities,
        and calculating the entropies with the biased estimator of integrating -p log p.
        """
        self.x1 = x1
        self.x2 = x2

        dep_type      = [self.variable_types[x1]]
        indep_type    = [self.variable_types[x2]]
        density_types = [self.variable_types[var] for var in [x1,x2]] 

        if 'c' not in density_types:
            bw = 'cv_ml'
        else:
            bw = 'normal_reference'
        if verbose:
            print "estimating %s density" % x2
        self.mi_density = KDEMultivariate(self.X[[x2]],
                                  var_type=''.join(dep_type),
                                  bw=bw)
        if verbose:
            print "estimating %s|%s density" % (x2,x1)
        self.mi_conditional_density = KDEMultivariateConditional(endog=self.X[x2],
                                                         exog=self.X[x1],
                                                         dep_type=''.join(dep_type),
                                                         indep_type=''.join(indep_type),
                                                         bw=bw)
        if verbose:
            print "integrating for marginal entropy"
        self.x1_integration_density = KDEMultivariate(self.X[[x2]],
                                  var_type=''.join(dep_type),
                                  bw=bw)
        x2_range = [self.support[x2] ]
        self.integration_density = self.mi_density
        self.cond_integration_density = self.mi_conditional_density
        Hx2 = nquad(self.entropy_integration_function, x2_range)[0]
        if verbose:
            print "H%s" % x2 ,Hx2 
            print "integrating for conditional entropy"
        x1x2_range = [self.support[x1], self.support[x2]]
        Hx2givenx1 = nquad(self.entropy_integration_function, x1x2_range)[0]
        if verbose:
            print "H%s|%s" % (x2, x1), Hx2givenx1
        if p:
            if verbose:
                print "calculating p value with %d trials" % p_value_trials
            p_value = self.__I_pvalue(x1,x2,Hx2 - Hx2givenx1, p_value_trials)
        if p == True and verbose:
            print "p I > I_measured: ", p_value
        if p == True:
            return Hx2 - Hx2givenx1, p_value
        return Hx2 - Hx2givenx1

    def __I_pvalue(self, x1, x2, I_measured, p_value_trials):
        """
        Calculate the probability that the empirical mutual information is 
        at least as extreme as the measured value (I >= I_measured) given that
        the null hypothesis I_actual = 0 is true.  Do this by generating a 
        sample dataset from the product of the empirical marginal distributions,
        measuring I for each dataset, and then estimating the density of I to get
        the distribution being sampled from under the null hypothesis.
        """
        p1 = KDEUnivariate(self.X[[x1]])
        p2 = KDEUnivariate(self.X[[x2]])
        p1.fit()
        p2.fit()
        class P1_dist(stats.rv_continuous):
            def _pdf(self, x):
                return p1.evaluate(x)
            def _stats(self):
                return 0.,0.,0.,0.
        class P2_dist(stats.rv_continuous):
            def _pdf(self, x):
                return p2.evaluate(x)
            def _stats(self):
                return 0.,0.,0.,0.

        p1_dist = P1_dist()
        p2_dist = P2_dist()

        null_I_samples = []
        args = [(p1_dist.rvs(size=self.X.count()[x1]), p2_dist.rvs(size=self.X.count()[x2]), x1, x2, self.variable_types) for i in range(p_value_trials)]
        p = Pool(processes=4)
        null_I_samples = p.map(I_sample, args)
        #for i in range(p_value_trials):
        #    print "trial %d" % i
        #    null_X = pd.DataFrame( { x1 : p1_dist.rvs(size=self.X.count()[x1]), x2 : p2_dist.rvs(size=self.X.count()[x2]) } )
        #    null_model_data = DataSet(null_X, self.variable_types)
        #    null_I = null_model_data.biased_mutual_information(x1, x2, p=False)
        #    print null_I
        #    null_I_samples.append(null_I)
        print null_I_samples
        I_dist = KDEUnivariate(null_I_samples)
        I_dist.fit()
        class I_pdf(stats.rv_continuous):
            def _pdf(self,x):
                return I_dist.evaluate(x)
            def _stats(self):
                0.,0.,0.,0.
        i_pdf = I_pdf()
        return 1. - i_pdf.cdf(I_measured)


    def entropy_integration_function(self, *args):
        """
        This is the function integrated to give an entropy for mutual information
        calculations.
        """
        if len( args ) == 2:
            var = [self.x1, self.x2]
        elif len(args) == 1:
            var = [self.x2]
        else:
            raise Exception( "Too few args in entropy integration" )
        data = pd.DataFrame( {k : [v] for k, v in zip(var, args) } )
        if len(args) == 2:
            p = self.cond_integration_density.pdf(exog_predict=data[self.x1],
                                    endog_predict=data[self.x2]) 
            return - self.x1_integration_density.pdf( data_predict=data[self.x1] ) * p * np.log(p)
        else:
            p = self.integration_density.pdf( data_predict=data[self.x2] )
            return - p * np.log( p ) 





class CausalEffect(object):
    def __init__(self, X, causes, effects, admissable_set=[], variable_types=None, expectation=False, density=True):
        """
        We want to calculate the causal effect of X and Y through
        back-door adjustment, P(Y|do(X)) = Sum( P(Y|X,Z)P(Z), Z) 
        for some admissable set of control variables, Z.  First we 
        calculate the conditional density P(Y|X,Z), then the density
        P(Z).  We find the support of Z so we can properly sum over
        it later.  variable_types are a dictionary with the column name
        pointing to an element of set(['o', 'u', 'c']), for 'ordered',
        'unordered discrete', or 'continuous'.
        """
        conditional_density_vars = causes + admissable_set
        self.causes = causes
        self.effects = effects
        self.admissable_set = admissable_set
        self.conditional_density_vars = conditional_density_vars
        
        if variable_types:
            self.variable_types = variable_types
            dep_type      = [variable_types[var] for var in effects]
            indep_type    = [variable_types[var] for var in conditional_density_vars]
            density_types = [variable_types[var] for var in admissable_set]
        else:
            self.variable_types = self.__infer_variable_types(X)

        if 'c' not in variable_types.values():
            bw = 'cv_ml'
        else:
            bw = 'normal_reference'


        if admissable_set:            
            self.density = KDEMultivariate(X[admissable_set], 
                                  var_type=''.join(density_types),
                                  bw=bw)
        
        self.conditional_density = KDEMultivariateConditional(endog=X[effects],
                                                         exog=X[conditional_density_vars],
                                                         dep_type=''.join(dep_type),
                                                         indep_type=''.join(indep_type),
                                                         bw=bw)
        if expectation:
            self.conditional_expectation = KernelReg(X[effects].values,
                                                 X[conditional_density_vars].values,
                                                 ''.join(indep_type),
                                                 bw='cv_ls')

        self.support = self.__get_support(X)
        
        self.discrete_variables = [ variable for variable, var_type in self.variable_types.items() if var_type in ['o', 'u']]
        self.discrete_Z = list(set(self.discrete_variables).intersection(set(admissable_set)))
        self.continuous_variables = [ variable for variable, var_type in self.variable_types.items() if var_type == 'c' ]
        self.continuous_Z = list(set(self.continuous_variables).intersection(set(admissable_set)))
       
 
    def __infer_variable_types(self,X):
        """
        fill this in later.
        """
        pass
       
 
    def __get_support(self, X):
        """
        find the smallest cube around which the densities are supported,
        allowing a little flexibility for variables with larger bandwidths.
        """
        data_support = { variable : (X[variable].min(), X[variable].max()) for variable in X.columns}
        variable_bandwidths = { variable : bw for variable, bw in zip(self.effects + self.conditional_density_vars, self.conditional_density.bw)}
        support = {}
        for variable in self.effects + self.conditional_density_vars:
            if self.variable_types[variable] == 'c':
                lower_support = data_support[variable][0] - 10. * variable_bandwidths[variable]
                upper_support = data_support[variable][1] + 10. * variable_bandwidths[variable]
                support[variable] = (lower_support, upper_support)
            else:
                support[variable] = data_support[variable]
        return support

        
    def integration_function(self,*args):
        # takes continuous z, discrete z, then x
        data = pd.DataFrame({ k : [v] for k, v in zip(self.continuous_Z + self.discrete_Z + self.causes + self.effects, args)})
        conditional = self.conditional_density.pdf(exog_predict=data[self.conditional_density_vars].values[0], 
                                                   endog_predict=data[self.effects].values[0]) 
        density = self.density.pdf(data_predict=data[self.admissable_set])
        return conditional * density

    
    def expectation_integration_function(self, *args):
        data = pd.DataFrame({ k : [v] for k, v in zip(self.continuous_Z + self.discrete_Z + self.causes, args)})
        conditional = self.conditional_expectation.fit(data_predict=data[self.conditional_density_vars].values)[0]
        density = self.density.pdf(data_predict=data[self.admissable_set])
        return conditional * density

    
    def pdf(self, x):
        """
        Currently, this does the whole sum/integral over the cube support of Z.
        We may be able to improve this by taking into account how the joint
        and conditionals factorize, and/or finding a more efficient support.
        
        This should be reasonably fast for |Z| <= 2 or 3, and small enough discrete
        variable cardinalities.  It runs in O(n_1 n_2 ... n_k) in the cardinality of
        the discrete variables, |Z_1| = n_1, etc.  It likewise runs in O(V^n) for n
        continuous Z variables.  Factorizing the joint/conditional distributions in
        the sum could linearize the runtime.
        """
        causal_effect = 0.
        x = x[self.causes + self.effects]
        if self.discrete_Z:
            discrete_variable_ranges = [ xrange(*(int(self.support[variable][0]), int(self.support[variable][1])+1)) for variable in self.discrete_Z]
            for z_vals in itertools.product(*discrete_variable_ranges):
                z_discrete = pd.DataFrame({k : [v] for k, v in zip(self.discrete_Z, z_vals)})
                if self.continuous_Z:
                    continuous_Z_ranges = [self.support[variable] for variable in self.continuous_Z]
                    args = z_discrete.join(x).values[0]
                    causal_effect += nquad(self.integration_function,continuous_Z_ranges,args=args)[0]
                else:
                    z_discrete = z_discrete[self.admissable_set]
                    exog_predictors = x.join(z_discrete)[self.conditional_density_vars]
                    conditional = self.conditional_density.pdf(exog_predict=exog_predictors, 
                                                               endog_predict=x[self.effects]) 
                    density = self.density.pdf(data_predict=z_discrete)
                    dc = conditional * density
                    causal_effect += dc
            return causal_effect
        elif self.continuous_Z:
            continuous_Z_ranges = [self.support[var] for var in self.continuous_Z]
            causal_effect = nquad(self.integration_function,continuous_Z_ranges,args=tuple(x.values[0]))[0]
            return causal_effect
        else:
            return self.conditional_density.pdf(exog_predict=x[self.causes],endog_predict=x[self.effects])

       
 
    def expected_value( self, x):
        """
        Currently, this does the whole sum/integral over the cube support of Z.
        We may be able to improve this by taking into account how the joint
        and conditionals factorize, and/or finding a more efficient support.
        
        This should be reasonably fast for |Z| <= 2 or 3, and small enough discrete
        variable cardinalities.  It runs in O(n_1 n_2 ... n_k) in the cardinality of
        the discrete variables, |Z_1| = n_1, etc.  It likewise runs in O(V^n) for n
        continuous Z variables.  Factorizing the joint/conditional distributions in
        the sum could linearize the runtime.
        """
        causal_effect = 0.
        x = x[self.causes]
        if self.discrete_Z:
            discrete_variable_ranges = [ xrange(*(int(self.support[variable][0]), int(self.support[variable][1])+1)) for variable in self.discrete_Z]
            for z_vals in itertools.product(*discrete_variable_ranges):
                z_discrete = pd.DataFrame({k : [v] for k, v in zip(self.discrete_Z, z_vals)})
                if self.continuous_Z:
                    continuous_Z_ranges = [self.support[variable] for variable in self.continuous_Z]
                    args = z_discrete.join(x).values[0]
                    causal_effect += nquad(self.expectation_integration_function,continuous_Z_ranges,args=args)[0]
                else:
                    z_discrete = z_discrete[self.admissable_set]
                    exog_predictors = x.join(z_discrete)[self.conditional_density_vars]
                    causal_effect += self.conditional_expectation.fit(data_predict=exog_predictors.values)[0] * self.density.pdf(data_predict=z_discrete.values)
            return causal_effect
        elif self.continuous_Z:
            continuous_Z_ranges = [self.support[var] for var in self.continuous_Z]
            causal_effect = nquad(self.expectation_integration_function,continuous_Z_ranges,args=tuple(x.values[0]))[0]
            return causal_effect
        else:
            return self.conditional_expectation.fit(data_predict=x[self.causes])[0]
       
