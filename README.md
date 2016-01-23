# causality
Tools for causal analysis

Nonparametric contains a tool for non-parametrically estimating a causal distribution from an observational data set. You can supply an "admissable set" of variables for controlling, and the measure either the causal effect distribution of an effect given the cause, or the expected value of the effect given the cause.

This repository is in its early phases.  The run-time for the tests is long.  Many optimizations will be made over the course of the coming weeks, including
* Implement fast mutual information calculation, O( N log N )
* Speed up integrating out variables for controlling
* Take a user-supplied graph, and deduce the admissable set
* Front-door criterion method for determining causal effects



# Causal Inference

The package will contain various algorithms for inferring causal DAGs.  Currently (2016/01/23), the only algorithm implemented is the IC\* algorithm from Pearl (2000).  

To run a graph search on a dataset, you can use the algorithms like (using IC\* as an example):

```
from causality.inference.search import IC
from causality.inference.independence_tests import RobustRegressionTest

# generate some toy data:
SIZE = 2000
x1 = numpy.random.normal(size=SIZE)
x2 = x1 + numpy.random.normal(size=SIZE)
x3 = x1 + numpy.random.normal(size=SIZE)
x4 = x2 + x3 + numpy.random.normal(size=SIZE)
x5 = x5 = x4 + numpy.random.normal(size=SIZE)

# load the data into a dataframe:
X = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3, 'x4' : x4, 'x5' : x5})

# define the variable types: 'c' is 'continuous'.  The variables defined here
# are the ones the search is performed over  -- NOT all the variables defined
# in the data frame.
variable_types = {'x1' : 'c', 'x2' : 'c', 'x3' : 'c', 'x4' : 'c', 'x5' : 'c'}

ic_algorithm = IC(RobustRegressionTest, X, variable_types)
graph = ic_algorithm.search()
```

