# Causality

This package contains tools for causal analysis using observational (rather than experimental) datasets.

## Installation

Assuming you have pip installed, just run
```
pip install causality 
```


## Causal Inference

The `causality.inference` module will contain various algorithms for inferring causal DAGs.  Currently (2016/01/23), the only algorithm implemented is the IC\* algorithm from Pearl (2000).  It has decent test coverage, but feel free to write some more!  I've left some stubs in `tests/unit/test\_IC.py`.

To run a graph search on a dataset, you can use the algorithms like (using IC\* as an example):

```python
import numpy
import pandas as pd

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

# run the search
ic_algorithm = IC(RobustRegressionTest, X, variable_types)
graph = ic_algorithm.search()
```

Now, we have the inferred graph stored in `graph`.  In this graph, each variable is a node (named from the DataFrame columns), and each edge represents statistical dependence between the nodes that can't be eliminated by conditioning on the variables specified for the search.  If an edge can be oriented with the data available, the arrowhead is indicated in `'arrows'`.  If the edge also satisfies the local criterion for genuine causation, then that directed edge will have `marked=True`.  If we print the edges from the result of our search, we can see which edges are oriented, and which satisfy the local criterion for genuine causation:
```python
>>> graph.edges(data=True)
[('x2', 'x1', {'arrows': [], 'marked': False}), 
 ('x2', 'x4', {'arrows': ['x4'], 'marked': False}), 
 ('x3', 'x1', {'arrows': [], 'marked': False}), 
 ('x3', 'x4', {'arrows': ['x4'], 'marked': False}), 
 ('x4', 'x5', {'arrows': ['x5'], 'marked': True})]
```

We can see the edges from `'x2'` to `'x4'`, `'x3'` to `'x4'`, and `'x4'` to `'x5'` are all oriented toward the second of each pair.  Additionally, we see that the edge from `'x4'` to `'x5'` satisfies the local criterion for genuine causation.  This matches the structure given in figure `2.3(d)` in Pearl (2000).


## Nonparametric Effects Estimation

The `causality.nonparametric` module contains a tool for non-parametrically estimating a causal distribution from an observational data set. You can supply an "admissable set" of variables for controlling, and the measure either the causal effect distribution of an effect given the cause, or the expected value of the effect given the cause.

This repository is in its early phases.  The run-time for the tests is long.  Many optimizations will be made over the course of the coming weeks, including
* Implement fast mutual information calculation, O( N log N )
* Speed up integrating out variables for controlling
* Take a user-supplied graph, and deduce the admissable set
* Front-door criterion method for determining causal effects

Pearl, Judea. _Causality_.  Cambridge University Press, (2000).
