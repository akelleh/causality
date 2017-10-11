# Causality

This package contains tools for causal analysis using observational (rather than experimental) datasets.

## Installation

Assuming you have pip installed, just run
```
pip install causality 
```

## Measuring Causal Effects

the [`causality.estimation`](https://github.com/akelleh/causality/tree/master/causality/estimation) module contains tools for estimating causal effects from observational and experimental data. Most tools are parametric, like `PropensityScoreMatching`, and can be found in `causality.estimation.parametric`. Other models are non-parametric, and rely on directly estimating densities and using the g-estimation approach.


## DAG Inference

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
x5 = x4 + numpy.random.normal(size=SIZE)

# load the data into a dataframe:
X = pd.DataFrame({'x1' : x1, 'x2' : x2, 'x3' : x3, 'x4' : x4, 'x5' : x5})

# define the variable types: 'c' is 'continuous'.  The variables defined here
# are the ones the search is performed over  -- NOT all the variables defined
# in the data frame.
variable_types = {'x1' : 'c', 'x2' : 'c', 'x3' : 'c', 'x4' : 'c', 'x5' : 'c'}

# run the search
ic_algorithm = IC(RobustRegressionTest)
graph = ic_algorithm.search(X, variable_types)
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

I've recently added adjustment for direct causes, where you can estimate the causal effect of fixing a set of X variables on a set of Y variables by adjusting for the parents of X in your graph.  Using the dataset above, you can run this like
```python
from causality.nonparametric.causal_reg import AdjustForDirectCauses
from networkx import DiGraph

g = DiGraph()

g.add_nodes_from(['x1','x2','x3','x4', 'x5'])
g.add_edges_from([('x1','x2'),('x1','x3'),('x2','x4'),('x3','x4')])
adjustment = AdjustForDirectCauses(g, X, ['x2'],['x3'],variable_types=variable_types)
```

Then, you can see the set of variables being adjusted for by
```python
>>> print adjustment.admissable_set
set(['x1'])
```
If we hadn't adjusted for `'x1'` we would have incorrectly found that `'x2'` had a causal effect on `'x3'` due to the counfounding pathway `x2, x1, x3`.  Adjustment for `'x1'` removes this bias.

You can see the causal effect of intervention, `P(x3|do(x2))` using the measured causal effect in `adjustment`,
```python
>>> x = pd.DataFrame({'x2' : [0.], 'x3' : [0.]})
>>> print adjustment.effect.pdf(x)
0.268915603296
```

Which is close to the correct value of `0.282` for a gaussian with mean 0. and variance 2.  If you adjust the value of `'x2'`, you'll find that the probability of `'x3'` doesn't change.  This is untrue with just the conditional distribution, `P(x3|x2)`, since in this case, observation and intervention are not equivalent.

## Other Notes

This repository is in its early phases.  The run-time for the tests is long.  Many optimizations will be made in the near future, including
* Implement fast mutual information calculation, O( N log N )
* Speed up integrating out variables for controlling
* Take a user-supplied graph, and find the set of admissable sets
* Front-door criterion method for determining causal effects

Pearl, Judea. _Causality_.  Cambridge University Press, (2000).
