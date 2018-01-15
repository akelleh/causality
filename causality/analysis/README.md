# Analysis

This module contains tools for using the Robin's G-Formula and arbitrary machine learning estimators to estimate and plot causal effects. By "causal effect" we mean the distribution or conditional expectation of Y given X, controlling for an admissable set of covariates, Z, to make the effect identifiable. For a primer on choosing these Z variables, check out the article [here](https://medium.com/@akelleh/a-technical-primer-on-causality-181db2575e41).

More intuitively, you want to estimate the effect of X on Y, but you know you need to control for some set of confounders, Z, to get the true effect. Otherwise, you expect there to be confounding bias.

# The `CausalDataFrame`

The `CausalDataFrame` is an extension of the `pandas.DataFrame`, so you can intialize it as you normally would intialize a `pandas.DataFrame`, e.g.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
from causality.analysis.dataframe import CausalDataFrame

N = 1000

z = np.random.normal(1., size=N)
x = np.random.binomial(1, p=1./(1. + np.exp(-z/.1)))
y = x + z + np.random.normal(size=N)

# It's easy to create a data frame
df = CausalDataFrame({'x': x, 'y': y, 'z': z})

# and the interface to zplot is basically the same as the pandas.DataFrame.plot method!
df.zplot(x='x', y='y', z_types={'z': 'c'}, z=['z'], kind='bar', bootstrap_samples=500); pp.ylabel("$E[Y|do(X=x)]$"); pp.show()

```
![The causal estimate](https://github.com/akelleh/causality/blob/CAUS-18-update-readmes/causality/analysis/img/discrete_zplot.png)

 You can also still use all of the usual methods, for example to get a naive plot for comparison.

 ```python
df.groupby('x').mean().reset_index().plot(x='x', y='y', kind='bar'); pp.ylabel("$E[Y|X=x]$"); pp.show()
 ```
 ![The naive estimate](https://github.com/akelleh/causality/blob/CAUS-18-update-readmes/causality/analysis/img/discrete_zplot_naive.png)

The correct answer in this example is that if you intervene to set the value of `x` to `x=0`, you'll find (on average) `y=1`. If you set `x=1`, you'll find (on average) `y=2`. You can see the causal `zplot` method finds the correct answer, within the 95% confidence level. You can see naive observational estimate has much lower `y` at `x=0`!
