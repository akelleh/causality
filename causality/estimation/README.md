# causality.estimation

This module is for causal effect estimation! When you run a randomized controlled experiment (e.g. an A/B test), you know that people in the test group are, on average, similar to people in the control group. For any given covariate, Z, you expect that the average of Z in each group is the same. 

When you only have observational data, you can't be sure that the group assignments are independent of other covariates. The worst case scenario is that the effect of the treatment is different between the test and the control group. Then, the treatment's effect on the test group no longer represents the average effect of the treatment over everyone. 

In a drug trial, for example, people might take the drug if they've taken it in the past and know it works, and might not take it if they've taken it before and found that it doesn't work. Then, you'll find that the drug is much more effective for people who normally take it (your observational test group) than people who don't normally take it. If you enacted a policy where everyone who gets sick gets the drug, then you'll find it much less effective on average than it would have appeared from your observational data: your controlled intervention not gives the treatment to people it has no effect on!

Our goal, then, is to take observational data and be able to answer questions about controlled interventions. There are some excellent books on the subject if you're interested in all of the details of how these methods work, but this package's documentation will give high-level explanations with a focus on application. Some excellent references for more depth are Morgan and Winship's [_Counterfactuals and Causal Inference_](https://www.amazon.com/Counterfactuals-Causal-Inference-Principles-Analytical/dp/1107694167), Hernan's [_Causal Inference_](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/), Pearl's groundbreaking (but extremely difficult, and not application-focused) [_Causality_] (https://www.amazon.com/Causality-Reasoning-Inference-Judea-Pearl/dp/052189560X), or Imbens and Rubin's [_Causal Inference_](https://www.amazon.com/Causal-Inference-Statistics-Biomedical-Sciences/dp/0521885884/ref=sr_1_1?s=books&ie=UTF8&qid=1496343137&sr=1-1&keywords=imbens+and+rubin).  

There are some critical caveats to all of these approaches. First, if you don't know what variables to control for, you're often out of luck. This is true of all methods that rely on controlling. Other methods, like Instrumental Variables, or mechanism-based methods, get around this by instead making certain assumptions about the structure of the system you're studying. We'll make a note of which type of algorithm you're dealing with in the tutorial for that algorithm, but it should be relatively clear from the context. This distinction is a little artificial, since you can often do controlling alongside approaches that rely on structural assumptions.


## Sub-modules:
### parametric
Most of the classic models you'd like to use are probably in this portion of the package. Currently, these include propensity score matching and difference-in-differences. 

#### PropensityScoreMatching

Propensity score matching tries to attack the problem of dissimilar test and control groups directly. You have the option of making the test group more similar to the control group, or vice versa. When we're talking about similarity, we mean similar by some metric. In the case of propensity score matching, that metric is the "propensity score". The propensity score is the probability a unit is assigned to the treatment given a set of covariates, $$P(D|Z_1, Z_2, ..., Z_n)$$.

If the variables $$Z_i$$ account for why the unit is assigned to the test or control group, then we can take all the control units that look like they should be test units, and throw away the rest! As you might guess, this can make inefficient use of our data, but it's a good method to use if you have the data for it.

There are a few diagnostics that help you figure out whether you've done a good job matching. Once you've done the matching, the distribution of the Z's between the test and control should end up pretty similar. The easiest trick is probably to examine the average value of each Z between the test and control group, and make sure most of the difference is gone. If so, your matching is probably okay. If not, you should play with the matching algorithm's parameters and try to do a better job.

Let's run through a quick example of propensity score matching to see how easy it can be!

First, we need to generate a data set that has some bias, since we're dealing with observational data. This will simulate an observational data set where the treatment's effectiveness varies depending on some other variables, Z. These will also correlate with whether a unit is assigned to the treatment or control group.

```python
import pandas as pd
import numpy as np
from causality.estimation.parametric import PropensityScoreMatching

N = 10000
z1 = np.random.normal(size=N)
z2 = np.random.normal(size=N)
z3 = np.random.normal(size=N)
arg = z1 + z2 + z3 #+ np.random.normal(size=N)
p = 1. / (1. + np.exp(-arg/4.))
d = np.random.binomial(1, p=p)

y0 = np.random.normal()
y1 = y0 + arg

y = d*y1 + (1-d)*y0

X = pd.DataFrame({'d': d, 'z1': z1, 'z2': z2, 'z3': z3, 'y': y, 'y0': y0, 'y1': y1, 'p': p})
```

The variable `y0` is the value that `y` would take if the unit is in the control group. The variable `y1` is the value the unit would take if it were in the test group. A unit can only be in one group when you measure its outcome, so you can only measure y = y0 or y = y1 in practice. These variables are called "potential outcomes," because they are the outcomes that are possible for each unit, depending on which treatment state the unit is assigned to.

Normally, you can't observe potential outcomes. The only reason we have them here is because we wrote the data-generating process.  
 
The variable `d` is a `1` if a unit is in the test group, and a `0` if they are in the control group. The outcome is defined using this variable as a switch to pick out the right potential outcome for the units treatment assignment. If `d=1`, then `y=y1`, and `d=0` implies `y=y0`.

Notice that these `z` variables determine both whether a unit will be assigned to the treatment (the higher the `z`s are, the higher `p` is), and the outcome (`arg` is just the sum of the `z`s, so higher `z` means higher treatment effectiveness.).  This results in bias if you just use a naive estimate for the average treatment effectiveness:

```python
> X[X['d'] == 1].mean()['y'] - X[X['d'] == 0].mean()['y']
0.3648
```
Taking a look at the true average treatment effect, the average difference between `(y1 - y0).mean()`, we can read off that it's just the average of `arg`. `arg` is the sum of three normal variables, so has mean zero. Thus, there is no average treatment effect! Our naive estimate of `0.36` is far from the true value. We can calculate the true value directly:
 
```python
> (y1 - y0).mean()
-0.0002
```
 
which is only different from zero due to sampling error.

Since we can't measure these potential outcome variables, we want to use PropensityScoreMatching to control for the variables that cause the bias. We can do this very easily!
```python
> matcher = PropensityScoreMatching()
> matcher.estimate_ATE(X, 'd', 'y', {'z1': 'c', 'z2': 'c', 'z3': 'c'})
-0.00011
```
and so we get the right average treatment effect (within measurement error). Bootstrap error bars are coming soon.

Here, we put in a dataframe, `X`, that contains a binary treatment assignment column, `'d'`, an outcome column, `'y'`, and a dictionary of variables to control for. The keys are the names of the columns to use for controlling, and the values are one of `('c', 'o', 'u')` corresponding to continuous, ordered discrete, or unordered discrete variables, respectively.
 
When you pass these arguments, the method builds a logistic regression model using the control variables to predict treatment assignment. The probabilities of treatment assingment, a.k.a. propensity scores, are used to match the treatment and control units using nearest neighbors (with a heuristic to improve matching for discrete variables). The matches are then used to calculate treatment effects on typical treated individuals, and typical control individuals, and then these effects are weighted and averaged to get the averate treatment effect on the whole population. This should agree with the value (within sampling error) of `(y1 - y0).mean()`, which is what we were trying to calculate!

### nonparametric


### adjustments