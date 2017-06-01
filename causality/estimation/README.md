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
asdf

```

### nonparametric


### adjustments