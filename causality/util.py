import pandas as pd

def bootstrap_statistic(df, function,  bootstrap_samples=1000, lower_confidence=0.025, upper_confidence=0.975, values=False):
    """ This gives bootstrap confidence intervals on the population value
        of function given the sample represented by iterable."""
    statistics = []
    for _ in range(bootstrap_samples):
      sampled_df = df.sample(n=len(df), replace=True)
      statistics.append(function(sampled_df))
    samples = pd.Series(statistics)
    if values:
        return samples
    else:
        cis = samples.quantile([lower_confidence,upper_confidence])
        lower_ci = cis[lower_confidence]
        expected = samples.mean()
        upper_ci = cis[upper_confidence]
        return lower_ci, expected, upper_ci