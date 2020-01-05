# autorank

## Summary

Autorank is a simple Python package with one task: simplify the comparison between (multiple) paired populations. 
The package basically implements Demsars guidelines for the the comparison of classifiers. This package provides the single function 
`autorank` which automatically determines suitable statistical tests, confidence intervals, etc. based on the normality and homogeneity of 
the data. 

## Installation

Autorank is available on PyPi and can be installed using pip.

```
pip install autorank
```

## Description

Autorank uses the following strategy for the statistical comparison of paired populations:
- First all populations are checked with the Shapiro-Wilk test for normality. We use Bonferoni correction for these
  tests, i.e., alpha/#populations.
- If all columns are normal, we use Bartlett's test for homogeneity, otherwise we use Levene's test.
- Based on the normality and the homogeneity, we select appropriate tests, effect sizes, and methods for determining
  the confidence intervals of the central tendency.

If all columns are normal, we calculate:
- The mean value as central tendency.
- The empirical standard deviation as measure for the variance.
- The confidence interval for the mean value.
- The effect size in comparison to the highest mean value using Cohen's d.

If at least one column is not normal, we calculate:
- The median as central tendency.
- The median absolute deviation from the median as measure for the variance.
- The confidence interval for the median.
- The effect size in comparison to the highest ranking approach using Cliff's delta.

For the statistical tests, there are four variants:
- If there are two populations and both populations are normal, we use the paired t-test.
- If there are two populations and at least one populations is not normal, we use Wilcoxon's signed rank test.
- If there are more than two populations and all populations are normal and homoscedastic, we use repeated measures
  ANOVA with Tukey's HSD as post-hoc test.
- If there are more than two populations and at least one populations is not normal or the populations are
  heteroscedastic, we use Friedman's test with the Nemenyi post-hoc test.
  
## Usage Example

The following example shows the usage of `autorank`.
```python
import numpy as np
import pandas as pd
from autorank import autorank

np.random.seed(42)
std = 0.3
means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
data = pd.DataFrame()
for i, mean in enumerate(means):
    data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
autorank(data, 0.05, True)
```

## Planned Features

In the (hopefully near) future, Autorank will be extended to support the generation of appropriate visualizations for the tests and for 
results tables. 

## License

Autorank is published under the Apache 2.0 Licence.
