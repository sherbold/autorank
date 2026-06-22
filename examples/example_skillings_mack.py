"""Example: ranking populations from *incomplete* data (with missing values).

When the data frame contains missing values (NaNs), autorank automatically
switches the frequentist omnibus test from Friedman to the Skillings-Mack test,
a generalization of the Friedman test that tolerates observations missing at
random. The selection is automatic; there is no parameter to force it. The
p-value is estimated with the assumption-free permutation variant of the test,
so passing a ``random_state`` makes the result reproducible.

The Skillings-Mack test is an experimental feature (see
``autorank._utils_experimental``) intended to be moved upstream.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank import autorank, create_report, plot_stats, latex_table

np.random.seed(42)
pd.set_option('display.max_columns', 7)
std = 0.3
means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
sample_size = 50
data = pd.DataFrame()
for i, mean in enumerate(means):
    data['pop_%i' % i] = np.random.normal(mean, std, sample_size).clip(0, 1)

# Introduce missing values at random (keeping at least two observations per row).
# This makes the design incomplete, which triggers the Skillings-Mack branch.
rng = np.random.default_rng(42)
n_missing = 20
rows = rng.integers(0, sample_size, size=n_missing)
cols = rng.integers(0, len(means), size=n_missing)
for r, c in zip(rows, cols):
    data.iat[r, c] = np.nan

print("Number of missing values per population:")
print(data.isnull().sum())
print()

# random_state makes the permutation p-value of the Skillings-Mack test reproducible
res = autorank(data, alpha=0.05, verbose=False, random_state=42)
print(res)
create_report(res)
plot_stats(res)
plt.show()
latex_table(res)
