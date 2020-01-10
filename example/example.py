import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autorank.autorank import autorank, create_report, plot_stats

np.random.seed(42)
pd.set_option('display.max_columns', 7)
std = 0.3
means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
sample_size = 50
data = pd.DataFrame()
for i, mean in enumerate(means):
    data['pop_%i' % i] = np.random.normal(mean, std, sample_size).clip(0, 1)

res = autorank(data, alpha=0.05, verbose=True)
create_report(res)
plot_stats(res)
plt.show()
