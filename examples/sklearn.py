# Requires sklearn!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification, make_hastie_10_2
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from autorank import autorank, create_report


clf_names = ["Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes"]

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    GaussianNB()]

data_names = []
datasets = []
for i in range(0, 5):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=i, n_clusters_per_class=1)
    rng = np.random.RandomState(i)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    data_names.append('moons_%i' % i)
    data_names.append('circles_%i' % i)
    data_names.append('linsep_%i' % i)
    data_names.append('hastie_%i' % i)
    datasets.append(make_moons(noise=0.3, random_state=i))
    datasets.append(make_circles(noise=0.2, factor=0.5, random_state=i))
    datasets.append(linearly_separable)
    datasets.append(make_hastie_10_2(1000, random_state=i))

results = pd.DataFrame(index=data_names, columns=clf_names)

for data_name, data in zip(data_names, datasets):
    X, y = data
    X = StandardScaler().fit_transform(X)
    scores = []
    # iterate over classifiers
    for clf_name, clf in zip(clf_names, classifiers):
        print("Applying %s to %s" % (clf_name, data_name))
        res = cross_val_score(estimator=clf, X=X, y=y, cv=10, scoring='accuracy')
        results.at[data_name, clf_name] = res.mean()

res = autorank(results)
create_report(res)
plot_stats(res)
plt.show()
latex_table(res)