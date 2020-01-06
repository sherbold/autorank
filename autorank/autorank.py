"""
Automated ranking of populations for ranking them. This is basically an implementation of Demsar's
Guidelines for the comparison of multiple classifiers. Details can be found in the description of the autorank function.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import AnovaRM
from collections import namedtuple


def _cohen_d(x, y):
    """
    Calculate the effect size using Cohen's d
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


def _cliffs_delta(x, y):
    """
    Calculates Cliff's delta.
    """
    delta = 0
    for x_val in x:
        result = 0
        for y_val in y:
            if y_val > x_val:
                result -= 1
            elif x_val > y_val:
                result += 1
        delta += result / len(y)
    if abs(delta) < 10e-16:
        # due to minor rounding errors
        delta = 0
    else:
        delta = delta / len(x)
    return delta


def _effect_level(effect_size, method='cohend'):
    """
    Determines magnitude of effect size.
    """
    if not isinstance(method, str):
        raise TypeError('method must be of type str')
    if method not in ['cohend', 'cliffdelta']:
        raise ValueError("method must be one of the following strings: 'cohend', 'cliffdelta'")
    effect_size = abs(effect_size)
    if method == 'cliffdelta':
        if effect_size < 0.147:
            return 'negligible'
        elif effect_size < 0.33:
            return 'small'
        elif effect_size < 0.474:
            return 'medium'
        else:
            return 'large'
    if method == 'cohend':
        if effect_size < 0.2:
            return 'negligible'
        elif effect_size < 0.5:
            return 'small'
        elif effect_size < 0.8:
            return 'medium'
        else:
            return 'large'


def _critical_distance(alpha, k, n):
    """
    Determines the critical distance for the Nemenyi test with infinite degrees of freedom.
    """
    return qsturng(1 - alpha, k, np.inf) * np.sqrt(k * (k + 1) / (12 * n))


def _confidence_interval(data, alpha, is_normal=True):
    """
    Determines the confidence interval.
    """
    if is_normal:
        mean = data.mean()
        ci_range = data.sem() * stats.t.ppf((1 + 1 - alpha) / 2, len(data) - 1)
        return mean - ci_range, mean + ci_range
    else:
        quantile = stats.norm.ppf(1 - (alpha / 2))
        r = (len(data) / 2) - (quantile * np.sqrt(len(data) / 2))
        s = 1 + (len(data) / 2) + (quantile * np.sqrt(len(data) / 2))
        sorted_data = data.sort_values()
        lower = sorted_data.iloc[int(round(r))]
        upper = sorted_data.iloc[int(round(s))]
        return lower, upper


_ComparisonResult = namedtuple('ComparisonResult', ('rankdf', 'pvalue', 'cd', 'omnibus', 'posthoc'))


def _rank_two(data, alpha, verbose, all_normal):
    """
    Uses paired t-test for normal data and Wilcoxon's signed rank test for other distributions.
    """
    if verbose:
        if all_normal:
            print("Using paired t-test")
        else:
            print("Using Wilcoxon's signed rank test (one-sided)")
    larger = np.argmax(data.median().values)
    smaller = int(bool(larger - 1))
    if all_normal:
        omnibus = 'ttest'
        pval = stats.ttest_rel(data.iloc[:, larger], data.iloc[:, smaller]).pvalue
    else:
        omnibus = 'wilcoxon'
        pval = stats.wilcoxon(data.iloc[:, larger], data.iloc[:, smaller], alternative='greater').pvalue
    if verbose:
        if pval >= alpha:
            print(
                "Fail to reject null hypothesis that there is no difference between the distributions (p=%f)" % pval)
        else:
            print("Rejecting null hypothesis that there is no difference between the distributions (p=%f)" % pval)
    rankdf = _create_result_df_skeleton(data, alpha, all_normal)
    return _ComparisonResult(rankdf, pval, None, omnibus, None)


def _rank_multiple_normal_homoscedastic(data, alpha=0.05, verbose=False):
    """
    Analyzes data using repeated measures ANOVA and Tukey HSD.
    """
    stacked_data = data.stack().reset_index()
    stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                                'level_1': 'treatment',
                                                0: 'result'})
    anova = AnovaRM(stacked_data, 'result', 'id', within=['treatment'])
    pval = anova.fit().anova_table['Pr > F'].iat[0]
    if verbose:
        if pval >= alpha:
            print(
                "Fail to reject null hypothesis that there is no difference between the distributions (p=%f)" % pval)
        else:
            print("Rejecting null hypothesis that there is no difference between the distributions (p=%f)" % pval)
            print(
                "Using Tukey HSD post hoc test.",
                "Differences are significant if the confidence intervals of the mean values are not overlapping.")

    multicomp = MultiComparison(stacked_data['result'], stacked_data['treatment'])
    tukey_res = multicomp.tukeyhsd()
    # must create plot to get confidence intervals
    tukey_res.plot_simultaneous()
    # delete plot instead of showing
    plt.clf()
    rankmat = data.rank(axis='columns', ascending=False)
    meanranks = rankmat.mean().sort_values()
    rankdf = pd.DataFrame(index=meanranks.index)
    rankdf['meanrank'] = meanranks
    rankdf = _create_result_df_skeleton(data, None, True)
    for population in rankdf.index:
        mean = data.loc[:, population].mean()
        ci_range = tukey_res.halfwidths[data.columns.get_loc(population)]
        lower, upper = mean - ci_range, mean + ci_range
        rankdf.at[population, 'ci_lower'] = lower
        rankdf.at[population, 'ci_upper'] = upper
    return _ComparisonResult(rankdf, pval, None, 'anova', 'tukeyhsd')


def _rank_multiple_nonparametric(data, alpha, verbose, all_normal):
    """
    Analyzes data following Demsar using Friedman-Nemenyi.
    """
    if verbose:
        print("Using Friedman test as omnibus test")
    pval = stats.friedmanchisquare(*data.transpose().values).pvalue
    if verbose:
        if pval >= alpha:
            print("Fail to reject null hypothesis that there is no difference between the distributions (p=%f)" % pval)
        else:
            print("Rejecting null hypothesis that there is no difference between the distributions (p=%f)" % pval)
            print(
                "Using Nemenyi post-hoc test.",
                "Differences are significant,"
                "if the distance between the mean ranks is greater than the critical distance.")
    cd = _critical_distance(alpha, k=len(data.columns), n=len(data))
    rankdf = _create_result_df_skeleton(data, alpha, all_normal)
    return _ComparisonResult(rankdf, pval, cd, 'friedman', 'nemenyi')


def _create_result_df_skeleton(data, alpha, all_normal):
    """
    Creates data frame for results. CI may be left empty in case alpha is None
    """
    if all_normal:
        effsize_method = 'cohend'
    else:
        effsize_method = 'cliffdelta'

    rankmat = data.rank(axis='columns', ascending=False)
    meanranks = rankmat.mean().sort_values()
    if effsize_method == 'cohend':
        rankdf = pd.DataFrame(index=meanranks.index,
                              columns=['meanrank', 'mean', 'std', 'ci_lower', 'ci_upper', 'effect_size', 'magnitude'])
    else:
        rankdf = pd.DataFrame(index=meanranks.index,
                              columns=['meanrank', 'median', 'mad', 'ci_lower', 'ci_upper', 'effect_size', 'magnitude'])
    rankdf['meanrank'] = meanranks
    for population in rankdf.index:
        if effsize_method == 'cohend':
            effsize = _cohen_d(data.loc[:, rankdf.index[0]], data.loc[:, population])
            rankdf.at[population, 'mean'] = data.loc[:, population].mean()
            rankdf.at[population, 'std'] = data.loc[:, population].std()
        elif effsize_method == 'cliffdelta':
            effsize = _cliffs_delta(data.loc[:, rankdf.index[0]], data.loc[:, population])
            rankdf.at[population, 'median'] = data.loc[:, population].median()
            rankdf.at[population, 'mad'] = stats.median_absolute_deviation(data.loc[:, population])
        else:
            raise ValueError("Unknown effsize method, this should not be possible.")
        rankdf.at[population, 'effect_size'] = effsize
        rankdf.at[population, 'magnitude'] = _effect_level(effsize, effsize_method)
        if alpha is not None:
            lower, upper = _confidence_interval(data.loc[:, population], alpha / len(data.columns),
                                                is_normal=all_normal)
            rankdf.at[population, 'ci_lower'] = lower
            rankdf.at[population, 'ci_upper'] = upper
    return rankdf


RankResult = namedtuple('RankResult', (
    'rankdf', 'pvalue', 'cd', 'omnibus', 'posthoc', 'all_normal', 'pvals_shapiro', 'homoscedastic', 'pval_homogeneity',
    'homogeneity_test'))


def autorank(data, alpha=0.05, verbose=False):
    """
    Automatically compares populations defined in a block-design data frame. Each column in the data frame contains
    the samples for one population. The data must not contain any NaNs. The data must have at least five measurements,
    i.e., rows. The current version is only reliable for less than 5000 measurements.

    The following approach is implemented by this function
    - First all columns are checked with the Shapiro-Wilk test for normality. We use Bonferoni correction for these
      tests, i.e., alpha/len(data.columns).
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
    - If there are two populations (columns) and both populations are normal, we use the paired t-test.
    - If there are two populations and at least one populations is not normal, we use Wilcoxon's signed rank test.
    - If there are more than two populations and all populations are normal and homoscedastic, we use repeated measures
      ANOVA with Tukey's HSD as post-hoc test.
    - If there are more than two populations and at least one populations is not normal or the populations are
      heteroscedastic, we use Friedman's test with the Nemenyi post-hoc test.

    :param data: Pandas DataFrame, where each column contains a population and each row contains the paired measurements
    for the populations.
    :param alpha: Significance level. We internally use correction to ensure that all results (incl. confidence
    intervals) together fulfill this confidence level.
    :param verbose: If true, details about the ranking are printed.
    :return: A named tuple with the following entries:
       - rankdf: Ranked populations including statistics about the populations
       - pvalue: p-value of the omnibus test for the difference in central tendency between the populations
       - omnibus: String with omnibus test that is used for the test of a difference ein the central tendency.
       - posthoc: String with the posthoc tests that was used. The posthoc test is performed even if the omnibus test is
         not significant. The results should only be used if the p-value of the omnibus test indicates significance.
       - cd: The critical distance of the Nemenyi posthoc test, if it was used. Otherwise None.
       - all_normal: True if all populations are normal
       - pvals_shapiro: p-values of the Shapiro-Wilk tests for normality (sorted by the order of the input columns)
       - homoscedastic: True if populations are homoscedastic
       - pval_homogeneity: p-value of the test for homogeneity.
       - homogeneity_test: Test used for homogeneity.
    """

    # validate inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    if len(data.columns) < 2:
        raise ValueError('requires at least two classifiers (i.e., columns)')
    if len(data) < 5:
        raise ValueError('requires at least five performance estimations (i.e., rows)')

    if not isinstance(alpha, float):
        raise TypeError('alpha must be a float')
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError('alpha must be in the open interval (0.0,1.0)')

    if not isinstance(verbose, bool):
        raise TypeError('verbose must be bool')

    # Bonferoni correction for normality tests
    alpha_normality = alpha / len(data.columns)

    # Check pre-conditions of statistical tests
    all_normal = True
    pvals_shapiro = []
    for column in data.columns:
        w, pval_shapiro = stats.shapiro(data[column])
        pvals_shapiro.append(pval_shapiro)
        if pval_shapiro < alpha_normality:
            all_normal = False
            if verbose:
                print("Rejecting null hypothesis that data is normal for column %s (p=%f<%f)" % (
                    column, pval_shapiro, alpha_normality))
        elif verbose:
            print("Fail to reject null hypothesis that data is normal for column %s (p=%f>=%f)" % (
                column, pval_shapiro, alpha_normality))

    if all_normal:
        if verbose:
            print("Using Bartlett's test for homoscedacity of normally distributed data")
        homogeneity_test = 'bartlett'
        pval_homogeneity = stats.bartlett(*data.transpose().values).pvalue
    else:
        if verbose:
            print("Using Levene's test for homoscedacity of non-normal data.")
        homogeneity_test = 'levene'
        pval_homogeneity = stats.levene(*data.transpose().values).pvalue
    var_equal = pval_homogeneity >= alpha
    if verbose:
        if var_equal:
            print("Fail to reject null hypothesis that all variances are equal (p=%f>=%f)" % (pval_homogeneity, alpha))
        else:
            print("Rejecting null hypothesis that all variances are equal (p=%f<%f)" % (pval_homogeneity, alpha))

    # Select appropriate tests
    if len(data.columns) == 2:
        res = _rank_two(data, alpha, verbose, all_normal)
    else:
        if all_normal and var_equal:
            res = _rank_multiple_normal_homoscedastic(data, alpha, verbose)
        else:
            res = _rank_multiple_nonparametric(data, alpha, verbose, all_normal)

    return RankResult(res.rankdf, res.pvalue, res.cd, res.omnibus, res.posthoc, all_normal, pvals_shapiro, var_equal,
                      pval_homogeneity, homogeneity_test)
