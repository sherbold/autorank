"""
Automated ranking of populations for ranking them. This is basically an implementation of Demsar's
Guidelines for the comparison of multiple classifiers. Details can be found in the description of the autorank function.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import AnovaRM
from collections import namedtuple

from autorank._util import cd_diagram, ci_plot, get_sorted_rank_groups


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
    plt.close()
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
    'homogeneity_test', 'alpha', 'alpha_normality', 'num_samples'))


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
       - alpha: Significance level that was used. Same as input.
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
                      pval_homogeneity, homogeneity_test, alpha, alpha_normality, len(data))


def plot_stats(result, allow_insignificant=False, ax=None, width=None):
    """
    Creates a plot that supports the analysis of the results of the statistical test. The plot depends on the
    statistical test that was used.
    - Creates a Confidence Interval (CI) plot for a paired t-test between two normal populations. The confidence
     intervals are calculated with Bonferoni correction, i.e., a confidence level of alpha/2.
    - Creats a CI plot for Tukey's HSD as post-hoc test with the confidence intervals calculated using the HSD approach
     such that the family wise significance is alpha.
    - Creates Critical Distance (CD) diagrams for the Nemenyi post-hoc test. CD diagrams visualize the mean ranks of
     populations. Populations that are not significantly different are connected by a horizontal bar.

    This function raises a ValueError if the omnibus test did not detect a significant difference. The allow_significant
    parameter allows the suppression of this exception and forces the creation of the plots.

    :param result: Result must be of type RankResult and should be the outcome of calling the autorank function.
    :param allow_insignificant: Forces plotting even if results are not significant. (Default: False)
    :param ax:  Matplotlib axis to which the results are added. A new figure with a single axis is
    created if None. (Default: None)
    :param width: Specifies the width of the created plot is not None. By default, we use a width of 6. The height is
    automatically determined, based on the type of plot and the number of populations. This parameter is ignored if ax
    is not None. (Default: None)
    :return: axis with the plot.
    """
    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")

    if result.pvalue >= result.alpha and not allow_insignificant:
        raise ValueError(
            "result is not significant and results of the plot may be misleading. If you want to create the plot "
            "regardless, use the allow_insignificant parameter to suppress this exception.")

    if ax is not None and width is not None:
        warnings.warn('width may be ignored because ax is defined.')
    if width is None:
        width = 6

    if result.omnibus == 'ttest':
        ax = ci_plot(result, True, ax, width)
    elif result.omnibus == 'wilcoxon':
        warnings.warn('No plot to visualize statistics for Wilcoxon test available. Doing nothing.')
    elif result.posthoc == 'tukeyhsd':
        ax = ci_plot(result, True, ax, width)
    elif result.posthoc == 'nemenyi':
        ax = cd_diagram(result, True, ax, width)
    return ax


def create_report(result):
    """
    Prints a report about the statistical analysis. 
    
    :param result: Result must be of type RankResult and should be the outcome of calling the autorank function.
    """
    # TODO add ranks to Nemenyi
    # TODO add CIs
    # TODO add effect sizes to multiple comparisons.
    def create_population_string(populations):
        if len(populations) == 1:
            popstr = " " + populations[0]
        elif len(populations) == 2:
            popstr = "s " + " and ".join(populations)
        else:
            popstr = "s " + ", ".join(populations[:-1]) + ", and " + populations[-1]
        return popstr

    print("The statistical analysis was conducted for %i populations with %i paired samples." % (len(result.rankdf),
                                                                                                 result.num_samples))
    if result.all_normal:
        print("We failed to reject the null hypothesis that the population is normal for all populations (adjusted "
              "alpha=%f). Therefore, we assume that all populations are normal." % result.alpha_normality)
    else:
        not_normal = []
        normal = []
        for i, pval in enumerate(result.pvals_shapiro):
            if pval < result.alpha_normality:
                not_normal.append(result.rankdf.index[i])
            else:
                normal.append(result.rankdf.index[i])
        print("We rejected the null hypothesis that the population is normal for the population%s "
              "(adjusted alpha=%f). Therefore, we assume that not all populations are "
              "normal." % (create_population_string(not_normal), result.alpha_normality))

    if len(result.rankdf) == 2:
        print("No check for homogeneity was required because we only have two populations.")
        if result.omnibus == 'ttest':
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (alpha=%f) of the paired t-test that the mean values of "
                      "the populations %s (M=%f, SD=%f) and %s (M=%f, SD=%f) are are equal. Therefore, we "
                      "assume that there is no statistically significant difference between the mean values of the "
                      "populations." % (result.alpha,
                                        result.rankdf.index[0], result.rankdf['mean'][0], result.rankdf['std'][0],
                                        result.rankdf.index[1], result.rankdf['mean'][1], result.rankdf['std'][1]))
            else:
                print("We reject the null hypothesis (alpha=%f) of the paired t-test that the mean values of the "
                      "populations %s (M=%f, SD=%f) and %s (M=%f, SD=%f) are "
                      "equal. Therefore, we assume that the mean value of %s is "
                      "significantly larger than the mean value of %s with a %s effect size (d=%f)."
                      % (result.alpha,
                         result.rankdf.index[0], result.rankdf['mean'][0], result.rankdf['std'][0],
                         result.rankdf.index[1], result.rankdf['mean'][1], result.rankdf['std'][1],
                         result.rankdf.index[0], result.rankdf.index[1],
                         result.rankdf.magnitude[1], result.rankdf.effect_size[1]))
        elif result.omnibus == 'wilcoxon':
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (alpha=%f) of Wilcoxon's signed rank test that "
                      "population %s (MD=%f, MAD=%f) is not greater than population %s (MD=%f, MAD=%f). Therefore, we "
                      "assume that there is no statistically significant difference between the medians of the "
                      "populations." % (result.alpha,
                                        result.rankdf.index[0], result.rankdf['median'][0], result.rankdf['mad'][0],
                                        result.rankdf.index[1], result.rankdf['median'][1], result.rankdf['mad'][1]))
            else:
                print("We reject the null hypothesis (alpha=%f) of Wilcoxon's signed rank test that population "
                      "%s (MD=%f, MAD=%f) is not greater than population %s (MD=%f, MAD=%f). Therefore, we assume "
                      "that the median of %s is "
                      "significantly larger than the median value of %s with a %s effect size (delta=%f)."
                      % (result.alpha,
                         result.rankdf.index[0], result.rankdf['median'][0], result.rankdf['mad'][0],
                         result.rankdf.index[1], result.rankdf['median'][1], result.rankdf['mad'][1],
                         result.rankdf.index[0], result.rankdf.index[1],
                         result.rankdf.magnitude[1], result.rankdf.effect_size[1]))
            pass
        else:
            raise ValueError('Unknown omnibus test for difference in the central tendency: %s' % result.omnibus)
    else:
        if result.all_normal:
            population_strings = []
            for population in result.rankdf.index:
                population_strings.append("%s (M=%f, SD=%f)" % (population, result.rankdf.at[population, 'mean'],
                                                                result.rankdf.at[population, 'std']))
            population_string = ", ".join(population_strings[:-1]) + ", and " + population_strings[-1]
        else:
            population_strings = []
            for population in result.rankdf.index:
                population_strings.append("%s (MD=%f, MAD=%f)" % (population, result.rankdf.at[population, 'median'],
                                                                  result.rankdf.at[population, 'mad']))
            population_string = ", ".join(population_strings[:-1]) + ", and " + population_strings[-1]

        if result.all_normal:
            if result.homoscedastic:
                print("We applied Bartlett's test for homogeneity and failed to reject the null hypothesis (alpha=%f) "
                      "that the data is homoscedastic. Thus, we assume that our data is "
                      "homoscedastic." % result.pval_homogeneity)
            else:
                print("We applied Bartlett's test for homogeneity and reject the null hypothesis (alpha=%f) that the"
                      "data is homoscedastic. Thus, we assume that our data is "
                      "heteroscedastic." % result.pval_homogeneity)

        if result.omnibus == 'anova':
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (alpha=%f) of the repeated measures ANOVA that there is "
                      "a difference between the mean values of the populations %s. Therefore, we "
                      "assume that there is no statistically significant difference between the mean values of the "
                      "populations." % (result.alpha, population_string))
            else:
                print("We reject the null hypothesis (alpha=%f) of the repeated measures ANOVA that there is "
                      "a difference between the mean values of the populations %s. Therefore, we "
                      "assume that there is a statistically significant difference between the mean values of the "
                      "populations." % (result.alpha, population_string))
                meanranks, names, groups = get_sorted_rank_groups(result, False)
                if len(groups) == 0:
                    print("Based on post-hoc Tukey HSD test, we assume that all differences between the populations "
                          "are significant.")
                else:
                    groupstrs = []
                    for group_range in groups:
                        group = range(group_range[0], group_range[1] + 1)
                        if len(group) == 1:
                            cur_groupstr = names[group[0]]
                        elif len(group) == 2:
                            cur_groupstr = " and ".join([names[pop] for pop in group])
                        else:
                            cur_groupstr = ", ".join([names[pop] for pop in group[:-1]]) + ", and " + names[group[-1]]
                        groupstrs.append(cur_groupstr)
                    print("Based post-hoc Tukey HSD test, we assume that there are no significant differences within "
                          "the following groups: %s. All other differences are significant." % ("; ".join(groupstrs)))
                print()
        elif result.omnibus == 'friedman':
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (alpha=%f) of the Friedman test that there is no "
                      "difference in the central tendency of the populations %s. Therefore, we "
                      "assume that there is no statistically significant difference between the median values of the "
                      "populations." % (result.alpha, population_string))
            else:
                print("We reject the null hypothesis (alpha=%f) of the Friedman test that there is no "
                      "difference in the central tendency of the populations %s. Therefore, we "
                      "assume that there is a statistically significant difference between the median values of the "
                      "populations." % (result.alpha, population_string))
                meanranks, names, groups = get_sorted_rank_groups(result, False)
                if len(groups) == 0:
                    print("Based on post-hoc Nemenyi test, we assume that all differences between the populations "
                          "are significant.")
                else:
                    groupstrs = []
                    for group_range in groups:
                        group = range(group_range[0], group_range[1] + 1)
                        if len(group) == 1:
                            cur_groupstr = names[group[0]]
                        elif len(group) == 2:
                            cur_groupstr = " and ".join([names[pop] for pop in group])
                        else:
                            cur_groupstr = ", ".join([names[pop] for pop in group[:-1]]) + ", and " + names[group[-1]]
                        groupstrs.append(cur_groupstr)
                    print("Based the post-hoc Nemenyi test, we assume that there are no significant differences within "
                          "the following groups: %s. All other differences are significant." % ("; ".join(groupstrs)))
        else:
            raise ValueError('Unknown omnibus test for difference in the central tendency: %s' % result.omnibus)
