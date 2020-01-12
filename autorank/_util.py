import math
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


def rank_two(data, alpha, verbose, all_normal):
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


def rank_multiple_normal_homoscedastic(data, alpha=0.05, verbose=False):
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


def rank_multiple_nonparametric(data, alpha, verbose, all_normal):
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


def get_sorted_rank_groups(result, reverse):
    if reverse:
        names = result.rankdf.iloc[::-1].index.to_list()
        if result.cd is not None:
            sorted_ranks = result.rankdf.iloc[::-1].meanrank
            critical_difference = result.cd
        else:
            sorted_ranks = result.rankdf.iloc[::-1]['mean']
            critical_difference = (result.rankdf.ci_upper[0] - result.rankdf.ci_lower[0]) / 2
    else:
        names = result.rankdf.index.to_list()
        if result.cd is not None:
            sorted_ranks = result.rankdf.meanrank
            critical_difference = result.cd
        else:
            sorted_ranks = result.rankdf['mean']
            critical_difference = (result.rankdf.ci_upper[0] - result.rankdf.ci_lower[0]) / 2

    groups = []
    cur_max_j = -1
    for i in range(len(sorted_ranks)):
        max_j = None
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks[i] - sorted_ranks[j]) <= critical_difference:
                max_j = j
                # print(i, j)
        if max_j is not None and max_j > cur_max_j:
            cur_max_j = max_j
            groups.append((i, max_j))
    return sorted_ranks, names, groups


def cd_diagram(result, reverse, ax, width):
    """
    Creates a Critical Distance diagram.
    """

    def plot_line(line, color='k', **kwargs):
        ax.plot([pos[0] / width for pos in line], [pos[1] / height for pos in line], color=color, **kwargs)

    def plot_text(x, y, s, *args, **kwargs):
        ax.text(x / width, y / height, s, *args, **kwargs)

    sorted_ranks, names, groups = get_sorted_rank_groups(result, reverse)
    cd = result.cd

    lowv = min(1, int(math.floor(min(sorted_ranks))))
    highv = max(len(sorted_ranks), int(math.ceil(max(sorted_ranks))))
    cline = 0.4
    textspace = 1
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            relative_rank = rank - lowv
        else:
            relative_rank = highv - rank
        return textspace + scalewidth / (highv - lowv) * relative_rank

    linesblank = 0.2 + 0.2 + (len(groups) - 1) * 0.1

    # add scale
    distanceh = 0.25
    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((len(sorted_ranks) + 1) / 2) * 0.2 + minnotsignificant

    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    plot_line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        plot_line([(rankpos(a), cline - tick / 2),
                   (rankpos(a), cline)],
                  linewidth=0.7)

    for a in range(lowv, highv + 1):
        plot_text(rankpos(a), cline - tick / 2 - 0.05, str(a),
                  ha="center", va="bottom")

    for i in range(math.ceil(len(sorted_ranks) / 2)):
        chei = cline + minnotsignificant + i * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline),
                   (rankpos(sorted_ranks[i]), chei),
                   (textspace - 0.1, chei)],
                  linewidth=0.7)
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line([(rankpos(sorted_ranks[i]), cline),
                   (rankpos(sorted_ranks[i]), chei),
                   (textspace + scalewidth + 0.1, chei)],
                  linewidth=0.7)
        plot_text(textspace + scalewidth + 0.2, chei, names[i],
                  ha="left", va="center")

    # upper scale
    if not reverse:
        begin, end = rankpos(lowv), rankpos(lowv + cd)
    else:
        begin, end = rankpos(highv), rankpos(highv - cd)

    plot_line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
    plot_line([(begin, distanceh + bigtick / 2),
               (begin, distanceh - bigtick / 2)],
              linewidth=0.7)
    plot_line([(end, distanceh + bigtick / 2),
               (end, distanceh - bigtick / 2)],
              linewidth=0.7)
    plot_text((begin + end) / 2, distanceh - 0.05, "CD",
              ha="center", va="bottom")

    # no-significance lines
    side = 0.05
    no_sig_height = 0.1
    start = cline + 0.2
    for l, r in groups:
        plot_line([(rankpos(sorted_ranks[l]) - side, start),
                   (rankpos(sorted_ranks[r]) + side, start)],
                  linewidth=2.5)
        start += no_sig_height

    return ax


def ci_plot(result, reverse, ax, width):
    """
    Uses error bars to create a plot of the confidence intervals of the mean value.
    """
    if reverse:
        sorted_df = result.rankdf.iloc[::-1]
    else:
        sorted_df = result.rankdf
    sorted_means = sorted_df['mean']
    ci_lower = sorted_df.ci_lower
    ci_upper = sorted_df.ci_upper
    names = sorted_df.index

    height = len(sorted_df)
    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor('white')
        ax = plt.gca()
    ax.errorbar(sorted_means, range(len(sorted_means)), xerr=(ci_upper[0] - ci_lower[0]) / 2, marker='o',
                linestyle='None', color='k', ecolor='k')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names.to_list())
    ax.set_title('%.1f%% Confidence Intervals of the Mean' % ((1 - result.alpha) * 100))
    return ax
