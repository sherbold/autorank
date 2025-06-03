import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.anova import AnovaRM
from baycomp import SignedRankTest
from collections import namedtuple

__all__ = ['rank_two', 'rank_multiple_normal_homoscedastic', 'rank_bayesian', 'RankResult',
           'rank_multiple_nonparametric', 'cd_diagram', 'get_sorted_rank_groups', 'ci_plot', 'test_normality', 'posterior_maps']


class RankResult(namedtuple('RankResult', ('rankdf', 'pvalue', 'cd', 'omnibus', 'posthoc', 'all_normal',
                                           'pvals_shapiro', 'homoscedastic', 'pval_homogeneity', 'homogeneity_test',
                                           'alpha', 'alpha_normality', 'num_samples', 'sample_matrix',
                                           'posterior_matrix', 'decision_matrix', 'rope', 'rope_mode', 'effect_size',
                                           'force_mode', 'plot_order'))):
    __slots__ = ()

    def __str__(self):
        return 'RankResult(rankdf=\n%s\n' \
               'pvalue=%s\n' \
               'cd=%s\n' \
               'omnibus=%s\n' \
               'posthoc=%s\n' \
               'all_normal=%s\n' \
               'pvals_shapiro=%s\n' \
               'homoscedastic=%s\n' \
               'pval_homogeneity=%s\n' \
               'homogeneity_test=%s\n' \
               'alpha=%s\n' \
               'alpha_normality=%s\n' \
               'num_samples=%s\n' \
               'posterior_matrix=\n%s\n' \
               'decision_matrix=\n%s\n' \
               'rope=%s\n' \
               'rope_mode=%s\n' \
               'effect_size=%s\n'\
               'force_mode=%s)' % (self.rankdf, self.pvalue, self.cd, self.omnibus, self.posthoc, self.all_normal,
                                self.pvals_shapiro, self.homoscedastic, self.pval_homogeneity,
                                self.homogeneity_test, self.alpha, self.alpha_normality, self.num_samples,
                                self.posterior_matrix, self.decision_matrix, self.rope, self.rope_mode,
                                self.effect_size, self.force_mode)


class _ComparisonResult(namedtuple('ComparisonResult', ('rankdf', 'pvalue', 'cd', 'omnibus', 'posthoc',
                                                        'effect_size', 'reorder_pos'))):
    __slots__ = ()

    def __str__(self):
        return '_ComparisonResult(rankdf=\n%s\n' \
               'pvalue=%s\n' \
               'cd=%s\n' \
               'omnibus=%s\n' \
               'posthoc=%s\n' \
               'effect_size=%s\n' \
               'reorder_pos=%s)' % (self.rankdf, self.pvalue, self.cd, self.omnibus, self.posthoc, self.effect_size,
                                    self.reorder_pos)


class _BayesResult(namedtuple('BayesResult', ('rankdf', 'sample_matrix', 'posterior_matrix', 'decision_matrix', 'effect_size',
                                              'reorder_pos'))):
    __slots__ = ()

    def __str__(self):
        return 'BayesResult(rankdf=\n%s\n' \
               'posterior_matrix=%s\n' \
               'decision_matrix=%s\n' \
               'effect_size=%s\n' \
               'reorder_pos=%s)' % (self.rankdf, self.posterior_matrix, self.decision_matrix, self.effect_size,
                                    self.reorder_pos)


def _pooled_std(x, y):
    """
    Calculate the pooled standard deviation of x and y
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)


def _pooled_mad(x, y):
    """
    Calculate the pooled median absolute deviation of x and y
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    mad_x = stats.median_abs_deviation(x, scale=1/1.4826)  # scale MAD to be similar to SD of a normal
    mad_y = stats.median_abs_deviation(y, scale=1/1.4826)  # scale MAD to be similar to SD of a normal
    return np.sqrt(((nx - 1) * mad_x ** 2 + (ny - 1) * mad_y ** 2) / dof)


def _cohen_d(x, y):
    """
    Calculate the effect size using Cohen's d
    """
    return (np.mean(x) - np.mean(y)) / _pooled_std(x, y)


def _akinshin_gamma(x, y):
    """
    Calculate the effect size using a non-parametric variant of Cohen's d that replaces the pooled
    standard deviation with the pooled median absolute devision. This metric is based on this blog
    post (no publication yet).
    https://aakinshin.net/posts/nonparametric-effect-size/
    """
    return (np.median(x) - np.median(y)) / _pooled_mad(x, y)


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


def _effect_level(effect_size, method='cohen_d'):
    """
    Determines magnitude of effect size.
    """
    if not isinstance(method, str):
        raise TypeError('method must be of type str')
    if method not in ['cohen_d', 'cliff_delta', 'akinshin_gamma']:
        raise ValueError("method must be one of the following strings: 'cohen_d', 'cliff_delta', 'akinshin_gamma'")
    effect_size = abs(effect_size)
    if method == 'cliff_delta':
        if effect_size < 0.147:
            return 'negligible'
        elif effect_size < 0.33:
            return 'small'
        elif effect_size < 0.474:
            return 'medium'
        else:
            return 'large'
    if method == 'cohen_d' or method == 'akinshin_gamma':
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

        r = max(0, r)
        s = min(len(data)-1, s)
        sorted_data = data.sort_values()
        lower = sorted_data.iloc[int(round(r))]
        upper = sorted_data.iloc[int(round(s))]
        return lower, upper


def _posterior_decision(probabilities, alpha):
    """
    calculate decision based on probabilities and desired significance
    """
    if len(probabilities) == 3:
        # with ROPE
        if probabilities[0] >= 1 - alpha:
            return 'smaller'
        elif probabilities[1] >= 1 - alpha:
            return 'equal'
        elif probabilities[2] >= 1 - alpha:
            return 'larger'
        else:
            return 'inconclusive'
    else:
        # without ROPE (i.e., rope=0)
        if probabilities[0] >= 1 - alpha:
            return 'smaller'
        elif probabilities[1] >= 1 - alpha:
            return 'larger'
        else:
            return 'inconclusive'


def rank_two(data, alpha, verbose, all_normal, order, effect_size, force_mode):
    """
    Uses paired t-test for normal data and Wilcoxon's signed rank test for other distributions. Can be overridden with
    force_mode.
    """
    larger = np.argmax(data.median().values)
    smaller = int(bool(larger - 1))
    if (force_mode is not None and force_mode == 'parametric') or (force_mode is None and all_normal):
        if verbose:
            print("Using paired t-test")
        omnibus = 'ttest'
        pval = stats.ttest_rel(data.iloc[:, larger], data.iloc[:, smaller]).pvalue
    else:
        if verbose:
            print("Using Wilcoxon's signed rank test (one-sided)")
        omnibus = 'wilcoxon'
        pval = stats.wilcoxon(data.iloc[:, larger], data.iloc[:, smaller], alternative='greater').pvalue
    if verbose:
        if pval >= alpha:
            print(
                "Fail to reject null hypothesis that there is no difference between the distributions (p=%f)" % pval)
        else:
            print("Rejecting null hypothesis that there is no difference between the distributions (p=%f)" % pval)
    rankdf, effsize_method, reorder_pos = _create_result_df_skeleton(data, alpha, all_normal, order,
                                                                     effect_size=effect_size, force_mode=force_mode)
    return _ComparisonResult(rankdf, pval, None, omnibus, None, effsize_method, reorder_pos)


def rank_multiple_normal_homoscedastic(data, alpha, verbose, order, effect_size, force_mode):
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

    rankdf, effsize_method, reorder_pos = _create_result_df_skeleton(data, None, True, order,
                                                                     effect_size=effect_size, force_mode=force_mode)
    for population in rankdf.index:
        mean = data.loc[:, population].mean()
        ci_range = tukey_res.halfwidths[data.columns.get_loc(population)]
        lower, upper = mean - ci_range, mean + ci_range
        rankdf.at[population, 'ci_lower'] = lower
        rankdf.at[population, 'ci_upper'] = upper
    return _ComparisonResult(rankdf, pval, None, 'anova', 'tukeyhsd', effsize_method, reorder_pos)


def rank_multiple_nonparametric(data, alpha, verbose, all_normal, order, effect_size, force_mode):
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
    rankdf, effsize_method, reorder_pos = _create_result_df_skeleton(data, alpha, all_normal, order,
                                                                     effect_size=effect_size, force_mode=force_mode)
    return _ComparisonResult(rankdf, pval, cd, 'friedman', 'nemenyi', effsize_method, reorder_pos)


def rank_bayesian(data, alpha, verbose, all_normal, order, rope, rope_mode, nsamples, effect_size, random_state, force_mode):
    # TODO check if some outputs for the verbose mode would be helpful
    if (force_mode is not None and force_mode == 'parametric') or (force_mode is None and all_normal):
        order_column = 'mean'
    else: # either with force_mode or not all_normal
        order_column = 'median'
    result_df, effsize_method, reorder_pos = _create_result_df_skeleton(data, alpha/len(data.columns), all_normal,
                                                                        order, order_column, effect_size, force_mode)
    result_df = result_df.drop('meanrank', axis='columns')
    result_df['p_equal'] = np.nan
    result_df['p_smaller'] = np.nan
    result_df['decision'] = 'NA'
    result_df['p_equal_above'] = np.nan
    result_df['p_smaller_above'] = np.nan


    # re-order columns to have the same order as results
    reordered_data = data.reindex(result_df.index, axis=1)

    sample_matrix = pd.DataFrame(index=reordered_data.columns, columns=reordered_data.columns)
    posterior_matrix = pd.DataFrame(index=reordered_data.columns, columns=reordered_data.columns)
    decision_matrix = pd.DataFrame(index=reordered_data.columns, columns=reordered_data.columns)
    for i in range(len(data.columns)):
        for j in range(i+1, len(reordered_data.columns)):
            if rope_mode == 'effsize':
                # half the size of a small effect size following Kruschke (2018)
                if (force_mode is not None and force_mode == 'parametric') or (force_mode is None and all_normal):
                    # use Cohen's d
                    cur_rope = rope*_pooled_std(reordered_data.iloc[:, i], reordered_data.iloc[:, j])
                else: # either with force_mode or not all_normal
                    # use Akinshin's gamma
                    cur_rope = rope*_pooled_mad(reordered_data.iloc[:, i], reordered_data.iloc[:, j])
            elif rope_mode == 'absolute':
                cur_rope = rope
            else:
                raise ValueError("Unknown rope_mode method, this should not be possible.")
            sample = SignedRankTest(x=reordered_data.iloc[:, i], y=reordered_data.iloc[:, j], rope=cur_rope,
                                    nsamples=nsamples, random_state=random_state)
            posterior_probabilities = sample.probs()
            sample_matrix.iloc[i, j] = sample
            posterior_matrix.iloc[i, j] = posterior_probabilities
            decision_matrix.iloc[i, j] = _posterior_decision(posterior_probabilities, alpha)
            decision_matrix.iloc[j, i] = _posterior_decision(posterior_probabilities[::-1], alpha)
            if i == 0:
                # comparison with "best"
                result_df.loc[result_df.index[j], 'p_equal'] = posterior_probabilities[1]
                result_df.loc[result_df.index[j], 'p_smaller'] = posterior_probabilities[0]
                result_df.loc[result_df.index[j], 'decision'] = _posterior_decision(posterior_probabilities, alpha)
    for i in range(1, len(data.columns)):
        result_df.loc[result_df.index[i], 'p_equal_above'] = posterior_matrix.iloc[i-1, i][1]
        result_df.loc[result_df.index[i], 'p_smaller_above'] = posterior_matrix.iloc[i-1, i][0]
        result_df.loc[result_df.index[i], 'decision_above'] = _posterior_decision(posterior_matrix.iloc[i-1, i], alpha)

    return _BayesResult(result_df, sample_matrix, posterior_matrix, decision_matrix, effsize_method, reorder_pos)


def _create_result_df_skeleton(data, alpha, all_normal, order, order_column='meanrank', effect_size=None,
                               force_mode=None):
    """
    Creates data frame for results. CI may be left empty in case alpha is None
    """
    if effect_size is None:
        if all_normal:
            effsize_method = 'cohen_d'
        else:
            effsize_method = 'akinshin_gamma'
    else:
        effsize_method = effect_size

    asc = None
    if order == 'descending':
        asc = False
    elif order == 'ascending':
        asc = True

    rankmat = data.rank(axis='columns', ascending=asc)
    meanranks = rankmat.mean()
    if (force_mode is not None and force_mode=='parametric') or (force_mode is None and all_normal):
        rankdf = pd.DataFrame(index=meanranks.index,
                              columns=['meanrank', 'mean', 'std', 'ci_lower', 'ci_upper', 'effect_size', 'magnitude', 'effect_size_above', 'magnitude_above'])
        rankdf['mean'] = data.mean().reindex(meanranks.index)
        rankdf['std'] = data.std().reindex(meanranks.index)
    else:
        rankdf = pd.DataFrame(index=meanranks.index,
                              columns=['meanrank', 'median', 'mad', 'ci_lower', 'ci_upper', 'effect_size', 'magnitude', 'effect_size_above', 'magnitude_above'])
        rankdf['median'] = data.median().reindex(meanranks.index)
        for population in rankdf.index:
            rankdf.at[population, 'mad'] = stats.median_abs_deviation(data.loc[:, population])
    rankdf['meanrank'] = meanranks

    # need to know reordering here (see issue #7)
    reorder_index = rankdf[order_column].sort_values(ascending=asc).index
    reorder_pos = [reorder_index.get_loc(old_index) for old_index in rankdf.index]
    rankdf = rankdf.reindex(reorder_index)

    population_above = None
    for population in rankdf.index:
        if effsize_method == 'cohen_d':
            effsize = _cohen_d(data.loc[:, rankdf.index[0]], data.loc[:, population])
            if population_above is not None:
                effsize_above = _cohen_d(data.loc[:, population_above], data.loc[:, population])
        elif effsize_method == 'cliff_delta':
            effsize = _cliffs_delta(data.loc[:, rankdf.index[0]], data.loc[:, population])
            if population_above is not None:
                effsize_above = _cliffs_delta(data.loc[:, population_above], data.loc[:, population])
        elif effsize_method == 'akinshin_gamma':
            effsize = _akinshin_gamma(data.loc[:, rankdf.index[0]], data.loc[:, population])
            if population_above is not None:
                effsize_above = _akinshin_gamma(data.loc[:, population_above], data.loc[:, population])
        else:
            raise ValueError("Unknown effsize method, this should not be possible.")
        if population_above is None:
            effsize_above = 0.0
        rankdf.at[population, 'effect_size'] = effsize
        rankdf.at[population, 'magnitude'] = _effect_level(effsize, effsize_method)
        rankdf.at[population, 'effect_size_above'] = effsize_above
        rankdf.at[population, 'magnitude_above'] = _effect_level(effsize_above, effsize_method)

        if alpha is not None:
            lower, upper = _confidence_interval(data.loc[:, population], alpha / len(data.columns),
                                                is_normal=all_normal)
            rankdf.at[population, 'ci_lower'] = lower
            rankdf.at[population, 'ci_upper'] = upper
        population_above = population

    print(rankdf)

    return rankdf, effsize_method, reorder_pos


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
            critical_difference = (result.rankdf.ci_upper.iloc[0] - result.rankdf.ci_lower.iloc[0]) / 2

    groups = []
    cur_max_j = -1
    for i in range(len(sorted_ranks)):
        max_j = None
        for j in range(i + 1, len(sorted_ranks)):
            if abs(sorted_ranks.iloc[i] - sorted_ranks.iloc[j]) <= critical_difference:
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

    result_copy = RankResult(**result._asdict())
    result_copy = result_copy._replace(rankdf=result.rankdf.sort_values(by='meanrank'))
    sorted_ranks, names, groups = get_sorted_rank_groups(result_copy, reverse)
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
        plot_line([(rankpos(sorted_ranks.iloc[i]), cline),
                   (rankpos(sorted_ranks.iloc[i]), chei),
                   (textspace - 0.1, chei)],
                  linewidth=0.7)
        plot_text(textspace - 0.2, chei, names[i], ha="right", va="center")

    for i in range(math.ceil(len(sorted_ranks) / 2), len(sorted_ranks)):
        chei = cline + minnotsignificant + (len(sorted_ranks) - i - 1) * 0.2
        plot_line([(rankpos(sorted_ranks.iloc[i]), cline),
                   (rankpos(sorted_ranks.iloc[i]), chei),
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
        plot_line([(rankpos(sorted_ranks.iloc[l]) - side, start),
                   (rankpos(sorted_ranks.iloc[r]) + side, start)],
                  linewidth=2.5)
        start += no_sig_height

    return ax


def ci_plot(result, reverse, ax, width):
    """
    Uses error bars to create a plot of the confidence intervals of the mean value.
    """
    if result.plot_order is not None:
        # if the result has a plot order, use it
        ordered_df = result.rankdf.loc[result.plot_order]
    else:
        # otherwise, use the default order
        ordered_df = result.rankdf

    # we usually revert, because the plot has the first list item at the bottom
    if reverse:
        sorted_df = ordered_df.iloc[::-1]
    else:
        sorted_df = ordered_df
    sorted_means = sorted_df['mean']
    ci_lower = sorted_df.ci_lower
    ci_upper = sorted_df.ci_upper
    names = sorted_df.index

    height = len(sorted_df)
    if ax is None:
        fig = plt.figure(figsize=(width, height))
        fig.set_facecolor('white')
        ax = plt.gca()
    ax.errorbar(sorted_means, range(len(sorted_means)), xerr=(ci_upper.iloc[0] - ci_lower.iloc[0]) / 2, marker='o',
                linestyle='None', color='k', ecolor='k')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names.to_list())
    ax.set_title('%.1f%% Confidence Intervals of the Mean' % ((1 - result.alpha) * 100))
    return ax


def test_normality(data, alpha, verbose):
    """
    Tests if all populations are normal and return whether this is true and a list of p-values
    """
    all_normal = True
    pvals_shapiro = []
    for column in data.columns:
        w, pval_shapiro = stats.shapiro(data[column])
        pvals_shapiro.append(pval_shapiro)
        if pval_shapiro < alpha:
            all_normal = False
            if verbose:
                print("Rejecting null hypothesis that data is normal for column %s (p=%f<%f)" % (
                    column, pval_shapiro, alpha))
        elif verbose:
            print("Fail to reject null hypothesis that data is normal for column %s (p=%f>=%f)" % (
                column, pval_shapiro, alpha))
    return all_normal, pvals_shapiro


def _heatmap(data, row_labels, col_labels, ax,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Adopted from the scikit-learn documentation:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """
    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=-30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-')#, linewidth=4)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def _annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     vrange=None, **textkw):
    """
    A function to annotate a heatmap.

    Adopted from the scikit-learn documentation:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if vrange is not None:
        threshold = (vrange[1]-vrange[0])/2
    else:
        threshold = im.norm(data.max())/2

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    if textcolors is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                im.axes.text(j, i, valfmt(data[i, j], None), **kw)


def _create_annotated_heatmap(ax, data, title_prefix, cmap, annot_color, cbarlabel, cbar_kw=None, vmin=0, vmax=1):
    """
    Creates an annotated heatmap on the given axes.
    """
    im = _heatmap(data.values, data.index, data.columns,
                        ax=ax, cbarlabel=cbarlabel,
                        cmap=cmap, vmin=vmin, vmax=vmax, cbar_kw=cbar_kw)
    _annotate_heatmap(im, valfmt="{x:.2f}", vrange=(vmin, vmax), textcolors=annot_color)
    ax.set_title(title_prefix + cbarlabel)
    ax.set_xlabel("B")
    ax.set_ylabel("A", rotation=0)

def posterior_maps(result, *, width, cmaps, annot_colors, axes=None):
    """
    Creates a figure with four subplots showing the posterior probabilities of the comparisons.
    """
    posterior_matrix_smaller_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)
    posterior_matrix_equal_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)
    posterior_matrix_larger_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)
    posterior_matrix_decision_df = pd.DataFrame(np.nan, index=result.posterior_matrix.index, columns=result.posterior_matrix.columns)

    for i in range(result.posterior_matrix.shape[0]):
        for j in range(result.posterior_matrix.shape[1]):
            if isinstance(result.posterior_matrix.iloc[i, j], tuple):
                p_smaller, p_equal, p_larger = result.posterior_matrix.iloc[i, j]
                posterior_matrix_smaller_df.iloc[i, j] = p_smaller
                posterior_matrix_equal_df.iloc[i, j] = p_equal
                posterior_matrix_larger_df.iloc[i, j] = p_larger
                # this should be dropped, there is already a decision matrix
                decision = 1
                if result.decision_matrix.iloc[i, j]=='smaller':
                    decision = 2
                if result.decision_matrix.iloc[i, j]=='equal':
                    decision = 3
                if result.decision_matrix.iloc[i, j]=='larger':
                    decision = 4
                posterior_matrix_decision_df.iloc[i, j] = decision
    
    # drop first column and last row
    posterior_matrix_smaller_df = posterior_matrix_smaller_df.drop(columns=posterior_matrix_smaller_df.columns[0])
    posterior_matrix_smaller_df = posterior_matrix_smaller_df.drop(index=posterior_matrix_smaller_df.index[-1])
    posterior_matrix_equal_df = posterior_matrix_equal_df.drop(columns=posterior_matrix_equal_df.columns[0])
    posterior_matrix_equal_df = posterior_matrix_equal_df.drop(index=posterior_matrix_equal_df.index[-1])
    posterior_matrix_larger_df = posterior_matrix_larger_df.drop(columns=posterior_matrix_larger_df.columns[0])
    posterior_matrix_larger_df = posterior_matrix_larger_df.drop(index=posterior_matrix_larger_df.index[-1])
    posterior_matrix_decision_df = posterior_matrix_decision_df.drop(columns=posterior_matrix_decision_df.columns[0])
    posterior_matrix_decision_df = posterior_matrix_decision_df.drop(index=posterior_matrix_decision_df.index[-1])
    
    if axes is None:
        fig = plt.figure(figsize=(width, width))
        gs = mpl.gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        plt_created = True
    else:
        ax1 = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
        ax4 = axes[3]

    if not isinstance(cmaps[3], mpl.colors.ListedColormap) or cmaps[3].N != 4:
        cmaps[3] = mpl.colormaps[cmaps[3]].resampled(4)
    decisions = ['', 'Inconclusive', '$A < B$', '$A = B$', '$A > B$', '']
    dec_fmt = mpl.ticker.FuncFormatter(lambda x, _: decisions[int(x)])

    _create_annotated_heatmap(ax1, posterior_matrix_smaller_df,
                              title_prefix="(a) ",
                              cmap=cmaps[0], cbarlabel="$P(A < B)$",
                              annot_color=annot_colors[0])
    _create_annotated_heatmap(ax2, posterior_matrix_equal_df,
                              title_prefix="(b) ",
                              cmap=cmaps[1], cbarlabel="$P(A = B)$",
                              annot_color=annot_colors[1])
    _create_annotated_heatmap(ax3, posterior_matrix_larger_df,
                              title_prefix="(c) ",
                              cmap=cmaps[2], cbarlabel="$P(A > B)$",
                              annot_color=annot_colors[2])
    _create_annotated_heatmap(ax4, posterior_matrix_decision_df,
                              title_prefix="(d) ",
                              cmap=cmaps[3], cbarlabel="Decision",
                              annot_color=None,
                              cbar_kw=dict(ticks=[0, 1, 2, 3, 4, 5], format=dec_fmt),
                              vmin=0.5, vmax=4.5)
    if plt_created:
        plt.tight_layout()