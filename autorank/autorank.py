"""
Automated ranking of populations for ranking them. This is basically an implementation of Demsar's
Guidelines for the comparison of multiple classifiers. Details can be found in the description of the autorank function.
"""

import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import stats
from io import StringIO
from autorank._util import *

__all__ = ['autorank', 'plot_stats', 'plot_posterior_maps', 'create_report', 'latex_table', 'latex_report']

if 'text.usetex' in plt.rcParams and plt.rcParams['text.usetex']:
    raise UserWarning("plot_stats may fail if the matplotlib setting plt.rcParams['text.usetex']==True.\n"
                      "In case of failures you can try to set this value to False as follows:"
                      "plt.rc('text', usetex=False)")


def autorank(data, alpha=0.05, verbose=False, order='descending', approach='frequentist', rope=0.1, rope_mode='effsize',
             nsamples=50000, effect_size=None, force_mode=None, random_state=None, plot_order=None):
    """
    Automatically compares populations defined in a block-design data frame. Each column in the data frame contains
    the samples for one population. The data must not contain any NaNs. The data must have at least five measurements,
    i.e., rows. The current version is only reliable for less than 5000 measurements.

    The following approach is implemented by this function.

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

    For the statistical tests, there are five variants:

    - If approach=='bayesian' we use a Bayesian signed rank test.
    - If there are two populations (columns) and both populations are normal, we use the paired t-test.
    - If there are two populations and at least one populations is not normal, we use Wilcoxon's signed rank test.
    - If there are more than two populations and all populations are normal and homoscedastic, we use repeated measures
      ANOVA with Tukey's HSD as post-hoc test.
    - If there are more than two populations and at least one populations is not normal or the populations are
      heteroscedastic, we use Friedman's test with the Nemenyi post-hoc test.

    # Parameters

    data (DataFrame):
        Each column contains a population and each row contains the paired measurements
        for the populations.

    alpha (float, default=0.05):
        Significance level. We internally use correction to ensure that all results (incl. confidence
        intervals) together fulfill this confidence level.

    verbose (bool, default=False):
        Prints decisions and p-values while running the autorank function to stdout.

    order (string, default='descending'):
        Determines the ordering central tendencies of the populations for the ranking. 'descending' results in higher
        ranks for larger values. 'ascending' results in higher ranks for smaller values.

    approach (string, default='frequentist'):
        With 'frequentist', a suitable frequentist statistical test is used (t-test, Wilcoxon signed rank test,
        ANOVA+Tukey's HSD, or Friedman+Nemenyi). With 'bayesian', the Bayesian signed ranked test is used.
        _(New in Version 1.1.0)_

    rope (float, default=0.01):
        Region of Practical Equivalence (ROPE) used for the bayesian analysis. The statistical analysis assumes that
        differences from the central tendency that are within the ROPE do not matter in practice. Therefore, such
        deviations may be considered to be equivalent. The ROPE is defined as an interval around the central tendency
        and the calculation of the interval is determined by the rope_mode parameter.
        _(New in Version 1.1.0)_

    rope_mode (string, default='effsize'):
        Method to calculate the size of the ROPE. With 'effsize', the ROPE is determined dynamically for each comparison
        of two populations as rope*effect_size, where effect size is either Cohen's d (normal data) or Akinshin's gamma
        (non-normal data). With 'absolute', the ROPE is defined using an absolute value that is used, i.e., the value of
        the rope parameter is used without any modification.
        _(New in Version 1.1.0)_

    nsamples (integer, default=50000):
        Number of samples used to estimate the posterior probabilities with the Bayesian signed rank test.
        _(New in Version 1.1.0)_

    effect_size (string, default=None):
        Effect size measure that is used for reporting. If None, the effect size is automatically selected as described
        in the flow chart. The following effect sizes are supported: "cohen_d", "cliff_delta", "akinshin_gamma".
        _(New in Version 1.1.0)_

    force_mode (string, default=None):
        Can be used to force autorank to use parametric or nonparametric frequentist tests. With 'parametric' you
        automatically get the t-test/repeated measures ANOVA. With 'nonparametric' you automatically get Wilcoxon's
        signed rank test/Friedman test. In case of Bayesian statistics, this parameter is used to override the automatic
        selection of the effect size measure, such that 'parametric' uses Cohen's d and 'nonparametric' uses Akinshin's,
        regardless of the normality of the data. If this parameter is None, the automatic selection is used.
        _(Support for Bayesian statistics added in Version 1.3.0)_

    random_state (integer, default=None):
        Seed for random state. Forwarded to Bayesian signed rank test to enable reproducible sampling and, thereby,
        reproducible results.
        _(New in Version 1.2.0)_

    plot_order (list):
        List with the order of the populations used for ploting, where reasonable (e.g., CI plots). If this is not none, this overrides the order parameter for visualizations. 
        _(New in Version 1.3.0)_

    # Returns

    A named tuple of type RankResult with the following entries.

    rankdf (DataFrame):
        Ranked populations including statistics about the populations.

    pvalue (float):
        p-value of the omnibus test for the difference in central tendency between the populations. Not used with
        Bayesian statistics.

    omnibus (string):
       Omnibus test that is used for the test of a difference ein the central tendency.

    posthoc (string):
        Posthoc tests that was used. The posthoc test is performed even if the omnibus test is not significant. The
        results should only be used if the p-value of the omnibus test indicates significance. None in case of two
        populations and Bayesian statistics.

    cd (float):
        The critical distance of the Nemenyi posthoc test, if it was used. Otherwise None.

    all_normal (bool):
        True if all populations are normal, false if at least one is not normal.

    pvals_shapiro (list):
        p-values of the Shapiro-Wilk tests for normality sorted by the order of the input columns.

    homoscedastic (bool):
        True if populations are homoscedastic, false otherwise. None in case of Bayesian statistics.

    pval_homogeneity (float):
        p-value of the test for homogeneity. None in case of Bayesian statistics.

    homogeneity_test (string):
        Test used for homogeneity. Either 'bartlet' or 'levene'.

    alpha (float):
        Family-wise significant level. Same as input parameter.

    alpha_normality (float):
        Corrected alpha that is used for tests for normality.

    num_samples (int):
        Number of samples within each population.

    order (string):
        Order of the central tendencies used for ranking.

    sample_matrix (DataFrame):
        Matrix with SignedRankTest objects from package baycomp. Can be used to do further analysis, e.g. to generate
        plots using the built-in plot() method of baycomp. For a detailed description of methods and parameters, see
        the documentation of baycomp: https://baycomp.readthedocs.io/en/latest/classes.html#multiple-data-sets
        _(New in Version 1.2.0)_

    posterior_matrix (DataFrame):
        Matrix with the pair-wise posterior probabilities estimated with the Bayesian signed ranked test. The matrix
        is a square matrix with the populations sorted by their central tendencies as rows and columns. The value of
        the matrix in the i-th row and the j-th column contains a 3-tuple (p_smaller, p_equal, p_greater) such that
        p_smaller is the probability that the population in column j is smaller than the population in row i, p_equal
        that both populations are equal, and p_larger that population j is larger than population i. If rope==0.0, the
        matrix contains only 2-tuples (p_smaller, p_greater) because equality is not possible without a ROPE.
        _(New in Version 1.1.0)_

    decision_matrix (DataFrame):
        Matrix with the pair-wise decisions made with the Bayesian signed ranked test. The matrix is a square matrix
        with the populations sorted by their central tendencies as rows and columns. The value of
        the matrix in the i-th row and the j-th column contains the value 'smaller' if the population in column j is
        significantly larger than the population in row i, 'equal' is both populations are equivalent (i.e., have no
        practically relevant difference), 'larger' if the population in column j is larger than the population in
        column i, and 'inconclusive' if the statistical analysis is did not yield a definitive result.
        _(New in Version 1.1.0)_

    rope (float):
        Region of Practical Equivalence (ROPE). Same as input parameter.
        _(New in Version 1.1.0)_

    rope_mode (string):
        Mode for calculating the ROPE. Same as input parameter.
        _(New in Version 1.1.0)_

    effect_size (string):
        Effect size measure that is used for reporting. Same as input parameter.

    force_mode (string):
        If not None, this is the force mode that was used to select the tests. Either 'parametric' or 'nonparametric'.

    plot_order (list):
        If not None, this is the fixed order that is used for plotting, where possible. Otherwise None.
        _(New in Version 1.3.0)_
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

    if not isinstance(order, str):
        raise TypeError('order must be str')
    if order not in ['ascending', 'descending']:
        raise ValueError("order must be either 'ascending' or 'descending'")

    if not isinstance(approach, str):
        raise TypeError('approach must be str')
    if approach not in ['frequentist', 'bayesian']:
        raise ValueError("approach must be either 'frequentist' or 'bayesian'")

    if not isinstance(rope, (int, float)):
        raise TypeError('rope must be a numeric')
    if rope < 0.0:
        raise ValueError('rope must be positive')

    if not isinstance(rope_mode, str):
        raise TypeError('rope_mode must be str')
    if rope_mode not in ['effsize', 'absolute']:
        raise ValueError("rope_mode must be either 'effsize' or 'absolute'")

    if not isinstance(nsamples, int):
        raise TypeError('nsamples must be an integer')
    if nsamples < 1:
        raise ValueError('nsamples must be positive')

    if effect_size is not None:
        if not isinstance(effect_size, str):
            raise TypeError("effect_size must be a string")
        if effect_size not in ['cohen_d', 'cliff_delta', 'akinshin_gamma']:
            raise ValueError("effect_size must be None or one of the following: 'cohen_d', 'cliff_delta', "
                             "'akinshin_gamma'")

    if force_mode is not None:
        if not isinstance(force_mode, str):
            raise TypeError("force mode must be a string")
        if force_mode not in ['parametric', 'nonparametric']:
            raise ValueError("force_mode must be None or one of the following 'parametric', 'nonparametric'")

    if force_mode is not None and approach=='frequentist':
        print("Tests for normality and homoscedacity are ignored for test selection, forcing %s tests" % force_mode)

    if plot_order is not None:
        if not isinstance(plot_order, list):
            raise TypeError("plot_order must be a list")
        if len(plot_order) != len(data.columns):
            raise ValueError("plot_order must have the same length as the number of columns in data")
        if not all(isinstance(x, str) for x in plot_order):
            raise TypeError("plot_order must contain only strings")
        if not all(x in data.columns for x in plot_order):
            raise ValueError("plot_order must contain only columns from data")
        if len(set(plot_order)) != len(plot_order):
            raise ValueError("plot_order must not contain duplicates (not supported for data frames with duplicate column names)")

    # ensure that the index is not named or a MultiIndex
    # this trips up some internal functions (e.g., Anova (see issue #16))
    if data.index.name is not None or isinstance(data.index, pd.MultiIndex):
        data = data.reset_index(drop=True)

    # ensure that index and columns are not named
    # this also trips up some internal functions (e.g., Anova (see issue #37))
    if data.index.name is not None:
        data = data.rename_axis(None, axis=0)
    if data.columns.name is not None:
        data = data.rename_axis(None, axis=1)

    # Bonferoni correction for normality tests
    alpha_normality = alpha / len(data.columns)
    all_normal, pvals_shapiro = test_normality(data, alpha_normality, verbose)

    # Select appropriate tests
    if approach == 'frequentist':
        # homogeneity needs only to be checked for frequentist approach
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
                print("Fail to reject null hypothesis that all variances are equal "
                      "(p=%f>=%f)" % (pval_homogeneity, alpha))
            else:
                print("Rejecting null hypothesis that all variances are equal (p=%f<%f)" % (pval_homogeneity, alpha))

        if len(data.columns) == 2:
            res = rank_two(data, alpha, verbose, all_normal, order, effect_size, force_mode)
        else:
            if (force_mode is not None and force_mode=='parametric') or \
               (force_mode is None and all_normal and var_equal):
                res = rank_multiple_normal_homoscedastic(data, alpha, verbose, order, effect_size, force_mode)
            else:
                res = rank_multiple_nonparametric(data, alpha, verbose, all_normal, order, effect_size, force_mode)
        # need to reorder pvals here (see issue #7)
        pvals_shapiro = [pvals_shapiro[pos] for pos in res.reorder_pos]
        return RankResult(res.rankdf, res.pvalue, res.cd, res.omnibus, res.posthoc, all_normal, pvals_shapiro,
                          var_equal, pval_homogeneity, homogeneity_test, alpha, alpha_normality, len(data), None, None,
                          None, None, None, res.effect_size, force_mode, plot_order)
    elif approach == 'bayesian':
        res = rank_bayesian(data, alpha, verbose, all_normal, order, rope, rope_mode, nsamples, effect_size, random_state, force_mode)
        # need to reorder pvals here (see issue #7)
        pvals_shapiro = [pvals_shapiro[pos] for pos in res.reorder_pos]
        return RankResult(res.rankdf, None, None, 'bayes', 'bayes', all_normal, pvals_shapiro, None, None, None, alpha,
                          alpha_normality, len(data), res.sample_matrix, res.posterior_matrix, res.decision_matrix, rope,
                          rope_mode, res.effect_size, force_mode, plot_order)


def plot_stats(result, *, allow_insignificant=False, ax=None, width=None):
    """
    Creates a plot that supports the analysis of the results of the statistical test. The plot depends on the
    statistical test that was used.

    - Creates a Confidence Interval (CI) plot for a paired t-test between two normal populations. The confidence
     intervals are calculated with Bonferoni correction, i.e., a confidence level of alpha/2.
    - Creates a CI plot for Tukey's HSD as post-hoc test with the confidence intervals calculated using the HSD approach
     such that the family wise significance is alpha.
    - Creates Critical Distance (CD) diagrams for the Nemenyi post-hoc test. CD diagrams visualize the mean ranks of
     populations. Populations that are not significantly different are connected by a horizontal bar.

    This function raises a ValueError if the omnibus test did not detect a significant difference. The allow_significant
    parameter allows the suppression of this exception and forces the creation of the plots.

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    allow_insignificant (bool, default=False):
        Forces plotting even if results are not significant.

    ax (Axis, default=None):
        Matplotlib axis to which the results are added. A new figure with a single axis is created if None.

    width (float, default=None):
        Specifies the width of the created plot is not None. By default, we use a width of 6. The height is
        automatically determined, based on the type of plot and the number of populations. This parameter is ignored if
        ax is not None.

    # Return

    Axis with the plot. None if no plot was generated.
    """
    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")

    if result.omnibus == 'bayes':
        raise ValueError("ploting results of bayesian analysis not yet supported.")

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


def plot_posterior_maps(result, *, width=None, cmaps=None, annot_colors=None, axes=None, ):
    """
    Creates a posterior map plot for the results of the Bayesian signed rank test. The posterior map shows the
    posterior probabilities of the pair-wise comparisons between the populations.
    _(New in Version 1.3.0)_

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    axes (list, default=None):
        List of matplotlib axes to which the results are added. A new figure with a single axis is created if None.
        If there are more than one axes, they are used to create multiple subplots.

    width (float, default=None):
        Specifies the width of the created plot is not None. By default, we use a width of 10. The height is
        automatically set to the same value of width, since the maps should be square. This parameter is ignored if
        axes is not None.

    cmaps (list, default=['Blues', 'Oranges', 'Greys', custom_cmap]):
        Colormaps used for the posterior maps. The default custom_cmap is used for the decisions with four colors matching the
        colors of the posterior maps (+ one color for inconclusive).

    annot_colors (list, default=[("black", "white"), ("black", "white"), ("black", "white"), ("black", "white")]):
        Colors used for the annotations in the posterior maps. The first color is used for less intensive backgrounds,
        the second color for intensive backgrounds.
    """

    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")
    if width is None:
        width = 10
    if not isinstance(width, int) and not isinstance(width, float):
        raise TypeError("width must be a number or None")
    if width <= 0:
        raise ValueError("width must be positive")
    if cmaps is None:
        # Define the colors
        colors = ['whitesmoke', 'mediumseagreen', 'lightgrey', 'orangered']
        # Create the colormap
        cmap = mpl.colors.ListedColormap(colors, name='custom_colormap')
        cmaps = ['Greens', 'Oranges', 'Greys', cmap]
    if not isinstance(cmaps, list):
        raise TypeError("cmaps must be a list of colormaps or None")
    if len(cmaps) != 4:
        raise ValueError("cmaps must have exactly 4 elements")
    if annot_colors is None:
        annot_colors = [("black", "white"), ("black", "white"), ("black", "white"), ("black", "white")]
    if not isinstance(annot_colors, list):
        raise TypeError("annot_colors must be a list of colors or None")
    if len(annot_colors) != 4:
        raise ValueError("annot_colors must have exactly 4 elements")
    if axes is not None:
        if not isinstance(axes, list):
            raise TypeError("axes must be a list of matplotlib axes or None")
        if len(axes) != 4:
            raise ValueError("axes must have exactly 4 elements")

    if result.omnibus != 'bayes':
        raise ValueError("plot_posterior_maps can only be used with Bayesian analysis results.")
    
    posterior_maps(result, axes=axes, width=width, cmaps=cmaps, annot_colors=annot_colors)

    
def create_report(result, *, decimal_places=3):
    """
    Prints a report about the statistical analysis. 

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    decimal_places (int, default=3):
        Number of decimal places that are used for the report.
    """

    # TODO add effect sizes to multiple comparisons.
    def single_population_string(population, with_stats=False, pop_pval=None, with_rank=True):
        if pop_pval is not None:
            return "%s (p=%.*f)" % (population, decimal_places, pop_pval)
        if with_stats:
            halfwidth = (result.rankdf.at[population, 'ci_upper'] - result.rankdf.at[population, 'ci_lower']) / 2
            mystats = []
            if (result.force_mode is not None and result.force_mode=='parametric') or \
                    (result.force_mode is None and result.all_normal):
                mystats.append("M=%.*f+-%.*f" % (decimal_places, result.rankdf.at[population, 'mean'],
                                                 decimal_places, halfwidth))
                mystats.append("SD=%.*f" % (decimal_places, result.rankdf.at[population, 'std']))
            else:
                mystats.append("MD=%.*f+-%.*f" % (decimal_places, result.rankdf.at[population, 'median'],
                                                  decimal_places, halfwidth))
                mystats.append("MAD=%.*f" % (decimal_places, result.rankdf.at[population, 'mad']))
            if with_rank:
                mystats.append("MR=%.*f" % (decimal_places, result.rankdf.at[population, 'meanrank']))
            return "%s (%s)" % (population, ", ".join(mystats))
        else:
            return str(population)

    def create_population_string(populations, with_stats=False, pop_pvals=None, with_rank=False):
        if isinstance(populations, str):
            populations = [populations]
        population_strings = []
        for index, population in enumerate(populations):
            if pop_pvals is not None:
                cur_pval = pop_pvals[index]
            else:
                cur_pval = None
            population_strings.append(single_population_string(population, with_stats, cur_pval, with_rank))
        if len(populations) == 1:
            popstr = population_strings[0]
        elif len(populations) == 2:
            popstr = " and ".join(population_strings)
        else:
            popstr = ", ".join(population_strings[:-1]) + ", and " + population_strings[-1]
        return popstr

    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")

    print("The statistical analysis was conducted for %i populations with %i paired samples." % (len(result.rankdf),
                                                                                                 result.num_samples))
    print("The family-wise significance level of the tests is alpha=%.*f." % (decimal_places, result.alpha))

    if result.all_normal:
        not_normal = []
        min_pvalue = min(result.pvals_shapiro)
        print("We failed to reject the null hypothesis that the population is normal for all populations "
              "(minimal observed p-value=%.*f). Therefore, we assume that all populations are "
              "normal." % (decimal_places, min_pvalue))
    else:
        not_normal = []
        pvals = []
        normal = []
        for i, pval in enumerate(result.pvals_shapiro):
            if pval < result.alpha_normality:
                not_normal.append(result.rankdf.index[i])
                pvals.append(pval)
            else:
                normal.append(result.rankdf.index[i])
        if len(not_normal) == 1:
            population_term = 'population'
        else:
            population_term = 'populations'
        print("We rejected the null hypothesis that the population is normal for the %s %s. "
              "Therefore, we assume that not all populations are "
              "normal." % (population_term, create_population_string(not_normal, pop_pvals=pvals)))

    if result.omnibus == 'bayes':
        if (result.force_mode is not None and result.force_mode=='parametric') or (result.force_mode is None and result.all_normal):
            central_tendency = 'mean value'
            central_tendency_long = 'mean value (M)'
            variability = 'standard deviation (SD)'
            effect_size = 'd'
        else:
            central_tendency = 'median'
            central_tendency_long = 'median (MD)'
            variability = 'median absolute deviation (MAD)'
            effect_size = 'gamma'
        print(
            "We used a bayesian signed rank test to determine differences between the mean values of the "
            "populations and report the %s and the %s for each population. We distinguish "
            "between populations being pair-wise smaller, equal, or larger and make a decision for one "
            "of these cases if we estimate that the posterior probability is at least "
            "alpha=%.*f." % (central_tendency_long, variability, decimal_places, result.alpha))
        if result.rope_mode == 'effsize':
            print(
                'We used the effect size to define the region of practical equivalence (ROPE) around the %s '
                'dynamically as %.*f*%s.' % (central_tendency, decimal_places, result.rope, effect_size))
        else:
            print(
                'We used a fixed value of %.*f to define the region of practical equivalence (ROPE) around the '
                '%s.' % (decimal_places, result.rope, central_tendency))
        decision_set = set(result.rankdf['decision'])
        decision_set.remove('NA')
        if {'inconclusive'} == decision_set:
            print("We failed to find any conclusive evidence for differences between the populations "
                  "%s." % create_population_string(result.rankdf.index, with_stats=True))
        elif {'equal'} == decision_set:
            print(
                "All populations are equal, i.e., the are no significant and practically relevant differences "
                "between the populations %s." % create_population_string(result.rankdf.index,
                                                                         with_stats=True))
        elif {'equal', 'inconclusive'} == decision_set:
            print(
                "The populations %s are all either equal or the results of the analysis are inconclusive." % create_population_string(result.rankdf.index, with_stats=True))
            print(result.decision_matrix)
        else:
            print("We found significant and practically relevant differences between the populations "
                  "%s." % create_population_string(result.rankdf.index, with_stats=True))
            for i in range(len(result.rankdf)):
                if len(result.rankdf.index[result.decision_matrix.iloc[i, :] == 'smaller']) > 0:
                    print('The %s of the population %s is larger than of the populations '
                          '%s.' % (central_tendency, result.rankdf.index[i],
                                   create_population_string(
                                       result.rankdf.index[
                                           result.decision_matrix.iloc[i, :] == 'smaller'])))
            equal_pairs = []
            for i in range(len(result.rankdf)):
                for j in range(i + 1, len(result.rankdf)):
                    if result.decision_matrix.iloc[i, j] == 'equal':
                        equal_pairs.append(result.rankdf.index[i] + ' and ' + result.rankdf.index[j])
            if len(equal_pairs) > 0:
                equal_pairs_str = create_population_string(equal_pairs).replace(',', ';')
                print('The following pairs of populations are equal: %s.' % equal_pairs_str)
            if 'inconclusive' in set(result.rankdf['decision']):
                print('All other differences are inconclusive.')
    elif len(result.rankdf) == 2:
        print("No check for homogeneity was required because we only have two populations.")
        if result.effect_size == 'cohen_d':
            effect_size = 'd'
        elif result.effect_size == 'cliff_delta':
            effect_size = 'delta'
        elif result.effect_size == 'akinshin_gamma':
            effect_size = 'gamma'
        else:
            raise ValueError('unknown effect size method, this should not be possible: %s' % result.effect_size)
        if result.omnibus == 'ttest':
            larger = np.argmax(result.rankdf['mean'].values)
            smaller = int(bool(larger - 1))
            if result.all_normal:
                print("Because we have only two populations and both populations are normal, we use the t-test to "
                      "determine differences between the mean values of the populations and report the mean value (M)"
                      "and the standard deviation (SD) for each population. ")
            else:
                if len(not_normal) == 1:
                    notnormal_str = 'one of them is'
                else:
                    notnormal_str = 'both of them are'
                print("Because we have only two populations and %s not normal, we use should Wilcoxon's signed rank "
                      "test to determine the differences in the central tendency and report the median (MD) and the "
                      "median absolute deviation (MAD) for each population. However, the user decided to force the "
                      "use of the t-test which assumes normality of all populations and we report the mean value (M) "
                      "and the standard deviation (SD) for each population." % notnormal_str)
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of the paired t-test that the mean values of "
                      "the populations %s are are equal. Therefore, we "
                      "assume that there is no statistically significant difference between the mean values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of the paired t-test that the mean values of the "
                      "populations %s are "
                      "equal. Therefore, we assume that the mean value of %s is "
                      "significantly larger than the mean value of %s with a %s effect size (%s=%.*f)."
                      % (decimal_places, result.pvalue,
                         create_population_string(result.rankdf.index, with_stats=True),
                         result.rankdf.index[larger], result.rankdf.index[smaller],
                         result.rankdf.magnitude.iloc[larger], effect_size, decimal_places, result.rankdf.effect_size.iloc[larger]))
        elif result.omnibus == 'wilcoxon':
            larger = np.argmax(result.rankdf['median'].values)
            smaller = int(bool(larger - 1))
            if result.all_normal:
                print("Because we have only two populations and both populations are normal, we should use the t-test "
                      "to determine differences between the mean values of the populations and report the mean value "
                      "(M) and the standard deviation (SD) for each population. However, the user decided to force the "
                      "use of the less powerful Wilcoxon signed rank test and we report the median (MD) and the median "
                      "absolute devivation (MAD) for each population.")
            else:
                if len(not_normal) == 1:
                    notnormal_str = 'one of them is'
                else:
                    notnormal_str = 'both of them are'
                print("Because we have only two populations and %s not normal, we use Wilcoxon's signed rank test to "
                      "determine the differences in the central tendency and report the median (MD) and the median "
                      "absolute deviation (MAD) for each population." % notnormal_str)
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of Wilcoxon's signed rank test that "
                      "population %s is not greater than population %s . Therefore, we "
                      "assume that there is no statistically significant difference between the medians of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index[larger], with_stats=True),
                                        create_population_string(result.rankdf.index[smaller], with_stats=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of Wilcoxon's signed rank test that population "
                      "%s is not greater than population %s. Therefore, we assume "
                      "that the median of %s is "
                      "significantly larger than the median value of %s with a %s effect size (%s=%.*f)."
                      % (decimal_places, result.pvalue,
                         create_population_string(result.rankdf.index[larger], with_stats=True),
                         create_population_string(result.rankdf.index[smaller], with_stats=True),
                         result.rankdf.index[larger], result.rankdf.index[smaller],
                         result.rankdf.magnitude.iloc[larger], effect_size, decimal_places, result.rankdf.effect_size.iloc[larger]))
        else:
            raise ValueError('Unknown omnibus test for difference in the central tendency: %s' % result.omnibus)
    else:
        if result.all_normal:
            if result.homoscedastic:
                print("We applied Bartlett's test for homogeneity and failed to reject the null hypothesis "
                      "(p=%.*f) that the data is homoscedastic. Thus, we assume that our data is "
                      "homoscedastic." % (decimal_places, result.pval_homogeneity))
            else:
                print("We applied Bartlett's test for homogeneity and reject the null hypothesis (p=%.*f) that the"
                      "data is homoscedastic. Thus, we assume that our data is "
                      "heteroscedastic." % (decimal_places, result.pval_homogeneity))

        if result.omnibus == 'anova':
            if result.all_normal and result.homoscedastic:
                print("Because we have more than two populations and all populations are normal and homoscedastic, we "
                      "use repeated measures ANOVA as omnibus "
                      "test to determine if there are any significant differences between the mean values of the "
                      "populations. If the results of the ANOVA test are significant, we use the post-hoc Tukey HSD "
                      "test to infer which differences are significant. We report the mean value (M) and the standard "
                      "deviation (SD) for each population. Populations are significantly different if their confidence "
                      "intervals are not overlapping.")
            else:
                if result.all_normal:
                    print(
                        "Because we have more than two populations and the populations are normal but heteroscedastic, "
                        "we should use the non-parametric Friedman test "
                        "as omnibus test to determine if there are any significant differences between the mean values "
                        "of the populations. However, the user decided to force the use of "
                        "repeated measures ANOVA as omnibus test which assume homoscedascity to determine if there are "
                        "any significant difference between the mean values of the populations. If the results of the "
                        "ANOVA test are significant, we use the post-hoc Tukey HSD test to infer which differences are "
                        "significant. We report the mean value (M) and the standard deviation (SD) for each "
                        "population. Populations are significantly different if their confidence intervals are not "
                        "overlapping.")
                else:
                    if len(not_normal) == 1:
                        notnormal_str = 'one of them is'
                    else:
                        notnormal_str = 'some of them are'
                    print("Because we have more than two populations and the populations and %s not normal, "
                          "we should use the non-parametric Friedman test "
                          "as omnibus test to determine if there are any significant differences between the median "
                          "values of the populations and report the median (MD) and the median absolute deviation "
                          "(MAD). However, the user decided to force the use of repeated measures ANOVA as omnibus "
                          "test which assume homoscedascity to determine if there are any significant difference "
                          "between the mean values of the populations. If the results of the ANOVA test are "
                          "significant, we use the post-hoc Tukey HSD test to infer which differences are "
                          "significant. We report the mean value (M) and the standard deviation (SD) for each "
                          "population. Populations are significantly different if their confidence intervals are not "
                          "overlapping." % (notnormal_str))
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of the repeated measures ANOVA that there is "
                      "a difference between the mean values of the populations %s. Therefore, we "
                      "assume that there is no statistically significant difference between the mean values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of the repeated measures ANOVA that there is "
                      "a difference between the mean values of the populations %s. Therefore, we "
                      "assume that there is a statistically significant difference between the mean values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True)))
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
            if result.all_normal and result.homoscedastic:
                print("Because we have more than two populations and all populations are normal and homoscedastic, we "
                      "should use repeated measures ANOVA as omnibus "
                      "test to determine if there are any significant differences between the mean values of the "
                      "populations. However, the user decided to force the use of the less powerful Friedman test as "
                      "omnibus test to determine if there are any significant differences between the mean values "
                      "of the populations. We report the mean value (M), the standard deviation (SD) and the mean rank "
                      "(MR) among all populations over the samples. Differences between populations are significant, "
                      "if the difference of the mean rank is greater than the critical distance CD=%.*f of the Nemenyi "
                      "test." % (decimal_places, result.cd))
            elif result.all_normal:
                print("Because we have more than two populations and the populations are normal but heteroscedastic, "
                      "we use the non-parametric Friedman test "
                      "as omnibus test to determine if there are any significant differences between the mean values "
                      "of the populations. We use the post-hoc Nemenyi test to infer which differences are "
                      "significant. We report the mean value (M), the standard deviation (SD) and the mean rank (MR) "
                      "among all populations over the samples. Differences between populations are significant, if the "
                      "difference of the mean rank is greater than the critical distance CD=%.*f of the Nemenyi "
                      "test." % (decimal_places, result.cd))
            else:
                if len(not_normal) == 1:
                    notnormal_str = 'one of them is'
                else:
                    notnormal_str = 'some of them are'
                print("Because we have more than two populations and the populations and %s not normal, "
                      "we use the non-parametric Friedman test "
                      "as omnibus test to determine if there are any significant differences between the median values "
                      "of the populations. We use the post-hoc Nemenyi test to infer which differences are "
                      "significant. We report the median (MD), the median absolute deviation (MAD) and the mean rank "
                      "(MR) among all populations over the samples. Differences between populations are significant, "
                      "if the difference of the mean rank is greater than the critical distance CD=%.*f of the Nemenyi "
                      "test." % (notnormal_str, decimal_places, result.cd))
            if result.pvalue >= result.alpha:
                print("We failed to reject the null hypothesis (p=%.*f) of the Friedman test that there is no "
                      "difference in the central tendency of the populations %s. Therefore, we "
                      "assume that there is no statistically significant difference between the median values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True, with_rank=True)))
            else:
                print("We reject the null hypothesis (p=%.*f) of the Friedman test that there is no "
                      "difference in the central tendency of the populations %s. Therefore, we "
                      "assume that there is a statistically significant difference between the median values of the "
                      "populations." % (decimal_places, result.pvalue,
                                        create_population_string(result.rankdf.index, with_stats=True, with_rank=True)))
                meanranks, names, groups = get_sorted_rank_groups(result, False)
                if len(groups) == 0:
                    print("Based on the post-hoc Nemenyi test, we assume that all differences between the populations "
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
                    print("Based on the post-hoc Nemenyi test, we assume that there are no significant differences "
                          "within the following groups: %s. All other differences are "
                          "significant." % ("; ".join(groupstrs)))
        else:
            raise ValueError('Unknown omnibus test for difference in the central tendency: %s' % result.omnibus)


def latex_table(result, *, decimal_places=3, label=None, effect_size_relation="best", posterior_relation="best"):
    """
    Creates a latex table from the results dataframe of the statistical analysis.

    # Parameters

    result (RankResult):
        Should be the return value the autorank function.

    decimal_places (int, default=3):
        Number of decimal places that are used for the report.

    label (str, default=None):
        Label of the table. Defaults to 'tbl:stat_results' if None.

    effect_size_relation (str, default="best"):
        Specifies which effect size relation is used in the table. Can be "best", "above", or both. 
        If "best", the effect size is compute in relation to the best-ranked value.
        If "above", the effect size is computed in relation to the value above in the row above. 
        With "both", both the best and the above are included in the table.
        _(New in Version 1.3.0)_

    posterior_relation (str, default="best"):
        Specifies which posterior relation is used in the table. Can be "best", "above", or both.
        If "best", the posterior is computed in relation to the best-ranked value.
        If "above", the posterior is computed in relation to the value above in the row above.
        With "both", both the best and the above are included in the table.
        _(New in Version 1.3.0)_

    """
    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")
    if effect_size_relation not in {'best', 'above', 'both'}:
        raise ValueError("effect_size_relation must be one of 'best', 'above', or 'both'.")
    if posterior_relation not in {'best', 'above', 'both'}:
        raise ValueError("posterior_relation must be one of 'best', 'above', or 'both'.")

    if label is None:
        label = 'tbl:stat_results'

    table_df = result.rankdf.copy(deep=True)
    columns = table_df.columns.to_list()
    if result.omnibus != 'bayes' and result.pvalue >= result.alpha or \
       result.omnibus == 'bayes' and len({'smaller', 'larger'}.intersection(set(result.rankdf['decision']))) == 0:
        columns.remove('effect_size')
        columns.remove('magnitude')
    if result.posthoc == 'tukeyhsd':
        columns.remove('meanrank')
    if result.omnibus == 'bayes':
        table_df.at[table_df.index[0], 'decision'] = '-'
        table_df.at[table_df.index[0], 'decision_above'] = '-'
    columns.insert(columns.index('ci_lower'), 'CI')
    columns.remove('ci_lower')
    columns.remove('ci_upper')
    rename_map = {}
    if result.effect_size == 'cohen_d':
        if effect_size_relation == 'best':
            rename_map['effect_size'] = '$d$'
            columns.remove('effect_size_above')
        elif effect_size_relation == 'above':
            rename_map['effect_size_above'] = '$d$'
            columns.remove('effect_size')
        elif effect_size_relation == 'both':
            rename_map['effect_size'] = '$d$ (best)'
            rename_map['effect_size_above'] = '$d$ (above)'
    elif result.effect_size == 'cliff_delta':
        if effect_size_relation == 'best':
            rename_map['effect_size'] = r'D-E-L-T-A'
            columns.remove('effect_size_above')
        elif effect_size_relation == 'above':
            rename_map['effect_size_above'] = r'D-E-L-T-A'
            columns.remove('effect_size')
        elif effect_size_relation == 'both':
            rename_map['effect_size'] = r'D-E-L-T-A (best)'
            rename_map['effect_size_above'] = r'D-E-L-T-A (above)'
    elif result.effect_size == 'akinshin_gamma':
        if effect_size_relation == 'best':
            rename_map['effect_size'] = r'G-A-M-M-A'
            columns.remove('effect_size_above')
        elif effect_size_relation == 'above':
            rename_map['effect_size_above'] = r'G-A-M-M-A'
            columns.remove('effect_size')
        elif effect_size_relation == 'both':
            rename_map['effect_size'] = r'G-A-M-M-A (best)'
            rename_map['effect_size_above'] = r'G-A-M-M-A (above)'
    if effect_size_relation == 'best':
        rename_map['magnitude'] = 'Magnitude'
        columns.remove('magnitude_above')
    elif effect_size_relation == 'above':
        rename_map['magnitude_above'] = 'Magnitude'
        columns.remove('magnitude')
    elif effect_size_relation == 'both':
        rename_map['magnitude'] = 'Magnitude (best)'
        rename_map['magnitude_above'] = 'Magnitude (above)'
    rename_map['mad'] = 'MAD'
    rename_map['median'] = 'MED'
    rename_map['meanrank'] = 'MR'
    rename_map['mean'] = 'M'
    rename_map['std'] = 'SD'
    if posterior_relation == 'best':
        rename_map['decision'] = 'Decision'
        if 'decision_above' in columns:
            columns.remove('decision_above')
            columns.remove('p_equal_above')
            columns.remove('p_smaller_above')
    elif posterior_relation == 'above':
        rename_map['decision_above'] = 'Decision'
        if 'decision' in columns:
            columns.remove('decision')
            columns.remove('p_equal')
            columns.remove('p_smaller')
    elif posterior_relation == 'both':
        rename_map['decision'] = 'Decision (best)'
        if 'decision_above' in columns:
            rename_map['decision_above'] = 'Decision (above)'
    format_string = '[{0[ci_lower]:.' + str(decimal_places) + 'f}, {0[ci_upper]:.' + str(decimal_places) + 'f}]'
    table_df['CI'] = table_df.agg(format_string.format, axis=1)
    table_df = table_df[columns]
    table_df = table_df.rename(rename_map, axis='columns')

    float_format = lambda x: ("{:0." + str(decimal_places) + "f}").format(x) if not np.isnan(x) else '-'
    table_string = table_df.to_latex(float_format=float_format, na_rep='-').strip()
    table_string = table_string.replace('D-E-L-T-A', r'$\delta$')
    table_string = table_string.replace('G-A-M-M-A', r'$\gamma$')
    if posterior_relation == 'best':
        table_string = table_string.replace(r'p_equal', r'$P(\textit{equal})$')
        table_string = table_string.replace(r'p_smaller', r'$P(\textit{smaller})$')
    elif posterior_relation == 'above':
        table_string = table_string.replace(r'p_equal_above', r'$P(\textit{equal})$')
        table_string = table_string.replace(r'p_smaller_above', r'$P(\textit{smaller})$')
    elif posterior_relation == 'both':
        table_string = table_string.replace(r'p_equal_above', r'$P(\textit{equal})$ (above)')
        table_string = table_string.replace(r'p_smaller_above', r'$P(\textit{smaller})$ (above)')
        table_string = table_string.replace(r'p_equal', r'$P(\textit{equal})$ (best)')
        table_string = table_string.replace(r'p_smaller', r'$P(\textit{smaller})$ (best)')
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(table_string)
    print(r"\caption{Summary of populations}")
    print(r"\label{%s}" % label)
    print(r"\end{table}")


def latex_report(result, *, decimal_places=3, prefix="", generate_plots=True, figure_path="", complete_document=True):
    """
    Creates a latex report of the statistical analysis.

    # Parameters

    result (AutoRank):
        Should be the return value the autorank function.

    decimal_places (int, default=3):
        Number of decimal places that are used for the report.

    prefix (str, default=""):
        Prefix that is added before all labels and plot file names.

    generate_plots (bool, default=True):
        Decides if plots are generated, if the results are statistically significant.

    figure_path (str, default=""):
        Path where the plots shall be written to. Ignored if generate_plots is False.

    complete_document (bool, default=True):
        Generates a complete latex document if true. Otherwise only a single section is generated.
    """
    if not isinstance(result, RankResult):
        raise TypeError("result must be of type RankResult and should be the outcome of calling the autorank function.")

    if complete_document:
        print(r"\documentclass{article}")
        print()
        print(r"\usepackage{graphicx}")
        print(r"\usepackage{booktabs}")
        print()
        print(r"\begin{document}")
        print()

    print(r"\section{Results}")
    print(r"\label{sec:%sresults}" % prefix)
    print()
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    create_report(result, decimal_places=decimal_places)
    report = sys.stdout.getvalue()
    sys.stdout = old_stdout
    report = report.replace("_", r"\_")
    report = report.replace("+-", r"$\pm$")
    report = report.replace("(d=", "($d$=")
    report = report.replace("(delta=", r"($\delta$=")
    report = report.replace("is alpha", r"$\alpha$")
    print(report.strip())
    print()

    if len(result.rankdf) > 2:
        latex_table(result, decimal_places=decimal_places, label='tbl:%sstat_results' % prefix)
        print()

    if result.omnibus != 'wilcoxon' and result.omnibus != 'bayes' and generate_plots and result.pvalue < result.alpha:
        # only include plots if the results are significant
        plot_stats(result)
        if len(figure_path) > 0 and not figure_path.endswith("/"):
            figure_path += '/'
        figure_path = "%s%sstat_results.pdf" % (figure_path, prefix)
        plt.savefig(figure_path)

        print(r"\begin{figure}[h]")
        print(r"\includegraphics[]{%s}" % figure_path)
        if result.posthoc == 'nemenyi':
            print(r"\caption{CD diagram to visualize the results of the Nemenyi post-hoc test. The horizontal lines "
                  r"indicate that differences are not significant.}")
        elif result.posthoc == 'TukeyHSD' or result.posthoc == 'ttest':
            print(r"\caption{Confidence intervals and mean values of the populations.}")
        else:
            # fallback in case of unknown post-hoc test. should not happen
            print(r"\caption{Plot of the results}")
        print(r"\label{fig:%sstats_fig}" % prefix)
        print(r"\end{figure}")
        print()

    if complete_document:
        print(r"\end{document}")
