import unittest
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autorank import *
from autorank._util import RankResult

pd.set_option('display.max_columns', 20)


class TestAutorank(unittest.TestCase):

    def setUp(self):
        print("In method", self._testMethodName)
        print('-------------------------------')
        self.sample_size = 50
        self.verbose = True
        np.random.seed(42)
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        print('-------------------------------')
        self.tmp_dir.cleanup()

    # Uncomment to view created plots
#    @classmethod
#    def tearDownClass(cls):
#        plt.show()

    def test_autorank_normal_homoscedactic_two(self):
        std = 0.15
        means = [0.3, 0.7]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        print("BEGIN FREQUENTIST ANALYSIS")
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'ttest')
        self.assertTrue(res.pvalue < res.alpha)
        plot_stats(res)
        plt.draw()
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_normal_homoscedactic_two_no_difference(self):
        std = 0.15
        means = [0.3, 0.3]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'ttest')
        self.assertFalse(res.pvalue < res.alpha)
        try:
            plot_stats(res)
            self.fail("ValueError expected")
        except ValueError:
            pass
        plot_stats(res, allow_insignificant=True)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_nonnormal_homoscedactic_two(self):
        std = 0.3
        means = [0.2, 0.5]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertFalse(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'wilcoxon')
        self.assertTrue(res.pvalue < res.alpha)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertFalse(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_nonnormal_homoscedactic_two_no_difference(self):
        std = 0.3
        means = [0.2, 0.2]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertFalse(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'wilcoxon')
        self.assertTrue(res.pvalue >= res.alpha)
        try:
            plot_stats(res)
            self.fail("ValueError expected")
        except ValueError:
            pass
        plot_stats(res, allow_insignificant=True)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertFalse(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_normal_homsocedactic(self):
        std = 0.2
        means = [0.2, 0.3, 0.5, 0.55, 0.6, 0.6, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'anova')
        self.assertEqual(res.posthoc, 'tukeyhsd')
        self.assertTrue(res.pvalue < res.alpha)
        plot_stats(res)
        plt.draw()
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_normal_homsocedactic_no_difference(self):
        std = 0.2
        means = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'anova')
        self.assertEqual(res.posthoc, 'tukeyhsd')
        self.assertTrue(res.pvalue >= res.alpha)
        try:
            plot_stats(res)
            self.fail("ValueError expected")
        except ValueError:
            pass
        plot_stats(res, allow_insignificant=True)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_normal_heteroscedactic(self):
        stds = [0.05, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertTrue(res.all_normal)
        self.assertFalse(res.homoscedastic)
        self.assertEqual(res.omnibus, 'friedman')
        self.assertEqual(res.posthoc, 'nemenyi')
        self.assertTrue(res.pvalue < res.alpha)
        plot_stats(res)
        plt.draw()
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_normal_heteroscedactic_no_difference(self):
        stds = [0.05, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertTrue(res.all_normal)
        self.assertFalse(res.homoscedastic)
        self.assertEqual(res.omnibus, 'friedman')
        self.assertEqual(res.posthoc, 'nemenyi')
        self.assertTrue(res.pvalue >= res.alpha)
        try:
            plot_stats(res)
            self.fail("ValueError expected")
        except ValueError:
            pass
        plot_stats(res, allow_insignificant=True)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_nonnormal_homoscedactic(self):
        std = 0.3
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9, 0.1]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertFalse(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'friedman')
        self.assertEqual(res.posthoc, 'nemenyi')
        self.assertTrue(res.pvalue < res.alpha)
        plot_stats(res)
        plt.draw()
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertFalse(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_nonnormal_homoscedactic_no_difference(self):
        std = 0.3
        means = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertFalse(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'friedman')
        self.assertEqual(res.posthoc, 'nemenyi')
        self.assertTrue(res.pvalue >= res.alpha)
        try:
            plot_stats(res)
            self.fail("ValueError expected")
        except ValueError:
            pass
        plot_stats(res, allow_insignificant=True)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertFalse(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_nonnormal_heteroscedactic(self):
        stds = [0.1, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size).clip(0, 1)
        res_asc = autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertFalse(res.all_normal)
        self.assertFalse(res.homoscedastic)
        self.assertEqual(res.omnibus, 'friedman')
        self.assertEqual(res.posthoc, 'nemenyi')
        self.assertTrue(res.pvalue < res.alpha)
        plot_stats(res)
        plot_stats(res_asc)  # this is not covered otherwise, because the CD diagrams respect the order
        plt.draw()
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertFalse(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_nonnormal_heteroscedactic_no_difference(self):
        stds = [0.1, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertFalse(res.all_normal)
        self.assertFalse(res.homoscedastic)
        self.assertEqual(res.omnibus, 'friedman')
        self.assertEqual(res.posthoc, 'nemenyi')
        self.assertTrue(res.pvalue >= res.alpha)
        try:
            plot_stats(res)
            self.fail("ValueError expected")
        except ValueError:
            pass
        plot_stats(res, allow_insignificant=True)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS - DYNAMIC ROPE")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertFalse(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_ropezero(self):
        std = 0.15
        means = [0.3, 0.7]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, alpha=0.05, rope=0, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_absoluterope(self):
        std = 0.15
        means = [0.3, 0.7]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, alpha=0.05, rope=0.1, rope_mode='absolute', nsamples=100, verbose=self.verbose,
                       approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_autorank_force_effect_size(self):
        std = 0.15
        means = [0.3, 0.7]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, 0.05, self.verbose, effect_size='cliff_delta')
        self.assertEqual(res.effect_size, 'cliff_delta')
        create_report(res)

    def test_autorank_force_mode_parametric_two(self):
        std = 0.3
        means = [0.2, 0.5]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, 0.05, self.verbose, force_mode='parametric')
        self.assertFalse(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'ttest')
        create_report(res)
        res = autorank(data, 0.05, self.verbose, force_mode='parametric', approach='bayesian')
        create_report(res)

    def test_autorank_force_mode_nonparametric_two(self):
        std = 0.15
        means = [0.3, 0.7]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, 0.05, self.verbose, force_mode='nonparametric')
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'wilcoxon')
        plot_stats(res)
        plt.draw()
        create_report(res)
        res = autorank(data, 0.05, self.verbose, force_mode='nonparametric', approach='bayesian')
        create_report(res)

    def test_autorank_force_mode_parametric_multiple_heteroscedastic(self):
        stds = [0.05, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size)
        res = autorank(data, 0.05, self.verbose, force_mode='parametric')
        self.assertTrue(res.all_normal)
        self.assertFalse(res.homoscedastic)
        self.assertEqual(res.omnibus, 'anova')
        self.assertEqual(res.posthoc, 'tukeyhsd')
        plot_stats(res)
        plt.draw()
        create_report(res)
        res = autorank(data, 0.05, self.verbose, force_mode='parametric', approach='bayesian')
        create_report(res)

    def test_autorank_force_mode_parametric_multiple_nonnormal(self):
        std = 0.3
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9, 0.1]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, 0.05, self.verbose, force_mode='parametric')
        self.assertFalse(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'anova')
        self.assertEqual(res.posthoc, 'tukeyhsd')
        plot_stats(res)
        plt.draw()
        create_report(res)
        res = autorank(data, 0.05, self.verbose, force_mode='parametric', approach='bayesian')
        create_report(res)

    def test_autorank_force_mode_nonparametric_multiple(self):
        std = 0.2
        means = [0.2, 0.3, 0.5, 0.55, 0.6, 0.6, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size)
        res = autorank(data, 0.05, self.verbose, force_mode='nonparametric')
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'friedman')
        self.assertEqual(res.posthoc, 'nemenyi')
        plot_stats(res)
        plt.draw()
        create_report(res)
        res = autorank(data, 0.05, self.verbose, force_mode='nonparametric', approach='bayesian')
        create_report(res)

    def test_autorank_invalid(self):
        self.assertRaises(TypeError, autorank,
                          data="foo")
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1], [2], [3], [4], [5], [6]]))
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8]]))
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          alpha="foo")
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          alpha=1.1)
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          alpha=-0.05)
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          verbose="foo")
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          order=True)
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          order="foo")
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          approach=True)
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          approach="foo")
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          rope="foo")
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          rope=-1.0)
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          rope_mode=0)
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          rope_mode="foo")
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          nsamples="foo")
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          nsamples=10.5)
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          nsamples=0)
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          effect_size=0)
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          effect_size="foo")
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          force_mode=0)
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]]),
                          force_mode="foo")
        print(pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]], columns=['A', 'B']))
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]], columns=['A', 'B']),
                          plot_order="foo")
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]], columns=['A', 'B']),
                          plot_order=["A"])
        self.assertRaises(TypeError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]], columns=['A', 'B']),
                          plot_order=["A", 1])
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]], columns=['A', 'B']),
                          plot_order=["A", "C"])
        self.assertRaises(ValueError, autorank,
                          data=pd.DataFrame(data=[[1, 2], [3, 4], [5, 6], [7, 8], [9, 0], [1, 2]], columns=['A', 'B']),
                          plot_order=["A", "A"])

    def test_plot_stats_invalid(self):
        self.assertRaises(TypeError, plot_stats,
                          result="foo")
        res = RankResult(None, None, None, 'bayes', None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None, None)
        self.assertRaises(ValueError, plot_stats,
                          result=res)

    def test_create_report_invalid(self):
        self.assertRaises(TypeError, create_report,
                          result="foo")

    def test_latex_report_invalid(self):
        self.assertRaises(TypeError, latex_report,
                          result="foo")
        
    def test_latex_table_invalid(self):
        self.assertRaises(TypeError, latex_table,
                          result="foo")
        res = RankResult(None, None, None, 'bayes', None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None, None)
        self.assertRaises(ValueError, latex_table,
                          result=res, effect_size_relation='foo')
        self.assertRaises(ValueError, latex_table,
                          result=res, posterior_relation='foo')

    def test_realdata(self):
        data = pd.DataFrame().from_dict(
            {'Nearest Neighbors': {'moons': 0.95,
                                   'circles': 0.8699999999999999,
                                   'linsep': 0.96,
                                   'iris': 0.9533333333333334,
                                   'digits': 0.9622147183989795,
                                   'wine': 0.954499914000688,
                                   'breast_cancer': 0.9647523982369716},
             'Linear SVM': {'moons': 0.8399999999999999,
                            'circles': 0.45999999999999996,
                            'linsep': 0.9400000000000001,
                            'iris': 0.9133333333333333,
                            'digits': 0.956694555961092,
                            'wine': 0.97218782249742,
                            'breast_cancer': 0.9754666839512574},
             'RBF SVM': {'moons': 0.9400000000000001,
                         'circles': 0.89,
                         'linsep': 0.95,
                         'iris': 0.9466666666666667,
                         'digits': 0.10576250429309801,
                         'wine': 0.3992539559683522,
                         'breast_cancer': 0.6274274047186933},
             'Gaussian Process': {'moons': 0.9199999999999999,
                                  'circles': 0.9,
                                  'linsep': 0.9099999999999999,
                                  'iris': 0.9666666666666668,
                                  'digits': 0.9510904874693658,
                                  'wine': 0.9780701754385965,
                                  'breast_cancer': 0.979038112522686},
             'Decision Tree': {'moons': 0.9,
                               'circles': 0.82,
                               'linsep': 0.9000000000000001,
                               'iris': 0.96,
                               'digits': 0.644574799023532,
                               'wine': 0.894702442380461,
                               'breast_cancer': 0.9229506957047791},
             'Random Forest': {'moons': 0.9199999999999999,
                               'circles': 0.8600000000000001,
                               'linsep': 0.95,
                               'iris': 0.9466666666666667,
                               'digits': 0.7922906126138967,
                               'wine': 0.9724802201582387,
                               'breast_cancer': 0.9456669691470054},
             'Neural Net': {'moons': 0.85,
                            'circles': 0.8700000000000001,
                            'linsep': 0.9400000000000001,
                            'iris': 0.9733333333333334,
                            'digits': 0.9521091316664249,
                            'wine': 0.9783625730994153,
                            'breast_cancer': 0.977222150203094},
             'AdaBoost': {'moons': 0.9199999999999999,
                          'circles': 0.82,
                          'linsep': 0.9100000000000001,
                          'iris': 0.9533333333333334,
                          'digits': 0.27032591395830086,
                          'wine': 0.8692982456140351,
                          'breast_cancer': 0.9596123930515945},
             'Naive Bayes': {'moons': 0.8400000000000001,
                             'circles': 0.9099999999999999,
                             'linsep': 0.95,
                             'iris': 0.9533333333333334,
                             'digits': 0.757390252067086,
                             'wine': 0.9616959064327485,
                             'breast_cancer': 0.9315681444991789},
             'QDA': {'moons': 0.8399999999999999,
                     'circles': 0.8800000000000001,
                     'linsep': 0.9400000000000001,
                     'iris': 0.9800000000000001,
                     'digits': 0.8327893459647431,
                     'wine': 0.9780701754385965,
                     'breast_cancer': 0.9559804684124102}})
        res = autorank(data, 0.05, self.verbose)
        plot_stats(res, allow_insignificant=True)
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_bayes_real_world(self):
        data = pd.DataFrame(
         {'Blizzard': {0: 0.28677562910530585, 1: 0.18803486971609495, 2: 0.3318155299087674, 3: 0.2337428123223579,
                       4: 0.20436577016364246, 5: 0.3928141303141303, 10: 0.21808391183391185, 11: 0.3488636363636364,
                       13: 0.19159496753246752, 14: 0.22057962646614088, 17: 0.4170813492063495, 18: 0.3652133580705008,
                       19: 0.44788286302175184, 20: 0.21976328809035572, 21: 0.2535163337762348, 22: 0.1964590028240334,
                       23: 0.26157183938929995, 24: 0.34817221215477023, 25: 0.06453497851802935, 26: 0.2646603394464393,
                       27: 0.2288482285421719, 28: 0.25049573442430595, 29: 0.2759963719434383, 30: 0.28873529997951275,
                       31: 0.2711802666619741, 32: 0.2758169442772093, 33: 0.29300865800865794, 34: 0.28920466899918945,
                       35: 0.223201333837354, 36: 0.3295345070221307, 37: 0.15489069537364994},
         'BugLocator': {0: 0.3913923833443673, 1: 0.2227783818221887, 2: 0.33810327700594617, 3: 0.295288033533152,
                        4: 0.3019967619974965, 5: 0.4812974026663249, 10: 0.3541311576586792, 11: 0.5746520904034691,
                        13: 0.2976822464241802, 14: 0.2705378569341504, 17: 0.7733265455106109, 18: 0.3803902776966097,
                        19: 0.6143439687884129, 20: 0.3575321981224423, 21: 0.4058654523737525, 22: 0.22167833496104894,
                        23: 0.36753061892703454, 24: 0.5291660588391219, 25: 0.08385334344896245,
                        26: 0.4226918380190359, 27: 0.3749533838680448, 28: 0.3319937956680037, 29: 0.4453744900384333,
                        30: 0.4055971273361142, 31: 0.4650505817180286, 32: 0.4530464133884682, 33: 0.4628466981466774,
                        34: 0.4492369766018965, 35: 0.23587434138343766, 36: 0.4770094999976819,
                        37: 0.2447661372034597},
         'BRTracer': {0: 0.4652435073482237, 1: 0.2602258859276184, 2: 0.3844674037830659, 3: 0.33973563253303185,
                      4: 0.338021181688019, 5: 0.5417314518452468, 10: 0.37023754671799936, 11: 0.5948835082246171,
                      13: 0.29114609133599245, 14: 0.3512933561128432, 17: 0.7993095041490482, 18: 0.4762522192644304,
                      19: 0.6434172806065415, 20: 0.35937785821395, 21: 0.4616035477365531, 22: 0.27561056610164986,
                      23: 0.4276905856529294, 24: 0.5954548187908132, 25: 0.13957702044689868, 26: 0.4331338825534523,
                      27: 0.4112118150769037, 28: 0.4165263196499692, 29: 0.4642558592522809, 30: 0.5053494051060591,
                      31: 0.4887435247957734, 32: 0.4763809515359289, 33: 0.4803407268997695, 34: 0.5194003206624807,
                      35: 0.2746692294004946, 36: 0.497533326991715, 37: 0.2714802339478054},
         'BLUiR': {0: 0.2703913506043468, 1: 0.22225370473948705, 2: 0.28820364348821353, 3: 0.18409331771696646,
                   4: 0.26405691962123873, 5: 0.3209493632196244, 10: 0.2245403918813755, 11: 0.20641064701036893,
                   13: 0.2614473754330304, 14: 0.24427184808976005, 17: 0.4634006106689384, 18: 0.4663551513390043,
                   19: 0.5615603725908402, 20: 0.26992522301387656, 21: 0.2786655258844377, 22: 0.19096114872116546,
                   23: 0.298978899910992, 24: 0.4374956655002684, 25: 0.0818398939076497, 26: 0.2792585426671105,
                   27: 0.2577548339059024, 28: 0.2594583903170947, 29: 0.40810411581060174, 30: 0.3399860808689741,
                   31: 0.368538114376461, 32: 0.33925029668352724, 33: 0.33072974674055033, 34: 0.3608551984179025,
                   35: 0.23088841763780385, 36: 0.4326046216441299, 37: 0.17132312145133974},
         'AmaLgam': {0: 0.2027876384324849, 1: 0.18585430566800135, 2: 0.1843375185075991, 3: 0.16288046412851012,
                     4: 0.21139015477475706, 5: 0.2204085916539479, 10: 0.17758841027682615, 11: 0.19651970962322327,
                     13: 0.31300208682039793, 14: 0.168716863830717, 17: 0.41827612886703935, 18: 0.3413104361220556,
                     19: 0.4790356698504831, 20: 0.2427394430565784, 21: 0.22113704311639573, 22: 0.19051790818308909,
                     23: 0.25144829923235684, 24: 0.37201198436969424, 25: 0.06881106729669698, 26: 0.2403433288847427,
                     27: 0.17917243187885254, 28: 0.20366440704045125, 29: 0.3293235522208235, 30: 0.2855917587190937,
                     31: 0.3247148826346217, 32: 0.31116262580938525, 33: 0.2450526907832962, 34: 0.2820493383250606,
                     35: 0.16113019722145927, 36: 0.3385417998681097, 37: 0.12784497553081395},
         'BLIA': {0: 0.478723384550084, 1: 0.2473838565520271, 2: 0.3908128642897997, 3: 0.3277801847389506,
                  4: 0.3289366702365609, 5: 0.6182498235807471, 10: 0.40756343077771656, 11: 0.5881283068783069,
                  13: 0.3614173848690285, 14: 0.288069526102313, 17: 0.8227297068740083, 18: 0.6058580418699465,
                  19: 0.7516482283148949, 20: 0.39190291022382173, 21: 0.4256317021787883, 22: 0.3013374055234717,
                  23: 0.4335381226925536, 24: 0.6439265304545647, 25: 0.12521657471823575, 26: 0.4038895682933508,
                  27: 0.3991998267656183, 28: 0.394396465850021, 29: 0.4918737565084129, 30: 0.5239547093484445,
                  31: 0.5157670325187115, 32: 0.481193208418459, 33: 0.5230256099266045, 34: 0.5044055202416154,
                  35: 0.2659556813080341, 36: 0.5348486854891374, 37: 0.2854928702382727},
         'Locus': {0: 0.4298684839195832, 1: 0.2302550135484991, 2: 0.3951885831999812, 3: 0.2568351097482315,
                   4: 0.33121315461322703, 5: 0.4195378405944936, 10: 0.2764668116217563, 11: 0.08035714285714285,
                   13: 0.28194305047856233, 14: 0.3080062258756191, 17: 0.7031379228199923, 18: 0.37021653619282896,
                   19: 0.5582876382876382, 20: 0.03121643394199785, 21: 0.4651263488937626, 22: 0.25672896964006353,
                   23: 0.4362008059126377, 24: 0.5366603746070315, 25: 0.07838968968546615, 26: 0.3903590633890013,
                   27: 0.3721630653843248, 28: 0.3845121791524532, 29: 0.3859428877439121, 30: 0.5439050127248313,
                   31: 0.5236548863118382, 32: 0.4318630002495771, 33: 0.3680012586317358, 34: 0.5024821158590015,
                   35: 0.2588491744973932, 36: 0.4088153074094002, 37: 0.15805194873964126},
         'Broccoli': {0: 0.4943008072930222, 1: 0.4239878173588785, 2: 0.6923275883530206, 3: 0.38659648326994156,
                      4: 0.38089439786821233, 5: 0.6680307261239858, 10: 0.5595198864817289, 11: 0.5406197311460469,
                      13: 0.34526357948480446, 14: 0.46565977738459413, 17: 0.7671692422428585, 18: 0.588919184131227,
                      19: 0.7394473327580662, 20: 0.4656242771015692, 21: 0.4968617593412508, 22: 0.38589850154502103,
                      23: 0.46772311571121045, 24: 0.5860236037220198, 25: 0.17894644742319193, 26: 0.4437814115653021,
                      27: 0.4389113437104677, 28: 0.46821858724966375, 29: 0.5208437015668166, 30: 0.6773552015591268,
                      31: 0.5804718443894237, 32: 0.5422938825604393, 33: 0.5700559782822201, 34: 0.5665209759935147,
                      35: 0.38633378023837195, 36: 0.646200149806921, 37: 0.4299884762876898}})
        res = autorank(data, rope=0.5, nsamples=100, verbose=self.verbose, approach='bayesian')

        latex_table(res)
        create_report(res)

    def test_bayes_absolute_rope_output(self):
        """
        Test case for issue #3
        """
        raw = np.array([[0.61874876, 0.61219062],
                        [0.89017217, 0.90443957],
                        [0.62806089, 0.63185734],
                        [0.96929193, 0.97255931],
                        [0.87340513, 0.95460121],
                        [0.84087749, 0.94438674],
                        [0.9863088, 0.98558508],
                        [0.94314842, 0.64510605],
                        [0.9862604, 0.99173966]])
        data = pd.DataFrame()
        data['pop_0'] = raw[:, 0]
        data['pop_1'] = raw[:, 1]

        result = autorank(data, alpha=0.05, verbose=False, approach="bayesian", rope=1.0, rope_mode="absolute")
        create_report(result)
        print(result)

    def test_reording(self):
        """
        Test case for issue #7: pvals of shapiro-wilk are not ordered correctly
        """

        std = 0.2
        means = [0.5, 0.3, 0.7]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size)
        result = autorank(data, 0.05, verbose=True, order="descending", )
        print(result)

    def test_cd_order(self):
        stds = [0.1, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size).clip(0, 1)
        res_asc = autorank(data, 0.05, self.verbose, order='ascending')
        res = autorank(data, 0.05, self.verbose)
        # currently one plot is always broken (on my machine)
        plot_stats(res)
        plot_stats(res_asc)
        plt.draw()

    def test_fix_random_state(self):
        std = 0.15
        means = [0.3, 0.3]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)

        res_42_1 = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian', random_state=42)
        res_42_2 = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian', random_state=42)
        res_43_1 = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian', random_state=43)

        self.assertTrue(res_42_1.posterior_matrix.equals(res_42_2.posterior_matrix))
        self.assertFalse(res_42_1.posterior_matrix.equals(res_43_1.posterior_matrix))
    
    def test_latex_duplicate_columns(self):
        std = 0.3
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
        sample_size = 50
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, sample_size).clip(0, 1)

        res = autorank(data, alpha=0.05, verbose=False)
        latex_table(res)
        latex_table(res)

    def test_named_index(self):
        # test to reproduce issue #16
        data = pd.DataFrame().from_dict({
            'index': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'F1': [-0.108493, 0.100764, 0.578502, 0.690802, 0.280736, 0.43442, 0.568025, 0.952764, 0.339497, -0.334314, 0.7448, 0.946012, 0.317754],
            'F2': [-0.182625, 0.170489, 0.60399, 0.746536, 0.269668, 0.371632, 0.498394, 0.95084, 0.211892, 1.03841e-14, 0.711472, 0.945422, 0.509453],
            'F3': [-0.217593, -0.0353945, 0.592368, 0.70053, 0.278789, 0.425941, 0.458875, 0.949877, 0.337175, 1.03841e-14, 0.74447, 0.950613, 0.354345]
        })
        data = data.set_index("index") # The column "in" becomes the index
        autorank(data, 0.05, True)

    def test_multindex(self):
        # more robustness for issue #16
        data = pd.DataFrame().from_dict({
            'index1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'index2': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
            'F1': [-0.108493, 0.100764, 0.578502, 0.690802, 0.280736, 0.43442, 0.568025, 0.952764, 0.339497, -0.334314, 0.7448, 0.946012, 0.317754],
            'F2': [-0.182625, 0.170489, 0.60399, 0.746536, 0.269668, 0.371632, 0.498394, 0.95084, 0.211892, 1.03841e-14, 0.711472, 0.945422, 0.509453],
            'F3': [-0.217593, -0.0353945, 0.592368, 0.70053, 0.278789, 0.425941, 0.458875, 0.949877, 0.337175, 1.03841e-14, 0.74447, 0.950613, 0.354345]
        })
        data = data.set_index(["index1", "index2"])
        autorank(data, 0.05, True)

    def test_pivot_tables(self):
        # test to reproduce issue #37
        # the underlying issue is that the index and columns are named
        d = {
            "k1": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b", "c", "c", "c", "c", "c", "c"],
            "k2": ["1", "2", "3", "4", "5", "6", "1", "2", "3", "4", "5", "6", "1", "2", "3", "4", "5", "6"],
            "v": [3.6780, 2.2840, 3.3200, 4.3570, 3.1290, 3.4530,
                5.0730, 6.1410, 7.0870, 6.8470, 7.2570, 5.4080,
                7.3680, 10.030, 8.2270, 8.6580, 9.4650, 8.6650]}
        source = pd.DataFrame(data=d)
        table = pd.pivot_table(source, values="v", index="k2", columns="k1")

        autorank(table, alpha=0.05, verbose=True)

    def test_plot_plot_order(self):
        # test for new function that allows sorting within CI plots (see issue #39)
        std = 0.2
        means = [0.2, 0.3, 0.5, 0.55, 0.6, 0.6, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        plot_order = ['pop_0', 'pop_2', 'pop_4', 'pop_6', 'pop_5', 'pop_3', 'pop_1']
        res = autorank(data, 0.05, self.verbose, plot_order=plot_order)
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'anova')
        self.assertEqual(res.posthoc, 'tukeyhsd')
        self.assertTrue(res.pvalue < res.alpha)
        plot_stats(res)
        plt.draw()
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")
        print()
        print("BEGIN BAYESIAN ANALYSIS")
        print()
        res = autorank(data, alpha=0.05, nsamples=100, verbose=self.verbose, approach='bayesian')
        self.assertTrue(res.all_normal)
        self.assertEqual(res.omnibus, 'bayes')
        create_report(res)
        print("----BEGIN LATEX----")
        latex_report(res, generate_plots=True, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_effect_size_relations(self):
        std = 0.3
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9, 0.1]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, 0.05, False)
        print("----BEGIN LATEX -----")
        latex_table(res, effect_size_relation='best')
        latex_table(res, effect_size_relation='above')
        latex_table(res, effect_size_relation='both')
        print("----END LATEX----")

    def test_posterior_relations(self):
        std = 0.3
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9, 0.1]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, alpha=0.05, nsamples=100, verbose=False, approach='bayesian')
        #print("----BEGIN LATEX----")
        latex_table(res, posterior_relation='best')
        latex_table(res, posterior_relation='above')
        latex_table(res, posterior_relation='both')
        print("----END LATEX----")

    def test_posterior_plots(self):
        std = 0.3
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9, 0.1]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = autorank(data, alpha=0.05, nsamples=100, verbose=False, approach='bayesian')
        plot_posterior_maps(res)
        plt.draw()

    def test_posterior_plots_invalid(self):
        self.assertRaises(TypeError, plot_posterior_maps,
                          result="foo")
        res = RankResult(None, None, None, 'bayes', None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None, None)
        res_not_bayes = RankResult(None, None, None, 'foo', None, None, None, None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None)
        self.assertRaises(ValueError, plot_posterior_maps,
                          result=res_not_bayes)
        self.assertRaises(TypeError, plot_posterior_maps,
                          result=res, width='foo')
        self.assertRaises(ValueError, plot_posterior_maps,
                          result=res, width=0.0)
        self.assertRaises(TypeError, plot_posterior_maps,
                          result=res, cmaps='foo')
        self.assertRaises(ValueError, plot_posterior_maps,
                          result=res, cmaps=['foo'])
        self.assertRaises(TypeError, plot_posterior_maps,
                          result=res, annot_colors='foo')
        self.assertRaises(ValueError, plot_posterior_maps,
                          result=res, annot_colors=['foo'])
        self.assertRaises(TypeError, plot_posterior_maps,
                          result=res, axes='foo')
        self.assertRaises(ValueError, plot_posterior_maps,
                          result=res, axes=['foo'])
