import unittest
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autorank import *

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
        autorank(data, 0.05, self.verbose, order='ascending') # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertTrue(res.all_normal)
        self.assertTrue(res.homoscedastic)
        self.assertEqual(res.omnibus, 'ttest')
        self.assertTrue(res.pvalue < res.alpha)
        plot_stats(res)
        plt.draw()
        create_report(res)
        print("----BEGIN LATEX----")
        print(self.tmp_dir)
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
        self.assertFalse(res.pvalue<res.alpha)
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
        self.assertTrue(res.pvalue<res.alpha)
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

    def test_autorank_nonnormal_heteroscedactic(self):
        stds = [0.1, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.2, 0.3, 0.5, 0.8, 0.85, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending')  # check if call works with ascending
        res = autorank(data, 0.05, self.verbose)
        self.assertFalse(res.all_normal)
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

    def test_autorank_nonnormal_heteroscedactic_no_difference(self):
        stds = [0.1, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size).clip(0, 1)
        autorank(data, 0.05, self.verbose, order='ascending') # check if call works with ascending
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

    def test_plot_stats_invalid(self):
        self.assertRaises(TypeError, plot_stats,
                          result="foo")

    def test_create_report_invalid(self):
        self.assertRaises(TypeError, create_report,
                          result="foo")

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

    def test_bayes_normal_homoscedactic_two(self):
        std = 0.15
        means = [0.3, 0.34]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, std, self.sample_size).clip(0, 1)
        res = bayesrank(data, rope=0.01, nsamples=100, verbose=self.verbose, order='ascending') # check if call works with ascending
        print("----BEGIN LATEX----")
        latex_table(res)
        # latex_report(res, generate_plots=False, figure_path=self.tmp_dir.name)
        print("----END LATEX----")

    def test_bayes_nonnormal_heteroscedactic_no_difference(self):
        stds = [0.1, 0.1, 0.5, 0.1, 0.05, 0.05]
        means = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        data = pd.DataFrame()
        for i, mean in enumerate(means):
            data['pop_%i' % i] = np.random.normal(mean, stds[i], self.sample_size).clip(0, 1)
        res = bayesrank(data, rope=0.01, nsamples=100, verbose=self.verbose)
        self.assertFalse(res.all_normal)
        self.assertIsNone(res.homoscedastic)
        self.assertEqual(res.omnibus, 'bayes')
        self.assertEqual(res.posthoc, 'bayes')
        try:
            plot_stats(res)
            self.fail("ValueError expected")
        except ValueError:
            pass

        create_report(res)
        print("----BEGIN LATEX----")
        latex_table(res)
        print("----END LATEX----")