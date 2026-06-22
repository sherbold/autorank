"""Tests for the experimental Skillings-Mack test (autorank._utils_experimental).

The expected statistic/df/p-values in ``sm_ref_data.json`` were frozen ahead of
time and cross-checked against the CRAN R implementation of the Skillings-Mack
test. They are reproducible from the mathematical definition of the statistic
and travel with the test, so this module needs only numpy + scipy (no R).

This is an experimental feature that is intended to be moved upstream (e.g., to
scipy); see autorank._utils_experimental for details.

Cross-checking against R (optional, for maintainers)
----------------------------------------------------
The live R cross-check (driving the CRAN ``Skillings.Mack`` package through
rpy2) is not needed to run these tests, but it lives in ``C:\\dev\\SkillingsMackTest``
(``test_equivalence.py``). rpy2 + R + the required R packages are provided by a
micromamba environment named ``smtest``. To run the R comparison::

    export MAMBA_ROOT_PREFIX=$HOME/micromamba-root
    ~/bin/micromamba run -n smtest python test_equivalence.py

The frozen values in ``sm_ref_data.json`` were produced from that same R source.
"""

import json
import os
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from autorank._utils_experimental import skillings_mack

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(HERE, "sm_ref_data.json")


def _load_reference():
    with open(REF_PATH) as fh:
        records = json.load(fh)
    cases = []
    for r in records:
        data = np.array(
            [[np.nan if v is None else v for v in row] for row in r["data"]],
            dtype=float,
        )
        cases.append((r["name"], data, r["statistic"], r["df"], r["pvalue"]))
    return cases


class TestSkillingsMack(unittest.TestCase):

    def test_matches_frozen_reference(self):
        """Statistic, df and p-value match the frozen R reference for every case."""
        for name, data, statistic, df, pvalue in _load_reference():
            res = skillings_mack(data)
            self.assertEqual(res.df, df, msg="%s: df %s != %s" % (name, res.df, df))
            assert_allclose(res.statistic, statistic, rtol=1e-9, atol=1e-10,
                            err_msg="%s: statistic" % name)
            assert_allclose(res.pvalue, pvalue, rtol=1e-9, atol=1e-12,
                            err_msg="%s: pvalue" % name)

    def test_matches_friedman_on_complete_data(self):
        """With complete, tie-free data the test reduces to Friedman."""
        rng = np.random.default_rng(0)
        data = rng.normal(size=(5, 14))  # 5 treatments, 14 blocks
        res = skillings_mack(data)
        fried = stats.friedmanchisquare(*data)
        assert_allclose(res.statistic, fried.statistic, rtol=1e-9)
        assert_allclose(res.pvalue, fried.pvalue, rtol=1e-9)
        self.assertEqual(res.df, data.shape[0] - 1)

    def test_variadic_matches_table(self):
        """The friedman-style variadic call equals the 2-D table call."""
        rng = np.random.default_rng(1)
        data = rng.normal(size=(4, 10))
        assert_allclose(skillings_mack(*data).statistic,
                        skillings_mack(data).statistic)

    def test_permutation_pvalue_in_range(self):
        """The assumption-free permutation p-value is a valid probability."""
        rng = np.random.default_rng(2)
        data = rng.integers(0, 3, size=(4, 8)).astype(float)
        res = skillings_mack(data, simulate_p_value=True, B=2000, random_state=3)
        self.assertGreaterEqual(res.sim_pvalue, 0.0)
        self.assertLessEqual(res.sim_pvalue, 1.0)


if __name__ == "__main__":
    unittest.main()
