"""Standalone Skillings-Mack test (drop-in for ``friedmanchisquare``).

This is an experimental module. The functionality here is *not* intended to
remain a permanent part of ``autorank``: the Skillings-Mack test is planned to
be contributed upstream (e.g., to ``scipy.stats``), after which ``autorank``
should depend on the upstream implementation instead of this module.

The Skillings-Mack test is a rank test for a randomized block design that
tolerates observations missing at random.  It generalizes the Friedman test;
with complete blocks and no ties the statistic and chi-squared p-value coincide
with Friedman's.

Mathematical definition
-----------------------
For ``m`` treatments across ``n`` blocks, with block ``j`` holding ``s_j``
observed values:

1. Rank the observed values within each block (mid-ranks for ties).
2. Standardize: an observed rank ``r`` in block ``j`` becomes
   ``sqrt(12 / (s_j + 1)) * (r - (s_j + 1) / 2)``; a missing cell becomes ``0``.
3. Adjusted treatment totals ``w[i] = sum_j ahat[i, j]``.
4. Null covariance ``Sigma`` is the weighted Laplacian of the treatment
   co-occurrence graph: ``Sigma[i, g] = -lam[i, g]`` for ``i != g`` and
   ``Sigma[i, i] = sum_{g != i} lam[i, g]``, where ``lam[i, g]`` counts blocks
   containing both treatments.
5. Statistic ``SM = w @ pinv(Sigma) @ w``.
6. ``SM`` is asymptotically chi-squared with ``df = rank(Sigma)``.

Reference: Skillings, J.H., Mack, G.A. (1981) "On the Use of a Friedman-Type
Statistic in Balanced and Unbalanced Block Designs", Technometrics 23(2),
171-177.

Drop-in replacement
-------------------
The public callable :func:`skillings_mack` mirrors the calling convention of
``scipy.stats.friedmanchisquare`` so that, in `autorank`'s ``_util.py``::

    pval = stats.friedmanchisquare(*data.transpose().values).pvalue

can be replaced by::

    pval = skillings_mack(*data.transpose().values).pvalue

Each positional argument is one treatment's measurements across all blocks
(exactly like ``friedmanchisquare``).  ``numpy.nan`` may be used for missing
observations.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.stats import chi2, rankdata

__all__ = ["skillings_mack", "SkillingsMackResult"]


SkillingsMackResult = namedtuple(
    "SkillingsMackResult",
    ["statistic", "pvalue", "df", "sim_pvalue"],
)
SkillingsMackResult.__doc__ = """Result of the Skillings-Mack test.

Attributes
----------
statistic : float
    The Skillings-Mack statistic.
pvalue : float
    P-value from the chi-squared distribution with ``df`` degrees of freedom.
    This is the drop-in analogue of ``friedmanchisquare(...).pvalue``.  It is an
    asymptotic approximation; for small designs or many ties/missing values the
    permutation p-value (``sim_pvalue``) is preferred.
df : int
    Degrees of freedom: the rank of the covariance matrix of the adjusted
    treatment totals (``m - 1`` for a connected design).
sim_pvalue : float or None
    Monte-Carlo estimated p-value when ``simulate_p_value=True``; otherwise
    ``None``.
"""


def _standardized_ranks(table):
    """Standardized within-block ranks and the presence mask.

    ``table`` has shape ``(m, n)`` (treatments by blocks); ``numpy.nan`` marks a
    missing cell.  Observed cells in block ``j`` (size ``s``) become
    ``sqrt(12 / (s + 1)) * (rank - (s + 1) / 2)``; missing cells stay ``0``.
    """
    m, n = table.shape
    present = ~np.isnan(table)
    ahat = np.zeros((m, n), dtype=float)
    for j in range(n):
        rows = np.flatnonzero(present[:, j])
        s = rows.size
        if s == 0:
            continue
        ranks = rankdata(table[rows, j], method="average")
        ahat[rows, j] = np.sqrt(12.0 / (s + 1.0)) * (ranks - (s + 1.0) / 2.0)
    return ahat, present


def _null_covariance(present):
    """Weighted Laplacian of the treatment co-occurrence graph.

    ``present`` is a boolean ``(m, n)`` mask.  With ``lam = present @ present.T``
    (co-occurrence counts), the covariance is ``diag(rowsum(lam_offdiag)) -
    lam_offdiag`` where ``lam_offdiag`` zeroes the diagonal of ``lam``.
    """
    p = present.astype(float)
    cooccurrence = p @ p.T
    np.fill_diagonal(cooccurrence, 0.0)
    degree = cooccurrence.sum(axis=1)
    return np.diag(degree) - cooccurrence


def _sm_statistic(table):
    """Statistic, degrees of freedom, standardized ranks and presence mask."""
    ahat, present = _standardized_ranks(table)
    w = ahat.sum(axis=1)
    sigma = _null_covariance(present)
    sigma_pinv = np.linalg.pinv(sigma)
    df = int(np.linalg.matrix_rank(sigma))
    statistic = float(w @ sigma_pinv @ w)
    return statistic, df, ahat, present, sigma_pinv


def _simulate_pvalue(ahat, present, sigma_pinv, statistic, n_resamples, rng):
    """Monte-Carlo p-value by permuting standardized ranks within each block.

    The presence pattern (hence ``Sigma``) is fixed under within-block
    permutation, so the cached pseudo-inverse is reused.
    """
    n = ahat.shape[1]
    block_rows = [np.flatnonzero(present[:, j]) for j in range(n)]
    at_least = 0
    for _ in range(n_resamples):
        shuffled = np.zeros_like(ahat)
        for j, rows in enumerate(block_rows):
            shuffled[rows, j] = rng.permutation(ahat[rows, j])
        w = shuffled.sum(axis=1)
        if float(w @ sigma_pinv @ w) >= statistic:
            at_least += 1
    return round(at_least / n_resamples, 7)


def _as_table(samples):
    """Build the ``(m, n)`` table from friedman-style treatment arguments.

    Each positional argument is one treatment measured across all blocks, so
    stacking them as rows yields ``m`` treatments by ``n`` blocks.
    """
    rows = [np.asarray(s, dtype=float).ravel() for s in samples]
    lengths = {r.shape[0] for r in rows}
    if len(lengths) != 1:
        raise ValueError("all treatments must have the same number of blocks")
    return np.vstack(rows)


def skillings_mack(*samples, simulate_p_value=False, B=10000, random_state=None):
    """Perform the Skillings-Mack test.

    Parameters
    ----------
    *samples : array_like
        Two or more measurement vectors, one per treatment, each containing one
        value per block.  ``numpy.nan`` marks a missing observation.  This is
        the same calling convention as ``scipy.stats.friedmanchisquare``.
        Alternatively a single 2-D array (``m`` treatments x ``n`` blocks) may
        be passed.
    simulate_p_value : bool, optional
        If True, also compute a Monte-Carlo (permutation) p-value (default
        False).  The permutation p-value is preferred over the asymptotic
        chi-squared ``pvalue`` for small designs or when there are many ties
        and/or missing observations, where the chi-squared approximation is
        unreliable.
    B : int, optional
        Number of Monte-Carlo replications (default 10000).
    random_state : None, int, or numpy.random.Generator, optional
        Seed or generator for the Monte-Carlo simulation.

    Returns
    -------
    SkillingsMackResult
        Named tuple ``(statistic, pvalue, df, sim_pvalue)``.
    """
    if len(samples) == 1 and np.asarray(samples[0]).ndim == 2:
        d = np.asarray(samples[0], dtype=float)
    else:
        d = _as_table(samples)

    m, n = d.shape
    # A block with one (or zero) observations carries no rank information.
    for j in range(n):
        if int(np.isnan(d[:, j]).sum()) >= (m - 1):
            raise ValueError(
                "Each block must contain at least two observations; "
                f"block {j + 1} has fewer."
            )

    statistic, df, ahat, present, sigma_pinv = _sm_statistic(d)
    pvalue = float(chi2.sf(statistic, df))

    sim_pvalue = None
    if simulate_p_value:
        rng = np.random.default_rng(random_state)
        sim_pvalue = _simulate_pvalue(ahat, present, sigma_pinv, statistic, B, rng)

    return SkillingsMackResult(statistic=statistic, pvalue=pvalue, df=df,
                               sim_pvalue=sim_pvalue)


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------
# Run directly:  python skillings_mack.py
if __name__ == "__main__":
    import pandas as pd
    from scipy import stats

    rng = np.random.default_rng(0)

    # 1) Drop-in replacement for scipy.stats.friedmanchisquare in autorank.
    #    `data` is laid out like autorank: rows = blocks/subjects, cols = methods.
    data = pd.DataFrame(rng.normal(size=(15, 5)), columns=list("ABCDE"))
    friedman_p = stats.friedmanchisquare(*data.transpose().values).pvalue
    sm_p = skillings_mack(*data.transpose().values).pvalue
    print("Drop-in replacement (complete data, no ties)")
    print(f"  friedmanchisquare p-value : {friedman_p:.10f}")
    print(f"  skillings_mack    p-value : {sm_p:.10f}")
    print(f"  agree to <1e-10           : {abs(friedman_p - sm_p) < 1e-10}\n")

    # 2) Skillings-Mack also handles missing observations, where Friedman fails.
    data_missing = data.copy()
    data_missing.iloc[0, 1] = np.nan   # method B missing in block 0
    data_missing.iloc[3, 4] = np.nan   # method E missing in block 3
    res = skillings_mack(*data_missing.transpose().values)
    print("With missing observations (Friedman cannot do this)")
    print(f"  statistic={res.statistic:.6f}  df={res.df}  p-value={res.pvalue:.6f}\n")

    # 3) Passing a 2-D table (treatments x blocks) directly, with a Monte-Carlo
    #    p-value for a small/tied design.
    table = rng.integers(0, 3, size=(4, 8)).astype(float)
    res = skillings_mack(table, simulate_p_value=True, B=5000, random_state=1)
    print("2-D table input + Monte-Carlo p-value")
    print(f"  statistic={res.statistic:.6f}  df={res.df}")
    print(f"  chi-squared p-value={res.pvalue:.6f}  simulated p-value={res.sim_pvalue}")