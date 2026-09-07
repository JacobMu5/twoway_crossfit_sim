"""
Module for the simulation-only error decomposition of the PLR estimator.

The feasible estimate and its standard error are formed in
estimators.plr; this module holds the diagnostics that exist only because
the DGP returns the true nuisances. error_decomposition reports the
standard-error variants, the term-by-term bias dissection,
and the Jacobian anatomy from one fit. It reuses the
additive two-way cluster variance already computed for the reported
standard error, so the row and column sums are not formed a second time.

Functions:
    error_decomposition: SE variants and bias/anatomy terms for one fit.
"""

from __future__ import annotations

import math

import numpy as np

from dgps.plr import ClusteredSample


def error_decomposition(
    sample: ClusteredSample,
    l_hat: np.ndarray,
    m_hat: np.ndarray,
    denom: float,
    psi: np.ndarray,
    n: int,
    var_sum_add: float,
) -> dict[str, float]:
    """SE variants, bias dissection (P01), and Jacobian anatomy (P02).

    With a = l_hat - l0 and b = m_hat - m0, the B_* terms are the exact
    error channels of the Robinson estimator (see protocol P02); the
    leak* terms split En(bV) into row, column, and cell parts. var_sum_add
    is the additive two-way cluster variance already formed for the
    reported SE; the cgm variant reuses it rather than re-summing psi.
    """
    var_sum_cgm = max(var_sum_add - float(psi @ psi), 0.0)

    a = l_hat - sample.l0
    b = m_hat - sample.m0
    v = sample.v
    eps = sample.eps
    th0 = float(sample.theta0)
    v_clu = sample.v_row + sample.v_col
    c = psi - psi.mean()
    se_iid = float(c.std(ddof=1) / math.sqrt(n) / denom)

    return {
        "se_hat_add": math.sqrt(var_sum_add) / (n * denom),
        "se_hat_cgm": math.sqrt(var_sum_cgm) / (n * denom),
        "se_hat_iid": se_iid,
        "denom": denom,
        "EnV2": float(np.mean(v * v)),
        "Eb2": float(np.mean(b * b)),
        "Ea2": float(np.mean(a * a)),
        "B_ab": float(np.mean(a * b)),
        "B_bb": float(-th0 * np.mean(b * b)),
        "B_bV": float(th0 * np.mean(b * v)),
        "B_aV": float(-np.mean(a * v)),
        "B_eps": float(np.mean(eps * v) - np.mean(eps * b)),
        "leak": float(np.mean(b * v_clu)),
        "leak_row": float(np.mean(b * sample.v_row)),
        "leak_col": float(np.mean(b * sample.v_col)),
        "leak_own": float(np.mean(b * sample.v_cell)),
        "coupling": float(
            np.mean((a - th0 * b) ** 2) / max(np.mean(b * b), 1e-12)
        ),
    }
