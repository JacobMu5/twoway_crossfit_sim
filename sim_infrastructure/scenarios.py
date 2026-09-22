"""Simulation settings: each list below is one reproducible experiment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log
from typing import Any

from dgps.reveal_plr import RevealPLRDGP, SimpleSignalPLRDGP
from dgps.pliv import ChenChiangPLIV, PackagePLIV
from estimators.pliv import PLIVDMLEstimator
from estimators.plr import PLRDMLEstimator
from sim_infrastructure.protocols import DGPProtocol, EstimatorProtocol


@dataclass(frozen=True)
class SimulationScenario:
    name: str
    dgp: type[DGPProtocol]
    dgp_params: dict[str, Any]
    estimator: type[EstimatorProtocol]
    estimator_params: dict[str, Any]
    n_rows: int
    n_cols: int
    design: str
    learner: str
    n_simulations: int
    first_seed: int = 1



def _scenario(
    design: str,
    learner: str,
    n: int,
    reps: int,
    *,
    run: str,
    learner_l: str | None = None,
    sub_exponent: float | None = None,
    dgp: type[DGPProtocol] = RevealPLRDGP,
    dgp_params: dict[str, Any] | None = None,
    estimator: type[EstimatorProtocol] = PLRDMLEstimator,
    first_seed: int = 1,
    n_cols: int | None = None,
) -> SimulationScenario:
    dgp_params = dgp_params or {}
    n_cols = n if n_cols is None else n_cols
    learner_label = learner if learner_l is None else f"{learner}-{learner_l}"
    design_label = design if sub_exponent is None else f"{design}_g{sub_exponent:g}"
    params: dict[str, Any] = {"design": design, "learner": learner}
    if learner_l is not None:
        params["learner_l"] = learner_l
    if sub_exponent is not None:
        params["sub_exponent"] = sub_exponent
 
    return SimulationScenario(
        name=f"{run}_{dgp(**dgp_params).name}_{design_label}_{learner_label}_{n}x{n_cols}",
        dgp=dgp,
        dgp_params=dgp_params,
        estimator=estimator,
        estimator_params=params,
        n_rows=n,
        n_cols=n_cols,
        design=design_label,
        learner=learner_label,
        n_simulations=reps,
        first_seed=first_seed,
    )
 
 
def _grid(run, designs, learners, n, reps, **kwargs):
    return [
        _scenario(design, learner, n, reps, run=run, learner_l=learner_l, **kwargs)
        for design, (learner, learner_l) in product(designs, learners)
    ]
 
 
RUNS = {
    "smoke": (
        _grid("smoke", ["no_cf", "as_iid", "multiway", "cluster_oob_sub", "cluster_oob_nodrop"],
              [("gbm", None), ("lasso", None), ("srf", "gbm")], 12, 2)
        + _grid("smoke", ["oracle"], [("gbm", None)], 12, 2)
    ),
}
 
PAPER_REPS = 2000
PAPER_SEED_OFFSET = 2000000
_DESIGNS = ['oracle', 'no_cf', 'as_iid', 'multiway', 'cluster_oob_sub', 'cluster_oob_nodrop']
_PLR_VARIANCES = {'resid_share': .9, 'eps_resid_share': .1}
 
 
def _signal_run(run, signal, seed, linear_controls=False, n=32, dgp_params=None):
    """One PLR family, two explicit signals; identical learners and bag counts."""
    kw = dict(dgp=SimpleSignalPLRDGP, dgp_params={**_PLR_VARIANCES, 'signal': signal, **(dgp_params or {})}, first_seed=seed + PAPER_SEED_OFFSET)
    return (_grid(run, _DESIGNS, [('gbm', None)], n, PAPER_REPS, **kw)
            + (_grid(run, ['no_cf', 'multiway'], [('lasso', None)], n, PAPER_REPS, **kw)
               if linear_controls else [])
            + [_scenario('cluster_oob_sub', 'gbm', n, PAPER_REPS, run=run,
                         sub_exponent=log(k-1e-7)/log(n), **kw) for k in (n//4, n//2)])
 
 
RUNS.update({
    'lead_plr': (
        _signal_run('lead_plr', 'threshold', 3410001, linear_controls=True, n=64,
                    dgp_params={'resid_share': .9, 'eps_resid_share': .1})
        + _grid('lead_plr', ['multiway', 'cluster_oob_sub'], [('gbm', None)], 64, PAPER_REPS,
                dgp=SimpleSignalPLRDGP, first_seed=3410001 + PAPER_SEED_OFFSET,
                dgp_params={'resid_share': .9, 'eps_resid_share': .1,
                            'include_signatures': False})),
    'linear_plr': _signal_run('linear_plr', 'linear', 987001),
    # Paper equations, explicitly calibrated; retain the five archived designs.
    'pliv': [sc for mu in (0., 1.) for sc in
             _grid('pliv', _DESIGNS[:5], [('gbm', None)], 25, PAPER_REPS,
                   dgp=ChenChiangPLIV, dgp_params={'x2_row_mean': mu},
                   estimator=PLIVDMLEstimator, first_seed=940001 + PAPER_SEED_OFFSET)],
    # Externally supplied package data; original two instruments, scalar z1 score.
    'package_pliv': (
        _grid('package_pliv', _DESIGNS[:5], [('gbm', None)], 40, PAPER_REPS,
              n_cols=35, dgp=PackagePLIV, estimator=PLIVDMLEstimator, first_seed=986001 + PAPER_SEED_OFFSET)
        + [_scenario('cluster_oob_sub', 'gbm', 40, PAPER_REPS, run='package_pliv',
                     n_cols=35, sub_exponent=.8, dgp=PackagePLIV,
                     estimator=PLIVDMLEstimator, first_seed=986001 + PAPER_SEED_OFFSET)]),
    'fewclusters': (
        _grid('fewclusters', ['oracle', 'as_iid', 'multiway', 'cluster_oob_sub'],
              [('gbm', None)], 200, PAPER_REPS, n_cols=8, dgp=SimpleSignalPLRDGP, dgp_params=_PLR_VARIANCES, first_seed=990001 + PAPER_SEED_OFFSET)
        + [_scenario('cluster_oob_sub', 'gbm', 200, PAPER_REPS, run='fewclusters',
                     n_cols=8, sub_exponent=.8, dgp=SimpleSignalPLRDGP,
                     dgp_params=_PLR_VARIANCES, first_seed=990001 + PAPER_SEED_OFFSET)]),
    'lead_exponent_sweep': [
        _scenario('cluster_oob_sub', 'gbm', n, PAPER_REPS, run='lead_exponent_sweep',
                  sub_exponent=gamma, dgp=SimpleSignalPLRDGP,
                  dgp_params=_PLR_VARIANCES, first_seed=991001 + PAPER_SEED_OFFSET)
        for gamma in (.35, .45, .55, .65, .8) for n in (32, 64)],
})
 
PAPER_RUNS = ('lead_plr', 'linear_plr', 'pliv', 'package_pliv',
              'fewclusters', 'lead_exponent_sweep')