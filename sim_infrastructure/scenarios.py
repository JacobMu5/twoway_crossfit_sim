"""Simulation settings: each list below is one reproducible experiment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from dgps.reveal_plr import RevealPLRDGP
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


DGP_NAME = RevealPLRDGP().name


def _scenario(
    design: str,
    learner: str,
    n: int,
    reps: int,
    *,
    run: str,
    learner_l: str | None = None,
    sub_exponent: float | None = None,
) -> SimulationScenario:
    learner_label = learner if learner_l is None else f"{learner}-{learner_l}"
    design_label = design if sub_exponent is None else f"{design}_g{sub_exponent:g}"
    params: dict[str, Any] = {"design": design, "learner": learner}
    if learner_l is not None:
        params["learner_l"] = learner_l
    if sub_exponent is not None:
        params["sub_exponent"] = sub_exponent
 
    return SimulationScenario(
        name=f"{run}_{DGP_NAME}_{design_label}_{learner_label}_{n}x{n}",
        dgp=RevealPLRDGP,
        dgp_params={},
        estimator=PLRDMLEstimator,
        estimator_params=params,
        n_rows=n,
        n_cols=n,
        design=design_label,
        learner=learner_label,
        n_simulations=reps,
    )
 
 
def _grid(run, designs, learners, n, reps):
    return [
        _scenario(design, learner, n, reps, run=run, learner_l=learner_l)
        for design, (learner, learner_l) in product(designs, learners)
    ]
 
 
RUNS = {
    "main_results": (
        _grid("main_results", ["no_cf", "as_iid", "multiway", "cluster_oob_sub", "cluster_oob_nodrop"],
              [("lasso", None), ("gbm", None)], 32, 300)
        + _grid("main_results", ["as_iid", "multiway", "cluster_oob_sub", "cluster_oob_nodrop"],
                [("srf", "gbm")], 32, 300)
        + _grid("main_results", ["oracle"], [("gbm", None)], 32, 300)
        + _grid("main_results", ["oracle", "as_iid", "multiway", "cluster_oob_sub"],
                [("gbm", None)], 64, 300)
        + [_scenario("cluster_oob_sub", "gbm", 32, 300, run="main_results", sub_exponent=0.65)]
    ),
    "smoke": (
        _grid("smoke", ["no_cf", "as_iid", "multiway", "cluster_oob_sub", "cluster_oob_nodrop"],
              [("gbm", None), ("lasso", None), ("srf", "gbm")], 12, 2)
        + _grid("smoke", ["oracle"], [("gbm", None)], 12, 2)
    ),
    "p03_exponent_sweep": [
        _scenario("cluster_oob_sub", "gbm", n, 100,
                  run="p03_exponent_sweep", sub_exponent=gamma)
        for gamma in (0.35, 0.45, 0.55, 0.65, 0.8)
        for n in (32, 64)
    ],
}