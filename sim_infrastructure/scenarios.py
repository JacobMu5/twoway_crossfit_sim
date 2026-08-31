"""
Module for defining simulation scenarios.

A scenario is one full simulation setting: the DGP and its parameters,
the estimator and its parameters, the grid, and the number of
replications. Scenarios are frozen data classes built as Cartesian
products of component lists, grouped into campaigns. Every campaign
runs on the concentrated RevealPLRDGP (X1/X2 are the exact cluster
labels); the superseded DGPs, including the diffuse TwoWayPLRDGP
robustness arm, are kept in dgps.legacy_plr:

    p01_design_comparison  design grid (fold designs plus the bagged
                           sub-cluster designs), plus the mixed-learner
                           arm.
    p02_plr_anatomy        error decomposition across designs.
    smoke                  tiny versions of both; runs in a few minutes.

Scenarios that should be comparable share first_seed and DGP parameters,
so replication sim_id sees identical data in every cell (common random
numbers).

Classes:
    SimulationScenario: data class for scenarios.

Variables:
    CAMPAIGNS (dict): campaign name -> list of scenarios.
    scenarios (list): default scenarios to run (everything except smoke).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from dgps.reveal_plr import RevealPLRDGP
from estimators.plr import PLRDMLEstimator
from sim_infrastructure.protocols import DGPProtocol, EstimatorProtocol


@dataclass(frozen=True)
class SimulationScenario:
    """A single simulation scenario: DGP, estimator, grid, replications."""

    name: str
    campaign: str
    dgp: type[DGPProtocol]
    dgp_params: dict[str, Any]
    estimator: type[EstimatorProtocol]
    estimator_params: dict[str, Any]
    n_rows: int
    n_cols: int
    design: str
    learner: str
    n_simulations: int = 200
    first_seed: int = 1


def _plr_campaign(
    campaign: str,
    dgp: type[DGPProtocol],
    dgp_params: dict[str, Any],
    designs: list[str],
    learners: list[tuple[str, str | None]],
    grids: list[tuple[int, int]],
    n_simulations: int,
    first_seed: int,
) -> list[SimulationScenario]:
    """Cartesian PLR campaign over designs x learners x grids.

    Each learners entry is a (learner, learner_l) pair: the
    treatment-nuisance learner and, optionally, a DIFFERENT
    outcome-nuisance learner (None = same learner for both).
    ("srf", "gbm") is the mixed-learner arm.
    """
    dgp_label = dgp(**dgp_params).name
    out = []
    for design, (learner, learner_l), (n_rows, n_cols) in product(
        designs, learners, grids
    ):
        label = learner if learner_l is None else f"{learner}-{learner_l}"
        params: dict[str, Any] = {"design": design, "learner": learner}
        if learner_l is not None:
            params["learner_l"] = learner_l
        out.append(SimulationScenario(
            name=f"{campaign}_{dgp_label}_{design}_{label}_{n_rows}x{n_cols}",
            campaign=campaign,
            dgp=dgp,
            dgp_params=dict(dgp_params),
            estimator=PLRDMLEstimator,
            estimator_params=params,
            n_rows=n_rows,
            n_cols=n_cols,
            design=design,
            learner=label,
            n_simulations=n_simulations,
            first_seed=first_seed,
        ))
    return out


TWO_WAY_DGP_PARAMS: dict[str, Any] = {
    "theta0": 1.0, "cov_cluster": 1.5, "sd_eps": 1.5, "sd_v": 1.5,
}

# Now Campaigns run on the concentrated reveal DGP only (the paper DGP).
# To re-enable the diffuse robustness arm, import TwoWayPLRDGP from
# dgps.legacy_plr and put it back in this list.
DGPS: list[tuple[type[DGPProtocol], dict[str, Any]]] = [
    (RevealPLRDGP, {}),
    # (TwoWayPLRDGP, TWO_WAY_DGP_PARAMS),
]

P01_DESIGNS = ["no_cf", "as_iid", "multiway",
               "cluster_oob_sub", "cluster_oob_nodrop"]
P01_MIXED_DESIGNS = ["as_iid", "multiway",
                     "cluster_oob_sub", "cluster_oob_nodrop"]
P02_DESIGNS = ["oracle", "as_iid", "multiway", "cluster_oob_sub"]

CAMPAIGNS: dict[str, list[SimulationScenario]] = {
    "p01_design_comparison": [
        scenario
        for dgp, dgp_params in DGPS
        for scenario in (
            _plr_campaign(
                "p01_design_comparison",
                dgp, dgp_params,
                P01_DESIGNS,
                [("lasso", None), ("gbm", None)], [(32, 32)],
                n_simulations=200, first_seed=1,
            )
            + _plr_campaign(
                "p01_design_comparison",
                dgp, dgp_params,
                P01_MIXED_DESIGNS,
                [("srf", "gbm")], [(32, 32)],
                n_simulations=200, first_seed=1,
            )
        )
    ],
    "p02_plr_anatomy": [
        scenario
        for dgp, dgp_params in DGPS
        for scenario in _plr_campaign(
            "p02_plr_anatomy",
            dgp, dgp_params,
            P02_DESIGNS,
            [("gbm", None)], [(32, 32), (64, 64)],
            n_simulations=400, first_seed=90_210,
         )
    ],
    # The oracle-only second block avoids duplicating the as_iid and
    # multiway cells the first block already smoke-tests.
    "smoke": [
        scenario
        for dgp, dgp_params in DGPS
        for scenario in (
            _plr_campaign(
                "smoke",
                dgp, dgp_params,
                P01_DESIGNS,
                [("gbm", None), ("lasso", None), ("srf", "gbm")], [(12, 12)],
                n_simulations=2, first_seed=1,
            )
            + _plr_campaign(
                "smoke",
                dgp, dgp_params,
                ["oracle"],
                [("gbm", None)], [(12, 12)],
                n_simulations=2, first_seed=90_210,
            )
        )
        ],
}

# Default run: every campaign except the smoke test
scenarios: list[SimulationScenario] = [
    scenario
    for campaign, campaign_scenarios in CAMPAIGNS.items()
    if campaign != "smoke"
    for scenario in campaign_scenarios
]
