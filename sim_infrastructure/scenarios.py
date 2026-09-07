"""
Module for defining simulation scenarios.

A scenario is one full simulation setting: the DGP and its parameters,
the estimator and its parameters, the grid, and the number of
replications. Scenarios are frozen data classes built as Cartesian
products of component lists, grouped into campaigns. Every campaign
runs on the confounded RevealPLRDGP (X1/X2 are the exact cluster
labels; Cov(g0, m0) = +0.31, so OLS is inconsistent); the superseded DGPs,
including the diffuse TwoWayPLRDGP robustness arm, are kept 
in dgps.legacy_plr:

    main_results  one run for both paper tables: the design grid at
                  32x32 (fold designs, bagged sub-cluster designs, the
                  mixed-learner arm) plus the anatomy rows at 64x64.
    smoke         tiny version; runs in a few minutes.

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

# Future-research gamma sweep: bag blocks are ceil(N**gamma) clusters a
# side, so gamma trades per-bag fit quality against the number of
# honest (omit-both) bags.
P03_GAMMAS = [0.35, 0.45, 0.55, 0.65, 0.8]


def _main_results_scenarios(
    dgp: type[DGPProtocol], dgp_params: dict[str, Any]
) -> list[SimulationScenario]:
    """The main_results campaign for one DGP, as five named blocks."""
    comparison_32x32 = _plr_campaign(
        "main_results", dgp, dgp_params,
        P01_DESIGNS,
        [("lasso", None), ("gbm", None)], [(32, 32)],
        n_simulations=300, first_seed=1,
    )
    mixed_arm = _plr_campaign(
        "main_results", dgp, dgp_params,
        P01_MIXED_DESIGNS,
        [("srf", "gbm")], [(32, 32)],
        n_simulations=300, first_seed=1,
    )
    oracle = _plr_campaign(
        "main_results", dgp, dgp_params,
        ["oracle"],
        [("gbm", None)], [(32, 32)],
        n_simulations=300, first_seed=1,
    )
    anatomy_64x64 = _plr_campaign(
        "main_results", dgp, dgp_params,
        P02_DESIGNS,
        [("gbm", None)], [(64, 64)],
        n_simulations=300, first_seed=1,
    )
    # sweep comparison: the honest design at a larger subsampling exponent
    larger_bag = [SimulationScenario(
        name=(f"main_results_{dgp(**dgp_params).name}"
              f"_cluster_oob_sub_g0.65_gbm_32x32"),
        campaign="main_results",
        dgp=dgp,
        dgp_params=dict(dgp_params),
        estimator=PLRDMLEstimator,
        estimator_params={"design": "cluster_oob_sub",
                          "learner": "gbm", "sub_exponent": 0.65},
        n_rows=32,
        n_cols=32,
        design="cluster_oob_sub_g0.65",
        learner="gbm",
        n_simulations=300,
        first_seed=1,
    )]
    return (comparison_32x32 + mixed_arm + oracle
            + anatomy_64x64 + larger_bag)


CAMPAIGNS: dict[str, list[SimulationScenario]] = {
    # One run, both paper tables: the 32x32 block feeds the design
    # table, the gbm rows at both grids feed the anatomy table. All
    # cells share one seed stream (common random numbers throughout).
    "main_results": [
        scenario
        for dgp, dgp_params in DGPS
        for scenario in _main_results_scenarios(dgp, dgp_params)
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
                n_simulations=2, first_seed=1,
            )
        )
        ],
        # Future-research exhibit: sweep the subsampling exponent gamma of
        # the honest bagged design, headline arm only (gbm, reveal DGP).
        # Common seeds across gamma make the curves smooth in gamma.
        # Run on demand: python main.py p03_exponent_sweep
        "p03_exponent_sweep": [
            SimulationScenario(
                name=(f"p03_exponent_sweep_{RevealPLRDGP().name}"
                    f"_cluster_oob_sub_g{gamma:g}_gbm_{n}x{n}"),
                campaign="p03_exponent_sweep",
                dgp=RevealPLRDGP,
                dgp_params={},
                estimator=PLRDMLEstimator,
                estimator_params={"design": "cluster_oob_sub",
                                "learner": "gbm", "sub_exponent": gamma},
                n_rows=n,
                n_cols=n,
                design=f"cluster_oob_sub_g{gamma:g}",
                learner="gbm",
                n_simulations=100,
                first_seed=1,
            )
            for gamma in P03_GAMMAS
            for n in (32, 64)
        ],
        
}

# Default run: the paper campaigns (the smoke test and the p03 sweep
# run on demand only)
scenarios: list[SimulationScenario] = [
    scenario
    for campaign, campaign_scenarios in CAMPAIGNS.items()
    if campaign not in ("smoke", "p03_exponent_sweep")
    for scenario in campaign_scenarios
]
