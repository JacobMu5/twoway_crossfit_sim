"""
Module for executing many Monte Carlo scenarios.

This module contains orchestrator classes that run a list of
SimulationScenario objects. Every scenario's DGP and estimator are
checked against the protocols before anything runs, so a bad component
fails up front. Results are stored in scenario order.

Classes:
    SimulationOrchestratorParallel: runs scenarios in parallel processes.
    SimulationOrchestratorSequential: runs scenarios one after another.
"""

from __future__ import annotations

import os
# We already run one scenario per core. If each process ALSO lets NumPy
# use all cores, you oversubscribe e.g. on an 8-core machine that's
# 8x8 = 64 threads fighting over 8 cores making them block each other.
# This solves this issue and lets the code run way faster!
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from sim_infrastructure.protocols import DGPProtocol, EstimatorProtocol
from sim_infrastructure.runner import SimulationRunner
from sim_infrastructure.scenarios import SimulationScenario


def _check_scenario(scenario: SimulationScenario) -> None:
    """Instantiate the scenario's components and enforce the protocols.

    Raises:
        TypeError: if the DGP or estimator does not satisfy its protocol.
    """
    dgp = scenario.dgp(**scenario.dgp_params)
    estimator = scenario.estimator(**scenario.estimator_params)
    if not isinstance(dgp, DGPProtocol):
        raise TypeError(f"{type(dgp).__name__} does not satisfy DGPProtocol")
    if not isinstance(estimator, EstimatorProtocol):
        raise TypeError(
            f"{type(estimator).__name__} does not satisfy EstimatorProtocol"
        )


def _run_single_scenario(scenario: SimulationScenario) -> tuple[dict, list[dict]]:
    """Run a single scenario and return its summary and records.

    Args:
        scenario (SimulationScenario): the scenario to run.

    Returns:
        tuple[dict, list[dict]]: labelled summary row and labelled
        per-replication records.
    """
    dgp = scenario.dgp(**scenario.dgp_params)
    estimator = scenario.estimator(**scenario.estimator_params)

    runner = SimulationRunner(dgp, estimator)
    runner.simulate(
        n_sim=scenario.n_simulations,
        n_rows=scenario.n_rows,
        n_cols=scenario.n_cols,
        first_seed=scenario.first_seed,
    )

    labels = {
        "scenario": scenario.name,
        "campaign": scenario.campaign,
        "dgp": dgp.name,
        "design": scenario.design,
        "learner": scenario.learner,
        "n_rows": scenario.n_rows,
        "n_cols": scenario.n_cols,
        "first_seed": scenario.first_seed,
    }
    summary = {**labels, **runner.summarize_results()}
    records = [{**labels, **record} for record in runner.records]
    return summary, records


class SimulationOrchestratorParallel:
    """Process-parallel simulation orchestrator.

    Attributes:
        scenarios (list[SimulationScenario]): scenarios to run.
        summary_results (list[dict]): one summary row per scenario, in
            scenario order.
        records (list[dict]): per-replication records of every scenario.
    """

    def __init__(self, scenarios: list[SimulationScenario]) -> None:
        self.scenarios = scenarios
        self.summary_results: list[dict] = []
        self.records: list[dict] = []

    def run_all(self, max_workers: int | None = None) -> None:
        """Run all scenarios with process-based parallelism.

        Args:
            max_workers (int | None): number of worker processes.
                Defaults to None (executor default: CPU count).
        """
        # Fail loudly on any malformed scenario before burning compute
        for scenario in self.scenarios:
            _check_scenario(scenario)

        # Submit every scenario, remembering each future's position, so
        # results land in scenario order -- not completion order
        results: list = [None] * len(self.scenarios)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_scenario, scenario): position
                for position, scenario in enumerate(self.scenarios)
            }
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Running simulations"):
                results[futures[future]] = future.result()

        for summary, records in results:
            self.summary_results.append(summary)
            self.records.extend(records)


class SimulationOrchestratorSequential:
    """Sequential simulation orchestrator (debugging and profiling).

    Attributes:
        scenarios (list[SimulationScenario]): scenarios to run.
        summary_results (list[dict]): one summary row per scenario, in
            scenario order.
        records (list[dict]): per-replication records of every scenario.
    """

    def __init__(self, scenarios: list[SimulationScenario]) -> None:
        self.scenarios = scenarios
        self.summary_results: list[dict] = []
        self.records: list[dict] = []

    def run_all(self) -> None:
        """Run all scenarios one after another in scenario order."""
        for scenario in self.scenarios:
            _check_scenario(scenario)
        for scenario in tqdm(self.scenarios, desc="Running simulations"):
            summary, records = _run_single_scenario(scenario)
            self.summary_results.append(summary)
            self.records.extend(records)
