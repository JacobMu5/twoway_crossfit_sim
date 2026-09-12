"""Run simulation scenarios, optionally in parallel."""

from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from sim_infrastructure.runner import SimulationRunner
from sim_infrastructure.scenarios import SimulationScenario


def _run_scenario(scenario: SimulationScenario) -> tuple[dict, list[dict]]: 
    dgp = scenario.dgp(**scenario.dgp_params)
    estimator = scenario.estimator(**scenario.estimator_params)
    runner = SimulationRunner(dgp, estimator)
    runner.simulate(
        scenario.n_simulations,
        scenario.n_rows,
        scenario.n_cols,
        scenario.first_seed,
    )

    labels = {
        "scenario": scenario.name,
        "dgp": dgp.name,
        "design": scenario.design,
        "learner": scenario.learner,
        "n_rows": scenario.n_rows,
        "n_cols": scenario.n_cols,
        "first_seed": scenario.first_seed,
    }
    return (
        {**labels, **runner.summarize_results()},
        [{**labels, **record} for record in runner.records],
    )


class SimulationOrchestrator:
    def __init__(self, scenarios: list[SimulationScenario]) -> None:
        self.scenarios = scenarios
        self.summary_results: list[dict] = []
        self.records: list[dict] = []

    def run_all(self, max_workers: int | None = None) -> None:
        if max_workers == 1:
            results = list(tqdm(map(_run_scenario, self.scenarios),
                                total=len(self.scenarios), desc="Running simulations"))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(_run_scenario, self.scenarios),
                                    total=len(self.scenarios), desc="Running simulations"))

        self.summary_results = [summary for summary, _ in results]
        self.records = [record for _, records in results for record in records]
