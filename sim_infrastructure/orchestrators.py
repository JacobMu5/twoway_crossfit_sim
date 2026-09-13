"""Run simulation scenarios, optionally in parallel."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from threadpoolctl import threadpool_limits
from tqdm import tqdm

from sim_infrastructure.runner import SimulationRunner
from sim_infrastructure.scenarios import SimulationScenario

CHUNK = 10  # replications per work unit, so every scenario is shared by all workers

def _run_chunk(chunk: SimulationScenario) -> list[dict]:
    runner = SimulationRunner(chunk.dgp(**chunk.dgp_params), chunk.estimator(**chunk.estimator_params))
    with threadpool_limits(limits=1):  # one BLAS/OpenMP thread per worker, as the saved results were made
        runner.simulate(chunk.n_simulations, chunk.n_rows, chunk.n_cols, chunk.first_seed)
    return runner.records
 
 
def _run_scenario(scenario: SimulationScenario, chunk_records: list[list[dict]]) -> tuple[dict, list[dict]]: 
    dgp = scenario.dgp(**scenario.dgp_params)
    estimator = scenario.estimator(**scenario.estimator_params)
    runner = SimulationRunner(dgp, estimator)
    records = [record for records in chunk_records for record in records]
    runner.records = [{**record, "sim_id": sim_id} for sim_id, record in enumerate(records)]

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
        chunks = [replace(sc, first_seed=sc.first_seed + a, n_simulations=min(CHUNK, sc.n_simulations - a))
                  for sc in self.scenarios for a in range(0, sc.n_simulations, CHUNK)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending = iter(list(tqdm(executor.map(_run_chunk, chunks),
                                     total=len(chunks), desc="Running simulations")))
        results = [_run_scenario(sc, [next(pending) for _ in range(0, sc.n_simulations, CHUNK)])
                   for sc in self.scenarios]

        self.summary_results = [summary for summary, _ in results]
        self.records = [record for _, records in results for record in records]
