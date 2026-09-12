"""Run one simulation set and write its summary and replication CSVs."""

import sys
from pathlib import Path

import pandas as pd
 
from sim_infrastructure.orchestrators import SimulationOrchestrator
from sim_infrastructure.scenarios import RUNS

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"


def main(run_name: str = "main_results") -> None:
    if run_name not in RUNS:
        raise SystemExit(f"unknown run {run_name!r}; choose from {list(RUNS)}")

    orchestrator = SimulationOrchestrator(RUNS[run_name])
    orchestrator.run_all()

    summary = pd.DataFrame(orchestrator.summary_results)
    records = pd.DataFrame(orchestrator.records)
    # Preserve the existing CSV schema without putting `campaign` in scenarios.
    summary.insert(1, "campaign", run_name)
    records.insert(1, "campaign", run_name)
 
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_FOLDER / f"{run_name}_summary.csv", index=False)
    records.to_csv(RESULTS_FOLDER / f"{run_name}_records.csv", index=False)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 1:
        raise SystemExit("usage: python main.py [main_results|smoke|p03_exponent_sweep]")
    main(args[0] if args else "main_results")