"""Run one simulation set and write its summary and replication CSVs."""

import sys
from pathlib import Path

import pandas as pd
 
from sim_infrastructure.orchestrators import SimulationOrchestrator
from sim_infrastructure.scenarios import RUNS, PAPER_RUNS

RESULTS_FOLDER = Path(__file__).resolve().parent / "results"


def main(run_name: str = "main_results", out_dir: str | Path = RESULTS_FOLDER) -> None:
    campaigns = PAPER_RUNS if run_name == "paper" else (run_name,)
    unknown = [c for c in campaigns if c not in RUNS]
    if unknown:
        raise SystemExit(f"unknown run {unknown!r}; choose 'paper' or one of {list(RUNS)}")

    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name in campaigns:
        orchestrator = SimulationOrchestrator(RUNS[name])
        orchestrator.run_all()
        for kind, values in (("summary", orchestrator.summary_results),
                             ("records", orchestrator.records)):
            frame = pd.DataFrame(values)
            frame.insert(1, "campaign", name)  # keep `campaign` out of scenarios
            frame.to_csv(folder / f"{name}_{kind}.csv", index=False)
        print(f"saved {name}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 2:
        raise SystemExit("usage: python main.py [run|paper] [output_dir]")
    main(*args) if args else main()