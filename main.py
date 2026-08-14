"""
Entry point for running the cross-fitting design simulations.

Overall goal of the simulations: measure how the cross-fitting design
affects DML estimates when the data are two-way clustered. Each campaign
writes one summary CSV and one records CSV under results/.

Usage:
    python main.py                 # all campaigns (long)
    python main.py smoke           # tiny end-to-end check (under a minute)
    python main.py p01_design_comparison p02_plr_anatomy

Rebuilding the thesis tables from these CSVs is a separate step; see
the export modules under sim_infrastructure/.
"""

import sys
from pathlib import Path

# Anchored to this file, so `python path/to/main.py` works from any cwd
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"


def run_campaigns(requested: list[str]) -> None:
    """Run the requested campaigns and write their summary/records CSVs."""
    import pandas as pd

    from sim_infrastructure.orchestrators import SimulationOrchestratorParallel
    from sim_infrastructure.scenarios import CAMPAIGNS
    from sim_infrastructure.scenarios import scenarios as default_scenarios

    unknown = [c for c in requested if c not in CAMPAIGNS]
    if unknown:
        raise SystemExit(
            f"unknown campaign(s) {unknown}; choose from {list(CAMPAIGNS)}"
        )
    selected = ([s for c in requested for s in CAMPAIGNS[c]]
                if requested else default_scenarios)

    # Create and execute simulations
    orchestrator = SimulationOrchestratorParallel(selected)
    orchestrator.run_all()

    # Write one summary and one records CSV per campaign
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(orchestrator.summary_results)
    records = pd.DataFrame(orchestrator.records)
    for campaign in summary["campaign"].unique():
        summary[summary["campaign"] == campaign].to_csv(
            RESULTS_FOLDER / f"{campaign}_summary.csv", index=False)
        records[records["campaign"] == campaign].to_csv(
            RESULTS_FOLDER / f"{campaign}_records.csv", index=False)


if __name__ == "__main__":
    run_campaigns(sys.argv[1:])