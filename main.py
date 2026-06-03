"""Entry point for Clustered RDD simulations.

Usage:
    python main.py

Output:
    results/rdd_simulation_results.csv
"""

from src.scenarios import get_rdd_scenarios
from src.orchestration.orchestrator import run_all_scenarios

if __name__ == "__main__":
    scenarios = get_rdd_scenarios(n_sim=200)
    run_all_scenarios(scenarios, results_path="results/rdd_simulation_results.csv")
