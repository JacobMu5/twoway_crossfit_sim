"""Orchestrator: parallel execution with resume support."""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from typing import TYPE_CHECKING

from src.orchestration.runner import run_single_simulation

if TYPE_CHECKING:
    from src.scenarios import ScenarioConfig


def run_all_scenarios(
    scenarios: list["ScenarioConfig"],
    results_path: str = "results/final_results.csv",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run simulations for each scenario in parallel with resume support.

    Args:
        scenarios: List of ScenarioConfigs to execute.
        results_path: Path to CSV file for checkpointing.
        n_jobs: Parallel jobs (-1 = all cores).

    Returns:
        DataFrame with all results.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list = []
    completed_sims: set = set()

    # Resume support
    if results_path.exists():
        try:
            existing_df = pd.read_csv(results_path)
            all_results = existing_df.to_dict("records")
            for _, row in existing_df.iterrows():
                completed_sims.add((row["scenario"], int(row["sim_id"])))
            print(f"Resuming execution: Found {len(all_results)} completed simulations.")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting from scratch.")

    total_scenarios = len(scenarios)
    pbar = tqdm(scenarios, desc="Scenarios", total=total_scenarios, unit="scen")

    for sc_idx, config in enumerate(pbar, 1):
        sims_to_run = [
            i for i in range(config.n_simulations)
            if (config.name, i) not in completed_sims
        ]

        if not sims_to_run:
            pbar.write(f"[{sc_idx}/{total_scenarios}] {config.name} — already done, skipping.")
            continue

        pbar.set_postfix_str(config.name)

        # Run simulations without inner tqdm to keep terminal clean
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_simulation)(
                config=config,
                seed=config.first_seed + i,
                sim_id=i,
            )
            for i in sims_to_run
        )

        all_results.extend(results)
        pd.DataFrame(all_results).to_csv(results_path, index=False)

        sc_df = pd.DataFrame(results)
        if "bias" in sc_df.columns and not sc_df["bias"].isna().all():
            bias_mean = sc_df['bias'].mean()
            bias_std = sc_df['bias'].std()
            rmse = np.sqrt(np.mean(sc_df['bias'] ** 2))

            cov_col = 'covers' if 'covers' in sc_df.columns else 'coverage'
            cov = sc_df[cov_col].mean() if cov_col in sc_df.columns else float('nan')

            pbar.write(f"[{sc_idx}/{total_scenarios}] {config.name}  ->  Bias={bias_mean:+.4f} (+/-{bias_std:.4f})  RMSE={rmse:.4f}  Cov={cov:.0%}")
        else:
            pbar.write(f"[{sc_idx}/{total_scenarios}] {config.name}  ->  All reps failed or no bias column.")

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"DONE — {len(final_df)} total results saved to {results_path}")
    print(f"{'='*60}")

    return final_df
