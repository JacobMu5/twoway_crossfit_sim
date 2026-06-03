"""Simulation runner: executes a single (config, seed, sim_id) → result dict."""

import pandas as pd

from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from src.scenarios import ScenarioConfig


def run_single_simulation(
    config: "ScenarioConfig", seed: int, sim_id: int
) -> dict[str, Any]:
    """Execute a single simulation replication.

    Args:
        config: Scenario configuration.
        seed: Random seed for this replication.
        sim_id: Replication index.

    Returns:
        Dict with scenario metadata and estimator results.
    """
    import warnings
    warnings.filterwarnings("ignore")
    try:
        df, tau_true = config.dgp_func(**config.dgp_kwargs, seed=seed)
        result_dict = config.estimator_func(
            df=df, **config.estimator_kwargs, seed=seed, tau_true=tau_true
        )

        out: dict[str, Any] = {
            'scenario': config.name,
            'sim_id': sim_id,
            'seed': seed,
            'n': len(df) if isinstance(df, pd.DataFrame) else (len(df.get('Y', [])) if isinstance(df, dict) else -1),
            'tau_true': tau_true,
        }
        out.update(config.metadata)
        out.update(result_dict)

        if "bias" not in out and "tau_hat" in out:
            out["bias"] = out["tau_hat"] - tau_true

        if "covers" not in out and "ci_lower" in out and "ci_upper" in out:
            out["covers"] = int(out["ci_lower"] <= tau_true <= out["ci_upper"])

        return out

    except Exception as e:
        out = {
            'scenario': config.name,
            'sim_id': sim_id,
            'seed': seed,
            'error': str(e),
            'tau_hat': float('nan'),
            'bias': float('nan'),
            'covers': float('nan'),
            'se': float('nan'),
        }
        out.update(config.metadata)
        return out
