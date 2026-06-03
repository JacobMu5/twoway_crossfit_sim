"""Simulation protocol.

This module defines the Protocol contracts for data-generating processes (DGPs) 
and estimators, ensuring compatibility across the simulation framework. 
Any callable matching these type signatures implements the protocol implicitly, 
allowing the runner to use different generators or estimators without code changes.

The automated runner executes these protocols as pure keyword-only functions:
    df, tau_true = dgp_func(**dgp_kwargs, seed=seed)
    result_dict  = estimator_func(df=df, **estimator_kwargs, seed=seed, tau_true=tau_true)

Protocols:
    DGPProtocol: Definitive interface for data-generating processes.
    EstimatorProtocol: Definitive interface for causal estimators.

Credits:
    Special thanks to Vladislav Morozov for the core structural design.
"""

from typing import Any, Protocol


class DGPProtocol(Protocol):
    """Protocol for Data Generating Processes.

    A DGP is any callable that accepts keyword arguments (forwarded from
    ScenarioConfig.dgp_kwargs) plus a seed, and returns (data, true_effect).
    """

    def __call__(self, *, seed: int, **kwargs: Any) -> tuple[Any, float]: ...


class EstimatorProtocol(Protocol):
    """Protocol for causal estimators.

    An estimator is any callable that accepts data (from a DGP), keyword
    arguments (forwarded from ScenarioConfig.estimator_kwargs), a seed,
    and the true treatment effect, and returns a result dict.
    """

    def __call__(
        self, *, df: Any, seed: int, tau_true: float, **kwargs: Any
    ) -> dict[str, Any]: ...
