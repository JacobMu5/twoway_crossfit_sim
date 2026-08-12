"""
Module for defining simulation protocols.

This module contains Protocol classes that define the interfaces for
data-generating processes (DGPs) and estimators. Any DGP that samples an
N x M grid and knows its true parameter, and any estimator that fits one
sample and reports an estimate, a standard error, and a dict of scalar
diagnostics, can be combined by the runner. The orchestrator checks both
protocols at runtime before anything runs.

Protocols:
    DGPProtocol: Interface for data-generating processes.
    EstimatorProtocol: Interface for estimators.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class DGPProtocol(Protocol):
    def sample(self, n_rows: int, n_cols: int, seed: int | None = None): ...

    @property
    def true_theta(self) -> float: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class EstimatorProtocol(Protocol):
    def fit(self, sample, seed: int | None = None) -> None: ...

    @property
    def theta_hat(self) -> float: ...

    @property
    def se_hat(self) -> float: ...

    @property
    def name(self) -> str: ...

    @property
    def diagnostics(self) -> dict[str, float]: ...
