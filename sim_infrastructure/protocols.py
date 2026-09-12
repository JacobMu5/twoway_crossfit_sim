"""Interfaces used by the simulation runner."""

from typing import Protocol


class DGPProtocol(Protocol):
    def sample(self, n_rows: int, n_cols: int, seed: int | None = None): ...

    @property
    def true_theta(self) -> float: ...

    @property
    def name(self) -> str: ...


class EstimatorProtocol(Protocol):
    def fit(self, sample, seed: int | None = None) -> None: ...

    @property
    def theta_hat(self) -> float: ...

    @property
    def se_hat(self) -> float: ...

    @property
    def diagnostics(self) -> dict[str, float]: ...
