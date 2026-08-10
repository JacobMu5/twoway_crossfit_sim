"""
Module for the base-learner registry used in nuisance estimation.

Each learner is a factory that takes an integer seed and returns a
fresh scikit-learn estimator, so all random parts of estimation are
reproducible from the Monte Carlo seed. lasso is the inflexible
contrast, gbm the flexible leak exhibitor, and srf the treatment-side
learner of the mixed-learner arm. Adding a learner is a one-entry
change here.

Classes:
    LearnerSpec: holds the per-fold learner factory.

Variables:
    LEARNERS (dict): registry of learner specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Lasso


@dataclass(frozen=True)
class LearnerSpec:
    """A base-learner specification.

    Attributes:
        fold (Callable[[int], object]): factory for the per-fold learner.
    """

    fold: Callable[[int], object]

LEARNERS: dict[str, LearnerSpec] = {
    "lasso": LearnerSpec(
        fold=lambda s: Lasso(alpha=0.02, max_iter=3000),
    ),
    "gbm": LearnerSpec(
        fold=lambda s: HistGradientBoostingRegressor(
            max_iter=110, max_leaf_nodes=15, learning_rate=0.1, random_state=s,
        ),
    ),
    "srf": LearnerSpec(
        fold=lambda s: ExtraTreesRegressor(
            n_estimators=80, max_features=1, min_samples_leaf=5,
            n_jobs=1, random_state=s,
        ),
    ),
}
