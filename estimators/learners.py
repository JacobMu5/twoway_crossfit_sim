"""
Module for the base-learner registry used in nuisance estimation.

Each learner is a LearnerSpec with one factories and a default bag
count. The "make" factory builds the base learner used everywhere: fit once by
+the one-fit-per-fold designs (no-CF, as-IID, multiway) and bagged inside
+the sub-sampled cluster-OOB design. A factory takes an integer seed and
+returns a fresh scikit-learn estimator, so all random parts of estimation
+are reproducible from the Monte Carlo seed. lasso is the inflexible
contrast, gbm the flexible leak exhibitor, and srf the treatment-side
learner of the mixed-learner arm. Adding a learner is a one-entry
change here.

Classes:
    LearnerSpec: holds the learner factory and the default bag count.

Variables:
    LEARNERS (dict): registry of learner specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class LearnerSpec:
    """A base-learner specification.

    Attributes:
        make (Callable[[int], object]): factory for the base learner.
        n_bags (int): default number of subsampled bag draws B.
    """

    make: Callable[[int], object]
    n_bags: int

LEARNERS: dict[str, LearnerSpec] = {
    "lasso": LearnerSpec(
        make=lambda s: Lasso(alpha=0.02, max_iter=3000),
        n_bags=150,
    ),
    "gbm": LearnerSpec(
        make=lambda s: HistGradientBoostingRegressor(
            max_iter=35, max_leaf_nodes=8, learning_rate=0.18,
            min_samples_leaf=5, random_state=s,
        ),
        n_bags=130,
    ),
    "srf": LearnerSpec(
        make=lambda s: DecisionTreeRegressor(
            splitter="random", max_features=1, min_samples_leaf=5,
            random_state=s,
        ),
        n_bags=250,
    ),
}
