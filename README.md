# twoway_crossfit_sim

Simulations for my master's thesis on double/debiased machine learning (DML)
in a partially linear model with two-way clustered data. I compare how training
designs (sample selection) affect bias, coverage and leakage through shared
row and column information.

The project structure follows Vladislav Morozov's course
[Fundamentals of Monte Carlo Simulations in Data Science](https://vladislav-morozov.eu/simulations-course/),
particularly the *Good Simulation Code* lectures and the
[minimum-design example](https://github.com/vladislav-morozov/simulations-course/tree/main/example-codes/1-minimum-design-example).

The course explains the simulation pipeline in greater detail, including
*separation of concerns*: keeping data generation, estimation, simulation runs
and experiment settings in separate components, each with a clear role.

## Designs

The comparisons cover no cross-fitting, random cell folds, two-way folds,
honest bagging, no-drop bagging and an oracle benchmark. Honest bagging averages
only bags excluding the evaluated cell's row and column.

For a given learner, base settings are shared across designs; training samples,
averaging and clipping can differ. All designs use the same PLR score and
two-way cluster variance formula.

## Structure

| Location | Contents |
| --- | --- |
| `dgps/` | Data generation; `reveal_plr.py` is used in the current experiments. |
| `estimators/` | Learners, training designs, PLR estimation and separate simulation diagnostics. |
| `sim_infrastructure/` | Experiment settings, runner, parallel/sequential orchestrators and component interfaces. |
| `main.py` | Runs selected simulation campaigns. |
| `results/` | Summary and replication CSV files. |

## Setup and run

From the repository folder, with Python installed:

```bash
python -m pip install -r requirements.txt
python main.py smoke                # small pipeline check
python main.py main_results         # 20 scenarios, 300 replications each
python main.py p03_exponent_sweep    # additional bag-size comparisons
```

The default is `main_results`. Each campaign writes
`results/<campaign>_summary.csv` and `results/<campaign>_records.csv`;
rerunning it replaces those files. Runtime depends on the campaign and hardware.

## Reproducibility

Replication `i` uses data seed `first_seed + i`. The estimator derives its
own seeds from this, so each run can be repeated. Comparable scenarios share
data seeds, allowing the designs to be compared on the same simulated samples.

To reproduce the results, use the same experiment settings and package versions.