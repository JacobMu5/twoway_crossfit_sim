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
    | `sim_infrastructure/` | Experiment settings, runner, one orchestrator and component interfaces. |
    | `reporting/` | Builds the LaTeX tables and figures from the saved CSVs. |
    | `main.py` | Selects a scenario set, runs it and writes the CSVs. |
    | `results/` | Summary and replication CSV files. |

     
```text
main.py <run>
   │
   ▼
sim_infrastructure/   scenarios.py → orchestrators.py → runner.py
                      run spec       one process per      R replications,
                                     scenario             seed = first_seed + i
   │
   ▼
dgps/reveal_plr.py    ClusteredSample: (Y, D, X, rows, cols) + true l0, m0
   │
   ▼
estimators/plr.py     designs.py  training design (folds / bags / oracle)
                      learners.py base learner (lasso / gbm / srf)
                      → l_hat, m_hat → PLR estimate → two-way clustered SE
                      diagnostics.py  error decomposition vs oracle
   │
   ▼
results/<run>_summary.csv, results/<run>_records.csv
   │
   ▼
reporting/make_table.py → LaTeX tables
```
 
The simulation infrastructure controls experiment execution only. The
statistical logic lives in `dgps/` and `estimators/`; the reporting scripts
read only the saved simulation outputs.

    ## Setup and run

    From the repository folder, with Python installed:

    ```bash
    python -m pip install -r requirements.txt
    python main.py smoke                # small pipeline check
    python main.py main_results         # 20 scenarios, 300 replications each
    python main.py p03_exponent_sweep    # additional bag-size comparisons
    ```

    The default is `main_results`. Each run writes
    `results/<run_name>_summary.csv` and `results/<run_name>_records.csv`;
    rerunning it replaces those files.

    ## Reproducibility

    Replication `i` uses data seed `first_seed + i`. The estimator derives its
    own seeds from this, so each run can be repeated. Comparable scenarios share
    data seeds, allowing the designs to be compared on the same simulated samples.

    To reproduce the results, use the same experiment settings and package versions.

    main.py <run>
    │
    ▼
    sim_infrastructure/   scenarios.py → orchestrators.py → runner.py
                        run spec       one process per      R replications,
                                        scenario             seed = first_seed + i
    │
    ▼
    dgps/reveal_plr.py    ClusteredSample: (Y, D, X, rows, cols) + true l0, m0
    │
    ▼
    estimators/plr.py     designs.py  training design (folds / bags / oracle)
                        learners.py base learner (lasso / gbm / srf)
                        → l_hat, m_hat → PLR estimate → two-way clustered SE
                        diagnostics.py  error decomposition vs oracle
    │
    ▼
    results/<run>_summary.csv, results/<run>_records.csv
    │
    ▼
    reporting/make_table.py → LaTeX tables