"""Appendix pilot: does a different learner remove the two-way (K=2) cross-fit penalty
on the lead threshold DGP? Compares Chiang's K=2 fit with the subsampled bagging design
of the main results across three learners, and writes the finished table. The main study
is untouched.

    python reporting/run_appendix.py
    -> results/appendix_summary.csv and reporting/tables/appendix_learner.tex
"""

import math, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so the imports below resolve

from dgps.reveal_plr import SimpleSignalPLRDGP
from estimators.designs import multiway_folds, predict_cluster_oob_sub, EST_SEED_OFFSET
from estimators.learners import LEARNERS, LearnerSpec
from estimators.plr import _fold_predict, two_way_variance

GRID, SEEDS, NOISE, BAGS, GAMMA = 64, range(9010001, 9010021), (1.0, 2.0), 130, 0.45
SIDE, Z = math.ceil(GRID ** GAMMA), 1.959963984540054
DESIGNS = {"Two-way cross-fit": "twoway", "Bagging": "bagging"}   # display name -> internal code
gbm = lambda **kw: LearnerSpec(lambda s: HistGradientBoostingRegressor(
    learning_rate=.1, min_samples_leaf=5, random_state=s, **kw), BAGS)
SPECS = {"Original GBM": LEARNERS["gbm"], "Stump GBM": gbm(max_iter=100, max_leaf_nodes=2),
         "Deeper GBM": gbm(max_iter=200, max_leaf_nodes=31)}


def draw(seed, sd):
    """Threshold DGP; raising sd grows only the cell-level outcome noise (treatment fixed)."""
    return SimpleSignalPLRDGP(signal="threshold", resid_share=.9,
                              sd_eps=sd, eps_resid_share=.1 / sd ** 2).sample(GRID, GRID, seed)


def predict(s, splits, target, spec, design, seed):
    if design == "twoway":
        return _fold_predict(splits, s.x, target, spec, seed)
    return predict_cluster_oob_sub(s.x, target, s.rows, s.cols, GRID, GRID, spec, seed,
                                   n_bags=BAGS, sub_exponent=GAMMA)


def fit(s, l_hat, m_hat):
    """Robinson estimate, treatment-nuisance MSE and a CGM coverage flag for one draw."""
    d, y = s.d - m_hat, s.y - l_hat
    denom = float(np.mean(d * d)); theta = float(np.mean(d * y) / denom)
    se = math.sqrt(two_way_variance(d * (y - theta * d), s.rows, s.cols, GRID, GRID, "cgm")) / (len(d) * denom)
    err = theta - s.theta0
    return dict(error=err, m_mse=float(np.mean((m_hat - s.m0) ** 2)), covered=int(abs(err) <= Z * se))


def run_seed(seed):
    """Every learner x design on one paired draw; returns a list of record dicts."""
    with threadpool_limits(limits=1):
        samples = {sd: draw(seed, sd) for sd in NOISE}
        base, est = samples[1.0], seed + EST_SEED_OFFSET
        splits = multiway_folds(GRID, GRID, np.random.default_rng(est), k=2)
        out = []
        for learner, spec in SPECS.items():
            for name, design in DESIGNS.items():
                m_hat = predict(base, splits, base.d, spec, design, est)
                o = est if design == "twoway" else est + 7919
                out += [dict(sd=sd, learner=learner, design=name,
                             **fit(s, predict(s, splits, s.y, spec, design, o), m_hat)) for sd, s in samples.items()]
    print(f"draw {seed} done", flush=True)
    return out


LABEL = {"Two-way cross-fit": "Two-way cross-fit", "Bagging": rf"Bagging, ${SIDE}\times{SIDE}$"}


def table(summary, sd, label, note):
    """One Bonn table float for a noise level: learner x design rows plus Oracle."""
    t = summary[summary.sd == sd].set_index(["learner", "design"])
    def row(head, key):
        r = t.loc[key]
        return (f"{head} & {r['bias']:.3f} & {r['std']:.3f} & {r['rmse']:.3f} & "
                f"{r['m_mse']:.3f} & {r['cover'] / r['reps']:.3f} \\\\").replace(" & -0.000 ", " & 0.000 ")
    body = []
    for learner in SPECS:
        if body:
            body.append(r"\addlinespace")
        body.append(r"\multicolumn{7}{@{}l}{\emph{%s}} \\" % learner)
        body += [row(f" & {LABEL[d]}", (learner, d)) for d in DESIGNS]
    return "\n".join([r"\begin{table}[!ht]", r"\centering",
        rf"\caption{{Learner sensitivity on the threshold PLR, outcome residual SD ${sd:g}$}}",
        rf"\label{{{label}}}", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Learner & Design & Bias & SD & RMSE & Treat. MSE & CGM \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}",
        r"\begin{tablenotes}[Notes]", note, r"\end{tablenotes}", r"\end{table}"])


def render(summary):
    note1 = (rf"{len(SEEDS)} paired pilot replications on the unchanged ${GRID}\times{GRID}$ threshold DGP; "
             rf"$\theta_0=1$; $B={BAGS}$ eligible ${SIDE}\times{SIDE}$ bags per target. The two-way design uses "
             r"Chiang et al.\ $K=2$ folds; bagging is the subsampled cluster-OOB design of the main results. SD "
             r"is the replication standard deviation; Treat. MSE the treatment-nuisance mean squared error "
             r"against the true mean; CGM is the fraction of nominal $95\%$ two-way CGM intervals that cover. "
             r"Equal base-learner settings do not imply equal effective complexity or cost.")
    note2 = (r"As Table~\ref{tab:appx-sd1}, raising only the independent cell-level outcome noise to residual "
             r"SD $2$; the threshold signal and cluster shocks are unchanged, so Treat. MSE is identical to "
             r"Table~\ref{tab:appx-sd1} by construction.")
    return (table(summary, NOISE[0], "tab:appx-sd1", note1) + "\n\n"
            + table(summary, NOISE[1], "tab:appx-sd2", note2) + "\n")


def main():
    with ProcessPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count() or 1)) as pool:
        records = [r for recs in pool.map(run_seed, SEEDS) for r in recs]
    summary = pd.DataFrame(records).groupby(["sd", "learner", "design"], sort=False).agg(
        bias=("error", "mean"), std=("error", lambda e: e.std(ddof=1)),
        rmse=("error", lambda e: math.sqrt((e ** 2).mean())),
        m_mse=("m_mse", "mean"), cover=("covered", "sum"), reps=("error", "size")).reset_index()
    print(summary.to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/appendix_summary.csv", index=False)
    out = Path(__file__).resolve().parent / "tables"; out.mkdir(exist_ok=True)
    (out / "appendix_learner.tex").write_text(render(summary), encoding="utf-8")
    print("wrote", out / "appendix_learner.tex")


if __name__ == "__main__":
    main()
