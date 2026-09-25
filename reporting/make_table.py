"""Write the numbers of the ten thesis tables with pandas' to_latex.

    python reporting/make_table.py   ->  reporting/tables/<name>.tex

Each file is a booktabs tabular; the caption and the notes of a table are in main.tex, around
\\input{tables/<name>}. The numbers come from the summary CSV of each campaign (one line per design),
in results/final2000/ or, for a campaign that was not rerun, in results/final300/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reporting" / "tables"

LEAD = "simple_reveal_threshold_rv0.9_re0.1"     # threshold DGP (lead_plr, fewclusters, sweep)
LINEAR = "simple_reveal_linear_rv0.9_re0.1"      # linear-signal DGP (linear_plr)

LEAD_DESIGNS = {"oracle": "Oracle",
                "no_cf": "Full sample",
                "as_iid": "Cell cross-fit",
                "multiway": "Two-way cross-fit",
                "cluster_oob_sub": r"Bagging, $7\times7$",
                "cluster_oob_sub_g0.666667": r"Bagging, $16\times16$",
                "cluster_oob_sub_g0.833333": r"Bagging, $32\times32$",
                "cluster_oob_nodrop": r"No-drop, $7\times7$"}
NOATTR_DESIGNS = {"multiway": "Two-way cross-fit",
                  "cluster_oob_sub": r"Bagging, $7\times7$"}


def load(campaign, kind="summary"):
    """Read results/final2000/<campaign>_<kind>.csv, or the final300 file if there is no rerun."""
    for folder in ["final2000", "final300"]:
        path = ROOT / "results" / folder / f"{campaign}_{kind}.csv"
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"no {kind} file for {campaign}")


def summary(campaign, design, dgp=None, learner="gbm", n=None):
    """One design's summary line (used by thesis_figures.py)."""
    s = load(campaign)
    s = s[s.design == design]
    if "learner" in s.columns:
        s = s[s.learner == learner]
    if dgp is not None:
        s = s[s.dgp == dgp]
    if n is not None:
        s = s[s.n_rows == n]
    assert len(s) == 1, f"{campaign}/{design}: expected one summary line, found {len(s)}"
    return s.iloc[0]


def cell(x, digits=3):
    """0.057 -> 0.057, -0.057 -> $-$0.057 (a real minus sign); a rounded zero gets no sign."""
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace("-", "$-$")


def pick(s, designs, columns):
    """The rows of these designs, renamed to their labels, with these columns."""
    return s.set_index("design").loc[list(designs), columns].rename(index=designs)


def headings(table, names, spanned=(), spanner=r"Coverage (95\%)"):
    """Name the columns; the ones in spanned get a common heading above them."""
    table.columns = pd.MultiIndex.from_tuples([(spanner if name in spanned else "", name) for name in names])


def write(table, name, columns):
    """Let pandas write the tabular, then add the two booktabs touches that to_latex cannot do.
    Numbers get cell(); columns already turned into text are kept."""
    text = table.to_latex(float_format=cell, na_rep="$-$", column_format=columns,
                          multicolumn_format="c", multirow=False, index_names=False)
    if table.columns.nlevels == 2:                   
        spanned = [i for i, top in enumerate(table.columns.get_level_values(0)) if top]
        skip = table.index.nlevels                   
        rule = r"\cmidrule(lr){%d-%d}" % (skip + spanned[0] + 1, skip + spanned[-1] + 1)
        text = text.replace("\\\\\n", "\\\\ " + rule + "\n", 1)           
    if table.index.nlevels == 2:                    
        width = table.index.nlevels + table.shape[1]                    
        for n, group in enumerate(table.index.get_level_values(0).unique()):
            heading = r"\multicolumn{%d}{l}{\emph{%s}} \\" % (width, group)
            if n > 0:
                heading = "\\addlinespace\n" + heading               
            text = text.replace("\n" + group + " & ", "\n" + heading + "\n & ")
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print("wrote", OUT / f"{name}.tex")


# ================================================================ Chapter 5

def lead_table():
    s = load("lead_plr")
    columns = ["bias", "sd", "rmse", "coverage", "covered_cgm", "coverage_chiang"]
    gbm = pick(s[(s.dgp == LEAD) & (s.learner == "gbm")], LEAD_DESIGNS, columns)
    lasso = pick(s[(s.dgp == LEAD) & (s.learner == "lasso")],
                 {"no_cf": "Full sample", "multiway": "Two-way cross-fit"}, columns)
    no_attributes = pick(s[s.dgp == LEAD + "_nosignatures"], NOATTR_DESIGNS, columns)

    table = pd.concat({"GBM": gbm, "Lasso": lasso, "No attributes": no_attributes})
    headings(table, ["Bias", "SD", "RMSE", "Add", "CGM", "Own"], spanned=["Add", "CGM", "Own"])
    write(table, "lead", "llrrrrrr")


def mechanism_table():
    s = load("lead_plr")
    s["product"] = (s.B_ab + s.B_bb) / s.denom        
    s["leakage"] = (s.B_bV + s.B_aV) / s.denom
    s["kappa"] = (s.B_ab / s.Eb2).where(s.Eb2 > 0)    
    columns = ["product", "leakage", "denom", "Eb2", "kappa"]
    baseline = pick(s[(s.dgp == LEAD) & (s.learner == "gbm")], LEAD_DESIGNS, columns)
    no_attributes = pick(s[s.dgp == LEAD + "_nosignatures"], NOATTR_DESIGNS, columns)

    table = pd.concat({"Baseline": baseline, "No attributes": no_attributes})
    table.columns = ["Product", "Leakage", r"$\widehat J$", r"$\mathbb{E}_n[b^2]$", r"$\kappa$"]
    write(table, "mechanism", "llrrrrr")


PLIV_COLUMNS = ["bias", "sd", "rmse", "coverage", "covered_cgm", "bias_elim_coverage", "jacobian", "mse_d"]
PLIV_NAMES = ["Bias", "SD", "RMSE", "Add", "CGM", "BE", r"$\widehat J$", r"$\mathrm{MSE}_m$"]


def pliv_rows(s, designs):
    table = pick(s, designs, PLIV_COLUMNS)
    table["jacobian"] = [cell(v, 2) for v in table["jacobian"]]      
    table["mse_d"] = [cell(v, 2) for v in table["mse_d"]]
    return table


def pliv_table():
    s = load("pliv")
    designs = {"oracle": "Oracle",
               "no_cf": "Full sample",
               "as_iid": "Cell cross-fit",
               "multiway": "Two-way cross-fit",
               "cluster_oob_sub": r"Bagging, $5\times5$"}
    centred = pliv_rows(s[s.dgp == "chen_chiang_pliv_p3"], designs)
    shifted = pliv_rows(s[s.dgp == "chen_chiang_pliv_p3_mu1"], designs)

    table = pd.concat({r"$\mu=0$": centred, r"$\mu=1$": shifted})
    headings(table, PLIV_NAMES, spanned=["Add", "CGM", "BE"])
    write(table, "pliv", "llrrrrrrrr")


def package_table():
    s = load("package_pliv")
    designs = {"oracle": "Oracle",
               "no_cf": "Full sample",
               "as_iid": "Cell cross-fit",
               "multiway": "Two-way cross-fit",
               "cluster_oob_sub": r"Bagging, $6\times5$",
               "cluster_oob_sub_g0.8": r"Bagging, $20\times18$"}

    table = pliv_rows(s, designs)
    headings(table, PLIV_NAMES, spanned=["Add", "CGM", "BE"])
    write(table, "package", "lrrrrrrrr")


# ================================================================ Appendix B

def exponents_table():
    s = load("lead_exponent_sweep")
    columns = ["bias", "sd", "rmse", "coverage", "bias_elim_coverage", "denom"]
    designs = {"cluster_oob_sub_g0.35": r"$\gamma=0.35$",
               "cluster_oob_sub_g0.45": r"$\gamma=0.45$",
               "cluster_oob_sub_g0.55": r"$\gamma=0.55$",
               "cluster_oob_sub_g0.65": r"$\gamma=0.65$",
               "cluster_oob_sub_g0.8": r"$\gamma=0.8$"}
    grid32 = pick(s[s.n_rows == 32], designs, columns)
    grid64 = pick(s[s.n_rows == 64], designs, columns)
    grid32.insert(0, "side", [4, 5, 7, 10, 17])       # bag side ceil(N^gamma); 32^0.8 is just above 16
    grid64.insert(0, "side", [5, 7, 10, 15, 28])

    table = pd.concat({r"$32\times32$": grid32, r"$64\times64$": grid64})
    headings(table, ["Side", "Bias", "SD", "RMSE", "Add", "BE", r"$\widehat J$"], spanned=["Add", "BE"],
             spanner="Coverage")
    write(table, "exponents", "llrrrrrrr")


def repeated_table():
    rep = load("repeated_partitions")
    s = load("lead_plr")
    lead = s[(s.dgp == LEAD) & (s.learner == "gbm")]
    columns = ["bias", "sd", "rmse", "mean_se", "coverage", "bias_elim_coverage", "Eb2"]

    table = pd.concat({
        "Single": pick(rep, {"rep_prediction_S1": "One partition"}, columns),
        "Estimates": pick(rep, {"rep_mean_S10": "Mean of $10$",
                                "rep_mean_S10_unadjusted": r"\quad without deviations",
                                "rep_median_S10": "Median of $10$"}, columns),
        "Predictions": pick(rep, {"rep_prediction_S2": "Mean of $2$",
                                  "rep_prediction_S5": "Mean of $5$",
                                  "rep_prediction_S10": "Mean of $10$"}, columns),
        "Bagging": pick(lead, {"cluster_oob_sub_g0.833333": r"$32\times32$",
                               "cluster_oob_sub": r"$7\times7$"}, columns)})
    table.columns = ["Bias", "SD", "RMSE", "Mean SE", "Cov.", "BE", r"$\mathbb{E}_n[b^2]$"]
    write(table, "repeated", "llrrrrrrr")


def signals_table():
    s = load("linear_plr")
    designs = {"oracle": "Oracle",
               "no_cf": "Full sample",
               "as_iid": "Cell cross-fit",
               "multiway": "Two-way cross-fit",
               "cluster_oob_sub": r"Bagging, $5\times5$",
               "cluster_oob_sub_g0.6": r"Bagging, $8\times8$",
               "cluster_oob_sub_g0.8": r"Bagging, $16\times16$",
               "cluster_oob_nodrop": r"No-drop, $5\times5$"}
    columns = ["bias", "sd", "rmse", "coverage", "covered_cgm", "coverage_chiang"]

    table = pick(s[(s.dgp == LINEAR) & (s.learner == "gbm")], designs, columns)
    headings(table, ["Bias", "SD", "RMSE", "Add", "CGM", "Own"], spanned=["Add", "CGM", "Own"])
    write(table, "signals", "lrrrrrr")


def fewclusters_table():
    s = load("fewclusters")
    s["product"] = (s.B_ab + s.B_bb) / s.denom        
    s["leakage"] = (s.B_bV + s.B_aV) / s.denom
    designs = {"oracle": "Oracle",
               "as_iid": "Cell cross-fit",
               "multiway": "Two-way cross-fit",
               "cluster_oob_sub": r"Bagging, $11\times3$",
               "cluster_oob_sub_g0.8": r"Bagging, $70\times6$"}
    columns = ["bias", "sd", "rmse", "coverage", "coverage_chiang", "leakage", "product"]

    table = pick(s[s.dgp == LEAD], designs, columns)
    headings(table, ["Bias", "SD", "RMSE", "Add", "Own", "Leakage", "Product"],
             spanned=["Add", "Own"])
    write(table, "fewclusters", "lrrrrrrr")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lead_table()     
    mechanism_table()
    pliv_table()
    package_table()
    exponents_table()       
    repeated_table()
    signals_table()
    fewclusters_table()
