"""Print a paste-ready LaTeX table float from the merged results summary.

All campaigns now live in one CSV (results/main_results_summary.csv), so the
table is chosen by NAME. Chapter 4 shows the classical designs only (the
problem); Chapter 5 adds the sub-cluster OOB design (the solution). Both draw
on the same run, so every row shares one seed stream.

    python reporting/make_table.py <table> [csv ...]

<table> is one of: ch4_perf, ch4_anat, ch5_perf, ch5_anat, ch5_gamma.
Extra CSV paths are concatenated (e.g. to fold in a separate lasso-64 run).

The LaTeX goes to stdout; diagnostics (rows loaded, duplicate keys, omitted
rows) go to stderr, so `... > table.tex` still gives a clean file.
"""
import csv, sys

f = lambda x, k: f"{float(x):.{k}f}"
warn = lambda msg: print("make_table:", msg, file=sys.stderr)
grid_of = lambda r: int(float(r["n_rows"]))

DESIGN = {"oracle": "Oracle", "no_cf": "No cross-fit", "as_iid": "As-i.i.d.",
          "multiway": r"Multiway ($K^2$-fold)",
          "cluster_oob_sub": "Sub-cluster OOB",
          "cluster_oob_nodrop": "No-drop OOB"}
LEARNER = {"lasso": "Lasso", "gbm": "GBM"}


def perf(rows, designs, learners, grid, caption, label):
    by = {}
    for r in rows:
        if grid_of(r) != grid:
            continue
        key = (r["learner"], r["design"])
        if key in by:
            warn(f"duplicate row {key} at {grid}x{grid}; keeping the last")
        by[key] = r
    out = [r"\begin{table}[!ht]", r"\centering",
           r"\caption{%s}  %% <-- your caption" % caption,
           r"\label{%s}" % label, r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Design & Bias & SD & RMSE & SE/SD & Cov.\ (95\%) \\"]
    for lk in learners:
        out += [r"\midrule",
                r"\multicolumn{6}{@{}l}{\emph{Learner: %s}} \\" % LEARNER[lk]]
        for dk in designs:
            r = by.get((lk, dk))
            if not r:
                warn(f"no row for learner={lk!r} design={dk!r} at {grid}x{grid}; omitted")
                continue
            out.append(f"{DESIGN[dk]} & {f(r['bias'],3)} & {f(r['sd'],3)} & "
                       f"{f(r['rmse'],3)} & {f(r['se_ratio'],2)} & {f(r['coverage'],3)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}[Notes]",
            r"SD is the Monte-Carlo standard deviation of $\widehat\theta$; RMSE its root-mean-square "
            r"error; SE/SD the mean estimated (additive two-way) standard error divided by SD, so $1$ "
            r"indicates calibrated inference; Cov.\ the empirical coverage of the nominal $95\%$ CI.",
            r"\end{tablenotes}", r"\end{table}"]
    return "\n".join(out)


def anat(rows, designs, grids, caption, label):
    by, seen = {}, set()
    for r in rows:
        if r["learner"] != "gbm" or grid_of(r) not in grids:
            continue
        key = (r["design"], grid_of(r))
        if key in seen:
            warn(f"duplicate row {key}; it will appear twice in the table")
        seen.add(key)
        by.setdefault(r["design"], []).append(r)
    out = [r"\begin{table}[!ht]", r"\centering",
           r"\caption{%s}  %% <-- your caption" % caption,
           r"\label{%s}" % label, r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r"Design & $N{=}M$ & Bias & SD & Cov. & Cov.$^{\dagger}$ & Leak \\", r"\midrule"]
    for dk in designs:
        if dk not in by:
            warn(f"no rows for design={dk!r} (gbm) at grids {grids}; omitted")
    present = [d for d in designs if d in by]
    for i, dk in enumerate(present):
        for j, r in enumerate(sorted(by[dk], key=grid_of)):
            name = DESIGN[dk] if j == 0 else ""
            out.append(f"{name} & {grid_of(r)} & {f(r['bias'],3)} & {f(r['sd'],3)} & "
                       f"{f(r['coverage'],3)} & {f(r['bias_elim_coverage'],3)} & {f(r['leak'],3)} \\\\")
        if i < len(present) - 1:
            out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}[Notes]",
            r"Concentrated (reveal) two-way PLR DGP, GBM learner, 300 replications. Cov.\ is coverage of "
            r"the nominal $95\%$ CI; Cov.$^{\dagger}$ the same after removing the estimated bias, so a gap "
            r"Cov.$^{\dagger}\!\gg\!$Cov.\ marks \emph{bias}-driven undercoverage. Leak is the clustering "
            r"leakage into the score. Oracle uses the true nuisances.",
            r"\end{tablenotes}", r"\end{table}"]
    return "\n".join(out)


def gamma(rows, caption, label):
    G = {"cluster_oob_sub": "0.45", "cluster_oob_sub_g0.65": "0.65"}
    by = {}
    for r in rows:
        if r["learner"] != "gbm" or grid_of(r) != 32:
            continue
        if r["design"] in by:
            warn(f"duplicate row for design={r['design']!r} at 32x32; keeping the last")
        by[r["design"]] = r
    out = [r"\begin{table}[!ht]", r"\centering",
           r"\caption{%s}  %% <-- your caption" % caption,
           r"\label{%s}" % label, r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"$\gamma$ & Bias & SD & RMSE & SE/SD & Cov.\ (95\%) \\", r"\midrule"]
    for dk, g in G.items():
        r = by.get(dk)
        if not r:
            warn(f"no row for design={dk!r} (gbm) at 32x32; gamma={g} omitted")
            continue
        out.append(f"${g}$ & {f(r['bias'],3)} & {f(r['sd'],3)} & "
                   f"{f(r['rmse'],3)} & {f(r['se_ratio'],2)} & {f(r['coverage'],3)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}[Notes]",
            r"Sub-cluster OOB at bag exponent $\gamma$ (bag side $\lceil N^{\gamma}\rceil$ clusters), "
            r"GBM, $32\times32$, 300 replications. Larger $\gamma$ trades bag fit quality against the "
            r"number of honest (omit-both) bags.",
            r"\end{tablenotes}", r"\end{table}"]
    return "\n".join(out)


args = sys.argv[1:]
name = args[0] if args and not args[0].endswith(".csv") else "ch5_perf"
paths = [a for a in args if a.endswith(".csv")] or ["results/main_results_summary.csv"]
rows = [r for p in paths for r in csv.DictReader(open(p))]
warn(f"loaded {len(rows)} rows from {', '.join(paths)}")

if name == "ch4_perf":
    print(perf(rows, ["no_cf", "as_iid", "multiway"], ["lasso", "gbm"], 32,

               "Classical cross-fitting designs across learners.", "tab:ch4-perf"))
elif name == "ch4_anat":
    print(anat(rows, ["oracle", "as_iid", "multiway"], [32, 64],
               "Anatomy of cross-fitting failure: the classical designs.", "tab:ch4-anat"))
elif name == "ch5_perf":
    print(perf(rows, ["oracle", "no_cf", "as_iid", "multiway",
                      "cluster_oob_sub", "cluster_oob_nodrop"], ["lasso", "gbm"], 32,
               "All designs across learners.", "tab:ch5-perf"))
elif name == "ch5_anat":
    print(anat(rows, ["oracle", "as_iid", "multiway", "cluster_oob_sub"], [32, 64],
               "Anatomy across designs, including sub-cluster OOB.", "tab:ch5-anat"))
elif name == "ch5_gamma":
    print(gamma(rows, "The sub-sampling exponent dial.", "tab:ch5-gamma"))
else:
    sys.exit(f"unknown table {name!r}; choose ch4_perf, ch4_anat, ch5_perf, ch5_anat, ch5_gamma")