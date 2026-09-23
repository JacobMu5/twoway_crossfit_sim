"""Draw the two thesis outcome figures.

    python reporting/thesis_figures.py   ->  reporting/figures/{fig_lead_errors, fig_sweep}.pdf

fig_lead_errors  Chapter 5: sampling distributions of theta_hat - theta_0 (from the lead_plr records)
fig_sweep        Chapter 5: bias and RMSE against the bag side (from the summary CSVs)
"""

import math

import matplotlib.pyplot as plt
import seaborn as sns

from make_table import LEAD, LINEAR, ROOT, load, summary

sns.set_theme(context="paper", style="whitegrid", font="serif")
OUT = ROOT / "reporting" / "figures"


def lead_errors():
    r = load("lead_plr", "records")                       
    r = r[(r.dgp == LEAD) & (r.learner == "gbm")]
    styles = {"oracle": dict(color="0.55", fill=True, label="Oracle"),
              "as_iid": dict(color="black", ls=":", label="Cell cross-fit"),
              "multiway": dict(color="black", ls="--", label="Two-way cross-fit"),
              "cluster_oob_sub": dict(color="black", lw=1.6, label=r"Bagging, $7\times7$")}
    fig, ax = plt.subplots(figsize=(5.25, 2.6))         
    for design, style in styles.items():
        err = r[r.design == design].err
        sns.kdeplot(x=err, ax=ax, **style)
        ax.plot([err.mean(), err.mean()], [0, 0.9], color=style["color"], lw=1.4)   
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set(xlabel=r"$\widehat\theta - \theta_0$", ylabel="Density", xlim=(-0.22, 0.22))
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_lead_errors.pdf")


def sweep():
    gammas = (0.35, 0.45, 0.55, 0.65, 0.8)
    oracle = {32: summary("linear_plr", "oracle", LINEAR)["rmse"],
              64: summary("lead_plr", "oracle", LEAD)["rmse"]}
    fig, (ax_bias, ax_rmse) = plt.subplots(1, 2, figsize=(5.71, 2.6))   
    for n, marker in ((32, "o"), (64, "s")):
        sides = [min(n - 1, max(2, math.ceil(n ** g))) for g in gammas]
        rows = [summary("lead_exponent_sweep", f"cluster_oob_sub_g{g:g}", n=n) for g in gammas]
        ax_bias.errorbar(sides, [x["bias"] for x in rows], yerr=[1.96 * x["bias_mcse"] for x in rows],
                         marker=marker, color="black", label=rf"${n}\times{n}$ grid")
        ax_rmse.plot(sides, [x["rmse"] for x in rows], marker=marker, color="black", label=rf"${n}\times{n}$ grid")
        ax_rmse.axhline(oracle[n], color="0.55", ls=":")
    ax_bias.axhline(0, color="0.4", lw=0.8)
    for ax in (ax_bias, ax_rmse):
        ax.set_xticks([4, 7, 10, 15, 20, 28])
        ax.set_xlabel("Bag side")
    ax_bias.set(ylabel=r"Bias of $\widehat\theta$", title="(a)")
    ax_rmse.set(ylabel=r"RMSE of $\widehat\theta$", title="(b)")
    ax_bias.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_sweep.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    lead_errors()
    sweep()
    print("wrote fig_lead_errors.pdf and fig_sweep.pdf to", OUT)
