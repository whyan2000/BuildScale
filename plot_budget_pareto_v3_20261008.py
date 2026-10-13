"""10-08 copy of plot_budget_pareto_20261006.py for routing v3 (R from targeted labels); main line = hybfb_2st_R.
Performance vs VLM-call budget (Pareto figure) from buildscale/route_v3_20261006/metrics_b*_s*.csv.

x = share of county buildings sent to the VLM (log scale; 0 drawn at a broken-axis position), tick labels also
give the GPU hours. Two panels (macro-F1, accuracy) -- no dual axis. Mean +- sd over 5 seeds.
Lines: naive pseudo-label propagation (naive_A), reliability-filtered two-stage propagation (2st_R), BuildScale
(hybfb_2st_R: VLM answer where given, else 2st_R). References: OSM-only (OSM+A, budget 0), full VLM and satellite-only VLM (100%).
Output: buildscale/route_v3_20261006/budget_pareto_v3_20261008.{pdf,png}
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = pd.concat([pd.read_csv(f) for f in glob.glob("buildscale/route_v3_20261006/metrics_b*_s*.csv")])
d = d[d.group == "all"]
N_COUNTY, N_SV, SEC = 364437, 36488, 1.93 + 1.41
X0 = 0.35                                            # where budget 0 is drawn on the log axis
BUDGETS = sorted(d.budget.unique())
pct = {b: (b * N_SV / N_COUNTY * 100 if b > 0 else X0) for b in BUDGETS}

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = [("hybfb_2st_R", "BuildScale (ours)", "#2a78d6", "o"),
          ("2st_R", "Reliability-filtered two-stage GBM", "#1baf7a", "^"),
          ("naive_A", "Naive pseudo-label propagation", "#eb6834", "s")]
REF = [("full VLM", "Full VLM (street + satellite)", "D"), ("sat-only", "Satellite-only VLM", "v")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True, facecolor=SURF)
for ax, (metric, ylab) in zip(axes, [("macroF1", "Macro-F1 (use, 6 classes)"), ("acc", "Accuracy")]):
    ax.set_facecolor(SURF)
    sub = d[d.metric == metric].groupby(["method", "budget"]).value.agg(["mean", "std"])
    osm = sub.loc[("osmonly_A", 0.0)]
    ax.axhline(osm["mean"], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(150, osm["mean"], " OSM-only", va="center", ha="left", color=INK2, fontsize=8)
    for key, label, col, mk in SERIES:
        s = sub.loc[key]
        xs = [pct[b] for b in s.index]
        ax.fill_between(xs, s["mean"] - s["std"], s["mean"] + s["std"], color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, s["mean"], color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.5, label=label, zorder=3)
    for key, label, mk in REF:
        v = sub.loc[(key, 1.0), "mean"]
        ax.plot([100], [v], marker=mk, ms=7, color=INK2, mec=SURF, mew=1.5, ls="none", label=label, zorder=3)
    ticks = [X0, 1, 2.5, 5, 10, 100]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "1%\n3.4 h", "2.5%\n8.5 h", "5%\n17 h", "10%\n34 h", "100%\n~136 h"])
    ax.minorticks_off()
    ax.set_xlim(0.28, 140)
    ax.set_ylabel(ylab)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    # broken-axis mark between 0 and 1%
    ax.text(0.6, ax.get_ylim()[0], "//", ha="center", va="center", color=INK2, fontsize=8, clip_on=False)
axes[1].set_xlabel("Buildings sent to the VLM (share of county; est. GPU hours)", color=INK2)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.1, 1, 1))
for ext in ["pdf", "png"]:
    fig.savefig(f"buildscale/route_v3_20261006/budget_pareto_v3_20261008.{ext}", dpi=200, facecolor=SURF)
print("saved")
