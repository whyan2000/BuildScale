"""10-11: add today's results to result/ (10-08 files untouched; make_result_20261008.py not changed).
  table9_floors_variants.{csv,txt}   floors propagation variants (N / U / G / Nm / confidence filter), 5 seeds
  table10_floors_pmax_bins.txt       VLM floor answer token p_max vs agreement with LiDAR (7 areas)
  table11_use_neighbour_feats.{csv,txt}  use model + neighbour-use features (A / AB / ABm), 5 seeds
  table12_county_floors_Nm_log*.txt  county floors files (Nm, Nm + confidence filter) logs
  fig2_floors_spearman.{pdf,png}     floors Spearman vs LiDAR: no-street-view propagation variants | all-building final outputs
  fig3_floors_pmax_bins.{pdf,png}    Spearman of the VLM floor answer by token p_max bin
Run from the project root."""
import glob
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = "result/"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

# ---------------------------------------------------------------- floors variants
runs = ["floors_v2_20261011", "floors_v2G_20261011", "floors_v2Nm_20261011", "floors_v2conf_20261011"]
fm = []
for r in runs:
    for f in glob.glob(f"buildscale/{r}/metrics_s*.csv"):
        d = pd.read_csv(f); d["run"] = r; fm.append(d)
fm = pd.concat(fm)
fm = fm.drop_duplicates(["seed", "group", "area", "method", "metric"])        # base / N / Nm appear in several runs (identical)
FM = ["osmonly", "naive", "base", "N", "U", "G", "NU", "NG", "Nm", "NmG", "Nm_f2", "Nm_fall",
      "full VLM", "hybfb_base", "hybq_base", "hybfb_Nm", "hybq_Nm", "hybfb_Nm_fall", "hybq_Nm_fall"]
FNAME = {"osmonly": "OSM-only", "naive": "Naive (single stage)", "base": "Two-stage (10-07)", "N": "+ neighbour heights",
         "U": "+ use (VLM / GBM)", "G": "+ use (GBM only)", "NU": "+ N + U", "NG": "+ N + G",
         "Nm": "+ masked neighbour heights (Nm)", "NmG": "+ Nm + G", "Nm_f2": "+ Nm, p_max filter stage>=2",
         "Nm_fall": "+ Nm, p_max filter all stages", "full VLM": "Full VLM (street else satellite)",
         "hybfb_base": "VLM-first + two-stage", "hybq_base": "VLM-first q-map + two-stage (10-07 best)",
         "hybfb_Nm": "VLM-first + Nm", "hybq_Nm": "VLM-first q-map + Nm",
         "hybfb_Nm_fall": "VLM-first + Nm filter (DATASET)", "hybq_Nm_fall": "VLM-first q-map + Nm filter"}
sp = fm[(fm.area == "all") & (fm.metric == "spearman") & fm.method.isin(FM)]
pv = sp.pivot_table(index=["group", "seed"], columns="method", values="value")
rows = []
for g in ["all", "no SV", "with SV"]:
    x = pv.loc[g]
    for m in FM:
        if m not in x:
            continue
        d = x[m] - x["base"]
        rows.append(dict(group=g, method=m, name=FNAME[m], spearman_mean=x[m].mean(), spearman_sd=x[m].std(),
                         d_vs_base=d.mean(), seeds_better_than_base=int((d > 1e-9).sum()), n_seeds=len(x[m].dropna())))
t9 = pd.DataFrame(rows)
t9.to_csv(R + "table9_floors_variants.csv", index=False, float_format="%.4f")
pa = fm[(fm.area != "all") & (fm.metric == "spearman") & fm.method.isin(["osmonly", "base", "N", "Nm", "Nm_fall", "hybq_Nm_fall", "hybfb_Nm_fall"])]
pa = pa.groupby(["area", "method"]).value.mean().unstack()[["osmonly", "base", "N", "Nm", "Nm_fall", "hybfb_Nm_fall", "hybq_Nm_fall"]]
with open(R + "table9_floors_variants.txt", "w") as f:
    print("Floors: Spearman(prediction, LiDAR class 6 p95), 7 areas leave-one-area-out, 5,414 buildings "
          "(922 with a numeric street VLM answer, 4,492 without); mean +- sd over 5 seeds.", file=f)
    print("d vs base = mean paired difference to the 10-07 two-stage model; 'better' = seeds out of 5 above it.", file=f)
    print("sources: buildscale/{" + ",".join(runs) + "}/metrics_s*.csv (train_floors_v2_20261011.py)\n", file=f)
    for g, d in t9.groupby("group", sort=False):
        print(f"## {g}", file=f)
        for _, r in d.iterrows():
            print(f"  {r['name']:44s} {r.spearman_mean:.3f} +- {r.spearman_sd:.3f}   d {r.d_vs_base:+.3f}  better {r.seeds_better_than_base}/{r.n_seeds}", file=f)
        print(file=f)
    print("## per-area Spearman, all scored buildings (mean of 5 seeds)", file=f)
    print(pa.round(3).to_string(), file=f)

# ---------------------------------------------------------------- p_max bins (seed-independent: VLM answers only)
F = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
v = []
for fn in glob.glob("vlm_floors_7areas_20261002_shard*.jsonl"):
    for l in open(fn):
        r = json.loads(l)
        if r["modality"] == "street" and r["height"] in F:
            p = np.array([r["p_height"].get(k, 0) for k in F]); s = p.sum()
            v.append((r["building_id"], (p @ np.arange(1, 11)) / s, p.max() / s))
v = pd.DataFrame(v, columns=["building_id", "E", "pmax"])
lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv")
lid = lid[(lid.edge_clipped == 0) & lid.height_p95_m.notna() & (lid.area != "random_c1")][["building_id", "height_p95_m"]]
bad = {int(fn.split("/")[-1][:-5]) for fn in glob.glob("annotation_20261002/labels_dump_20261006/*/labels/*.json")
       if json.load(open(fn))["code"] in ("NB", "BADFP")}
v = v.merge(lid, on="building_id"); v = v[~v.building_id.isin(bad)]
BINS = [(0, 0.5, "< 0.5"), (0.5, 0.8, "0.5-0.8"), (0.8, 1.01, ">= 0.8")]
rng = np.random.default_rng(0)
pb = []
for lo, hi, lab in BINS:
    g = v[(v.pmax >= lo) & (v.pmax < hi)]
    bs = [spearmanr(*g.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))[["E", "height_p95_m"]].T.values)[0] for _ in range(1000)]
    pb.append(dict(bin=lab, n=len(g), spearman=spearmanr(g.E, g.height_p95_m)[0], lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                   share_multi=(g.E > 1.5).mean()))
pb = pd.DataFrame(pb)
with open(R + "table10_floors_pmax_bins.txt", "w") as f:
    print("Street VLM floor answer (7 areas, numeric answer, LiDAR class 6 present, not NB/BADFP): Spearman(E[floors], LiDAR p95)", file=f)
    print("by token p_max of the HEIGHT answer; 95% CI = 1,000 building bootstrap. Filter threshold 0.5 was fixed in advance.\n", file=f)
    for _, r in pb.iterrows():
        print(f"  p_max {r.bin:8s} n={r.n:4d}  Spearman {r.spearman:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  VLM says > 1 floor {r.share_multi*100:.0f}%", file=f)
    print(f"\n  county pseudo-labels dropped by p_max < 0.5: 278 of 34,601 VLM floor labels (see floors_v2conf_20261011/eval_s0.txt)", file=f)

# ---------------------------------------------------------------- use + neighbour-use features
um = []
for ft in ["A", "AB", "ABm"]:
    for fn in glob.glob(f"buildscale/route_v3nm_20261011/{ft}/metrics_b1_s*.csv"):
        d = pd.read_csv(fn); d["feats"] = ft; um.append(d)
um = pd.concat(um)
UM = ["osmonly_A", "naive_A", "2st_R", "hybfb_2st_R"]
x = um[um.metric.isin(["macroF1", "acc"]) & um.method.isin(UM)]
t11 = x.groupby(["group", "method", "metric", "feats"]).value.agg(["mean", "std"]).unstack("feats")
t11.columns = [f"{c[1]}_{c[0]}" for c in t11.columns]
t11 = t11.reset_index(); t11.to_csv(R + "table11_use_neighbour_feats.csv", index=False, float_format="%.4f")
p = um[um.metric == "macroF1"].pivot_table(index=["group", "method", "seed"], columns="feats", values="value")
with open(R + "table11_use_neighbour_feats.txt", "w") as f:
    print("Use (major classes), budget 1, 7 areas LOAO, 5 seeds. A = 10-06 main line (reproduces route_v3: BuildScale 0.536);", file=f)
    print("AB = + neighbour-use features (OSM else street VLM use of the 50 nearest other labelled buildings), unmasked;", file=f)
    print("ABm = same, masked for labelled rows (as floors Nm). Conclusion: neighbour-use features do not help use.\n", file=f)
    for g in ["all", "no SV", "with SV", "human non-RES"]:
        print(f"## {g}  macro-F1 mean (A | AB | ABm)   ABm - A per seed", file=f)
        for m in UM:
            q = p.loc[(g, m)]
            print(f"  {m:12s} {q.A.mean():.3f} | {q.AB.mean():.3f} | {q.ABm.mean():.3f}    {[round(z, 3) for z in (q.ABm - q.A).tolist()]}", file=f)
        print(file=f)

for fn in glob.glob("buildscale/floors_county_20261011/log_20261011*.txt"):
    shutil.copy2(fn, R + "table12_county_floors_Nm_" + fn.split("/")[-1])

# ---------------------------------------------------------------- fig 2: floors Spearman
def panel(ax, g, methods, hl, title):
    x = pv.loc[g]
    for i, m in enumerate(methods):
        c = ORANGE if m in hl else BLUE
        vals = x[m].to_numpy()
        ax.plot([vals.mean() - vals.std(), vals.mean() + vals.std()], [i, i], color=c, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(vals, np.full(len(vals), i) + 0.18, s=10, color=c, alpha=0.45, lw=0, zorder=2)
        ax.scatter([vals.mean()], [i], s=42, color=c, edgecolor="white", lw=1.5, zorder=3)
        ax.text(vals.max() + 0.006, i, f"{vals.mean():.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels([FNAME[m] for m in methods])
    ax.invert_yaxis(); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    ax.set_xlabel("Spearman with LiDAR roof height")


fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw=dict(wspace=0.95))
panel(axs[0], "no SV", ["osmonly", "naive", "base", "N", "G", "U", "Nm", "Nm_fall"], {"Nm", "Nm_fall"},
      "(a) Buildings without street view (n = 4,492): propagation")
panel(axs[1], "all", ["osmonly", "hybq_base", "hybfb_Nm_fall", "hybq_Nm_fall"], {"hybfb_Nm_fall", "hybq_Nm_fall"},
      "(b) All buildings (n = 5,414): final output")
for ax in axs:
    ax.set_xlim(0.36, 0.64)
fig.text(0.01, -0.04, "Dot = mean of 5 seeds, bar = +- 1 sd, small dots = single seeds. Orange = uses masked neighbour heights (Nm). "
         "LiDAR class 6 under-measures high-rise roofs.", fontsize=7.5, color=MUTED)
for ext in ["pdf", "png"]:
    fig.savefig(R + f"fig2_floors_spearman.{ext}", bbox_inches="tight", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 3: p_max bins
fig, ax = plt.subplots(figsize=(4.2, 2.6))
xs = np.arange(len(pb))
ax.bar(xs, pb.spearman, width=0.55, color=BLUE, zorder=2)
ax.errorbar(xs, pb.spearman, yerr=[pb.spearman - pb.lo, pb.hi - pb.spearman], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
ax.axhline(0, color=MUTED, lw=0.8)
for i, r in pb.iterrows():
    ax.text(i, max(r.hi, 0) + 0.03, f"n={r.n}", ha="center", fontsize=8, color=MUTED)
ax.set_xticks(xs); ax.set_xticklabels([f"p_max {b}" for b in pb.bin])
ax.set_ylabel("Spearman with LiDAR"); ax.set_ylim(min(-0.35, pb.lo.min() - 0.05), 0.8)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("VLM floor answers: token confidence vs accuracy", fontsize=9.5, loc="left", color=INK)
for ext in ["pdf", "png"]:
    fig.savefig(R + f"fig3_floors_pmax_bins.{ext}", bbox_inches="tight", dpi=200)
print(open(R + "table10_floors_pmax_bins.txt").read())
print(t9[t9.group == "no SV"][["method", "spearman_mean", "d_vs_base", "seeds_better_than_base"]].round(3).to_string())
