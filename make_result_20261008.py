"""10-08: collect the key results of the current best pipeline (BuildScale_pipeline_20261008.txt) into result/.
Copies figures / text tables (originals untouched) and builds clean CSV tables from the per-seed metrics.
Run from the project root."""
import glob
import re
import shutil

import pandas as pd

R = "result/"
COPY = {
    "buildscale/route_v3_20261006/budget_pareto_v3_20261008.pdf": "fig1_budget_pareto_use.pdf",
    "buildscale/route_v3_20261006/budget_pareto_v3_20261008.png": "fig1_budget_pareto_use.png",
    "buildscale/route_v3_20261006/summary_20261006.txt": "raw_use_route_v3_summary_20261006.txt",
    "buildscale/table2_ci_20261007.txt": "table2_use_bootstrap_ci.txt",
    "buildscale/floors_20261007/summary_20261007.txt": "raw_floors_summary_20261007.txt",
    "buildscale/cheap_signals_eval_20261006.txt": "table6_signal_auroc_7areas.txt",
    "buildscale/audit_analysis_withaudit_20261006.txt": "table7_gpu_audit_signals.txt",
}
for s, d in COPY.items():
    shutil.copy2(s, R + d)

# ---- use: main table (budget 1) and budget curve, mean / sd over 5 seeds
m = pd.concat([pd.read_csv(f) for f in glob.glob("buildscale/route_v3_20261006/metrics_b*_s*.csv")])
ORDER = ["osmonly_A", "naive_A", "full VLM", "sat-only", "2st_nofilt", "2st_pmax", "2st_R",
         "hybfb_naive_A", "hybfb_2st_R", "route_2st_R", "cmp_2st_R"]
NAME = {"osmonly_A": "OSM-only", "naive_A": "Naive propagation", "full VLM": "Full VLM", "sat-only": "Satellite-only VLM",
        "2st_nofilt": "Two-stage (no filter)", "2st_pmax": "Two-stage + p_max filter", "2st_R": "Two-stage + R filter",
        "hybfb_naive_A": "VLM-first + naive", "hybfb_2st_R": "BuildScale (VLM-first + two-stage R)",
        "route_2st_R": "Ablation: R>=0.5 routing", "cmp_2st_R": "Ablation: R vs GBM confidence"}
b1 = m[(m.budget == 1) & m.method.isin(ORDER)]
t = b1.groupby(["group", "method", "metric"]).value.agg(["mean", "std"]).reset_index()
wide = t.pivot_table(index=["group", "method"], columns="metric", values=["mean", "std"])
wide.columns = [f"{c[1]}_{c[0]}" for c in wide.columns]
wide = wide.reset_index()
wide["order"] = wide.method.map({k: i for i, k in enumerate(ORDER)})
wide["name"] = wide.method.map(NAME)
g_order = {"all": 0, "with SV": 1, "no SV": 2, "human non-RES": 3}
wide = wide.sort_values(["group", "order"], key=lambda s: s.map(g_order) if s.name == "group" else s)
cols = ["group", "method", "name"] + [f"{k}_{s}" for k in ["macroF1", "acc", "ECE", "Brier", "hc_acc", "hc_cov", "dF1_vs_naive_A"] for s in ["mean", "std"]]
wide[cols].to_csv(R + "table1_use_main.csv", index=False, float_format="%.4f")
with open(R + "table1_use_main.txt", "w") as f:
    print("Use (6 human major classes), 7 areas leave-one-area-out, budget = all street-view buildings; mean +- sd over 5 seeds", file=f)
    print("source: buildscale/route_v3_20261006/metrics_b1_s*.csv (train_route_v3_20261006.py, job 910327)\n", file=f)
    for g, d in wide.groupby("group", sort=False):
        print(f"## {g}", file=f)
        print(f"  {'method':40s} macro-F1        acc     ECE     Brier   dF1 vs naive", file=f)
        for _, r in d.iterrows():
            print(f"  {r['name']:40s} {r.macroF1_mean:.3f}+-{r.macroF1_std:.3f}  {r.acc_mean:.3f}  {r.ECE_mean:.3f}  "
                  f"{r.Brier_mean:.4f}  {r.dF1_vs_naive_A_mean:+.3f}", file=f)
        print(file=f)

bud = m[(m.group == "all") & m.metric.isin(["macroF1", "acc"]) & m.method.isin(["osmonly_A", "naive_A", "2st_R", "hybfb_2st_R"])]
bt = bud.groupby(["budget", "method", "metric"]).value.agg(["mean", "std"]).unstack("metric")
bt.columns = [f"{c[1]}_{c[0]}" for c in bt.columns]
bt = bt.reset_index()
N_COUNTY, N_SV, SEC = 364437, 36488, 1.93 + 1.41
bt.insert(1, "share_of_county_pct", bt.budget * N_SV / N_COUNTY * 100)
bt.insert(2, "est_gpu_h", bt.budget * N_SV * SEC / 3600)
bt.to_csv(R + "table3_use_budget.csv", index=False, float_format="%.4f")

# ---- floors (Spearman vs LiDAR class 6 p95), parsed from the 10-07 summary (hybq is computed there)
rows, grp = [], None
for l in open("buildscale/floors_20261007/summary_20261007.txt"):
    h = re.match(r"## (all|with SV|no SV)\s*$", l)
    if h:
        grp = h.group(1); continue
    if l.startswith("## per area"):
        grp = None
    x = re.match(r"\s+(\S+(?: VLM)?)\s+([\d.-]+)\+-([\d.]+)\s+vs naive ([+-][\d.]+) \((\d)/5\) \[([+-][\d.]+),([+-][\d.]+)\]", l)
    if grp and x:
        rows.append(dict(group=grp, method=x.group(1), spearman_mean=float(x.group(2)), spearman_sd=float(x.group(3)),
                         d_vs_naive=float(x.group(4)), seeds_better=int(x.group(5)),
                         d_ci_lo=float(x.group(6)), d_ci_hi=float(x.group(7))))
pd.DataFrame(rows).to_csv(R + "table4_floors_spearman.csv", index=False)
print("tables written:", len(wide), "use rows,", len(bt), "budget rows,", len(rows), "floor rows")
