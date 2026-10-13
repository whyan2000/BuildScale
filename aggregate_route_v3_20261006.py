"""Aggregate train_route_v3_20261006: (1) main table at budget 1, (2) VLM-call budget curve (all scored buildings)."""
import glob
import pandas as pd
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("buildscale/route_v3_20261006/metrics_b*_s*.csv"))])
N_COUNTY, N_SV, SEC = 364437, 36488, 1.93 + 1.41
out = [f"# aggregate_route_v3_20261006.py  budgets {sorted(d.budget.unique())}  seeds {sorted(d.seed.unique())}"]
ORDER = ["osmonly_A", "naive_A", "2st_nofilt", "2st_pmax", "2st_R", "full VLM", "sat-only",
         "hybfb_naive_A", "route_naive_A", "cmp_naive_A", "hybfb_2st_R", "route_2st_R", "cmp_2st_R"]
b1 = d[d.budget == 1]
for g in ["all", "with SV", "no SV", "human non-RES"]:
    out.append(f"\n## budget 1, {g}   (mean +- sd over seeds; dF1 paired per seed, (k/n) = seeds with a positive difference)")
    for m in ORDER:
        x = b1[(b1.group == g) & (b1.method == m)].pivot(index="seed", columns="metric", values="value")
        f = lambda k: f"{x[k].mean():.3f}+-{x[k].std():.3f}"
        dd = lambda k: f"{x[k].mean():+.3f}+-{x[k].std():.3f} ({(x[k] > 0).sum()}/{len(x)})"
        out.append(f"  {m:14s} F1 {f('macroF1')}  acc {f('acc')}  ECE {f('ECE')}  Brier {f('Brier')}  "
                   f"acc@.9 {f('hc_acc')} cov {x.hc_cov.mean()*100:4.1f}%  vs naive_A {dd('dF1_vs_naive_A')}  vs hybfb_naive_A {dd('dF1_vs_hybfb_naive_A')}")
out.append("\n## VLM-call budget curve, all scored buildings (n=5479): macro-F1 / accuracy, mean +- sd over seeds")
a = d[d.group == "all"]
CURVE = ["osmonly_A", "naive_A", "2st_R", "hybfb_2st_R", "cmp_2st_R"]
for b in sorted(a.budget.unique()):
    calls = b * N_SV
    out.append(f"  budget {b:<5g} calls {calls:8.0f} = {calls / N_COUNTY * 100:5.2f}% of county, ~{calls * SEC / 3600:5.1f} GPU h")
    for m in CURVE:
        x = a[(a.budget == b) & (a.method == m)].pivot(index="seed", columns="metric", values="value")
        out.append(f"      {m:12s} F1 {x.macroF1.mean():.3f}+-{x.macroF1.std():.3f}  acc {x.acc.mean():.3f}+-{x.acc.std():.3f}")
for m in ["full VLM", "sat-only"]:
    x = a[(a.budget == 1) & (a.method == m)]
    v = x.pivot(index="seed", columns="metric", values="value").iloc[0]
    out.append(f"  100% of county ({m}, ~136 GPU h county-wide): F1 {v.macroF1:.3f}  acc {v.acc:.3f}")
open("buildscale/route_v3_20261006/summary_20261006.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
