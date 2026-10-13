"""10-07 aggregate train_floors_20261007 over seeds + one post-hoc variant.

Seed 1 showed: on buildings with street view the VLM beats every GBM (Spearman 0.634 vs 0.525), but hybfb_2st on ALL
buildings is below 2st (0.500 vs 0.509) -> the two sources are on different scales (VLM E[floors] vs GBM E[floors]),
so pasting VLM values into the GBM ranking mixes scales. Variant hybq_<m>: label-free quantile mapping, per area,
of the VLM E onto the GBM E distribution of the same street-view buildings (keeps the VLM's ordering among them,
puts them on the GBM scale). Uses no LiDAR / no human label.
Output: buildscale/floors_20261007/summary_20261007.txt
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = "buildscale/floors_20261007/"
F = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
ORDER = ["osmonly", "naive", "2st", "full VLM", "hybfb_naive", "hybfb_2st", "hybq_naive", "hybq_2st"]
NBOOT = 1000
out = open(D + "summary_20261007.txt", "w")


def log(s=""):
    print(s); print(s, file=out)


res = []          # (seed, group, area, method, spearman)
boots = {}        # (seed, group) -> {method: bootstrap array}
for f in sorted(glob.glob(D + "preds_s*.csv.gz")):
    seed = int(f.split("_s")[-1][:-7])
    p = pd.read_csv(f, keep_default_na=False, na_values=[""])
    num = p.street_ans.astype(str).isin(F).to_numpy()
    for g in ["naive", "2st"]:
        q = p[f"hybfb_{g}"].to_numpy(float).copy()
        for a, idx in p.groupby("area").indices.items():
            m = np.zeros(len(p), bool); m[idx] = True; m &= num
            if m.sum() < 2:
                continue
            r = (pd.Series(p["full VLM"].to_numpy()[m]).rank(pct=True).to_numpy())
            q[m] = np.quantile(p[g].to_numpy()[m], np.clip(r, 0, 1))
        p[f"hybq_{g}"] = q
    s = p[p.scored.astype(str) == "True"]
    sn = s.street_ans.astype(str).isin(F).to_numpy()
    rng = np.random.default_rng(2000 + seed)
    for tag, m in [("all", np.ones(len(s), bool)), ("with SV", sn), ("no SV", ~sn)]:
        ss = s[m]; n = len(ss)
        idx = [rng.integers(0, n, n) for _ in range(NBOOT)]
        boots[(seed, tag)] = {}
        for name in ORDER:
            x = ss[name].fillna(ss[name].median()).to_numpy(); h = ss.lidar_p95.to_numpy()
            res.append((seed, tag, "all", name, spearmanr(x, h)[0]))
            boots[(seed, tag)][name] = np.array([spearmanr(x[i], h[i])[0] for i in idx])
        if tag == "all":
            for a, g in ss.groupby("area"):
                for name in ORDER:
                    res.append((seed, tag, a, name, spearmanr(g[name].fillna(g[name].median()), g.lidar_p95)[0]))
R = pd.DataFrame(res, columns=["seed", "group", "area", "method", "spearman"])
seeds = sorted(R.seed.unique())
log(f"# aggregate_floors_20261007.py  seeds {seeds}  (Spearman with LiDAR class 6 p95; mean +- sd over seeds;"
    f" d vs naive / 2st paired per seed, (k/n) = seeds with a positive difference; CI = seed-0 paired bootstrap)")
for tag in ["all", "with SV", "no SV"]:
    r = R[(R.group == tag) & (R.area == "all")].pivot(index="seed", columns="method", values="spearman")
    log(f"\n## {tag}")
    for name in ORDER:
        d1 = r[name] - r["naive"]; d2 = r[name] - r["2st"]
        b = boots[(seeds[0], tag)]
        ci = lambda d: f"[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]"
        log(f"  {name:12s} {r[name].mean():.3f}+-{r[name].std():.3f}   vs naive {d1.mean():+.3f} ({(d1 > 0).sum()}/{len(d1)}) "
            f"{ci(b[name] - b['naive'])}   vs 2st {d2.mean():+.3f} ({(d2 > 0).sum()}/{len(d2)}) {ci(b[name] - b['2st'])}")
log("\n## per area (all scored), mean over seeds")
pa = R[(R.group == "all") & (R.area != "all")].groupby(["area", "method"]).spearman.mean().unstack()[ORDER]
log(pa.round(3).to_string())
out.close()
