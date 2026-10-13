"""Targeted sample of street pseudo-labels for learning R (10-06): oversample likely VLM errors.

Flags (cheap_signals_20261006.csv.gz): nonres = VLM major non-RES; lowm = tok_use_margin < 0.8;
xmdis = satellite major != street major; gbmdis = OSM-only GBM argmax != street major.
Pool: county street pseudo-labels outside all 8 LiDAR areas (7 eval areas + random_c1).
Strata (mutually exclusive, in this order):
  A  >= 3 flags                                   300
  B  VLM non-RES, < 3 flags (REL/GOV/EDU/ACC/IND/AGR first, then COM)   150
  C  VLM RES, exactly 2 flags                     100
  D  0-1 flags, uniform                            50
Spatial spread: at most 3 buildings per 2 km cell overall (cells filled in random order).
pi = sampled / stratum pool (for B: per sub-stratum minority / COM), used as 1/pi weights when fitting R.
The labels are for fitting R only, never for evaluation. Labelling order is randomly shuffled across strata,
so a partly finished set still covers every stratum. The page shows no model output.
Output: annotation_hard_20261006/sample_hard_20261006.csv
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(20261006)
NR = ["COM", "IND", "AGR", "REL", "GOV", "EDU", "ACC"]
S = pd.read_csv("buildscale/cheap_signals_20261006.csv.gz", low_memory=False)
lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv", usecols=["building_id"])
feat = pd.read_csv("features/building_features_county_20260922.csv.gz", usecols=["building_id", "f_x_utm", "f_y_utm"])
S = S[S.ans_major.isin(["RES"] + NR) & ~S.building_id.isin(lid.building_id)].merge(feat, on="building_id")
S["nonres"] = S.ans_major.isin(NR); S["lowm"] = S.tok_use_margin < 0.8
S["xmdis"] = S.xm_agree == 0; S["gbmdis"] = S.osm_agree == 0
S["nflag"] = S[["nonres", "lowm", "xmdis", "gbmdis"]].sum(1)
S["cell"] = (S.f_x_utm // 2000).astype(int).astype(str) + "_" + (S.f_y_utm // 2000).astype(int).astype(str)
S["stratum"] = np.select([S.nflag >= 3, S.nonres, (~S.nonres) & (S.nflag == 2), S.nflag <= 1], ["A", "B", "C", "D"], "-")
S.loc[S.stratum == "B", "stratum"] = np.where(S.loc[S.stratum == "B", "ans_major"] == "COM", "B_com", "B_min")
S = S.sample(frac=1, random_state=20261006).reset_index(drop=True)
TARGET = [("A", 300), ("B_min", 150), ("B_com", 150), ("C", 100), ("D", 50)]   # B_com only tops B up to 150
per_cell, picked = {}, []
for st, n in TARGET:
    if st == "B_com":
        n = 150 - sum(1 for p in picked if p[1] == "B_min")
    k = 0
    for i, r in S[S.stratum == st].iterrows():
        if k >= n: break
        if st != "B_min" and per_cell.get(r.cell, 0) >= 3: continue   # B_min (30 in the county) is taken whole
        per_cell[r.cell] = per_cell.get(r.cell, 0) + 1; picked.append((i, st)); k += 1
out = S.loc[[i for i, _ in picked]].copy()
pool = S.stratum.value_counts()
out["pi"] = out.stratum.map(lambda s: (out.stratum == s).sum() / pool[s])
out["order"] = rng.permutation(len(out)) + 1
out = out.sort_values("order")
cols = ["order", "building_id", "stratum", "pi", "nflag", "nonres", "lowm", "xmdis", "gbmdis", "ans_major", "tok_use_margin", "cell"]
out[cols].to_csv("annotation_hard_20261006/sample_hard_20261006.csv", index=False)
print("pool per stratum:", pool.to_dict())
print(out.groupby("stratum").agg(n=("building_id", "size"), pi=("pi", "first")).to_string())
print("VLM major in sample:", out.ans_major.value_counts().to_dict(), "| cells used", out.cell.nunique(), "| total", len(out))
