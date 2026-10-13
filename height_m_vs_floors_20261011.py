"""10-11: VLM asked height in metres (vlm_height_m_7areas_20261011_shard*.jsonl) vs the 10-02 floor-count run, on the same
7-area street-view buildings, against LiDAR class 6 p95 (edge_clipped == 0, C1 excluded, NB/BADFP excluded).
Reports answer rate, Spearman, MAE / median error in metres, per-area Spearman, and by LiDAR height band. Run from project root."""
import glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
F = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
fl = []
for f in glob.glob("vlm_floors_7areas_20261002_shard*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["modality"] != "street": continue
        p = np.array([r["p_height"].get(k, 0) for k in F]) if r["p_height"] else np.zeros(10); s = p.sum()
        fl.append(dict(building_id=r["building_id"], fl_ans=r["height"], fl_E=(p @ np.arange(1, 11)) / s if s > 0 else np.nan))
fl = pd.DataFrame(fl)
hm = pd.DataFrame([json.loads(l) for f in glob.glob("vlm_height_m_7areas_20261011_shard*.jsonl") for l in open(f)])
hm = hm[["building_id", "height_m", "height_unseen", "raw1"]]
lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv")
lid = lid[(lid.edge_clipped == 0) & lid.height_p95_m.notna() & (lid.area != "random_c1")][["building_id", "area", "height_p95_m"]]
bad = set()
for f in glob.glob("annotation_20261002/labels_dump_20261006/*/labels/*.json"):
    if json.load(open(f))["code"] in ("NB", "BADFP"): bad.add(int(f.split("/")[-1][:-5]))
m = hm.merge(fl, on="building_id", how="left").merge(lid, on="building_id")
m = m[~m.building_id.isin(bad)]
print(f"buildings with LiDAR: {len(m)}  (height run total {len(hm)})")
print(f"height answered a number {m.height_m.notna().mean()*100:.1f}% | X {m.height_unseen.mean()*100:.1f}% | unparsed {(m.height_m.isna() & ~m.height_unseen).sum()}")
print(f"floors answered a number {m.fl_ans.isin(F).mean()*100:.1f}%")
print("height answer quantiles (m):", m.height_m.quantile([.05, .25, .5, .75, .95, 1]).round(1).to_dict())
both = m[m.height_m.notna() & m.fl_ans.isin(F)]
print(f"\n## same buildings (both answered a number), n={len(both)}")
print(f"  Spearman metres        {spearmanr(both.height_m, both.height_p95_m)[0]:.3f}")
print(f"  Spearman floors E      {spearmanr(both.fl_E, both.height_p95_m)[0]:.3f}")
err = both.height_m - both.height_p95_m
print(f"  metres: MAE {err.abs().mean():.2f} m, median abs err {err.abs().median():.2f} m, median bias {err.median():+.2f} m")
for k in [3.0, 3.5, 4.0]:
    e2 = both.fl_E * k - both.height_p95_m
    print(f"  floors E x {k} m: MAE {e2.abs().mean():.2f} m, median abs err {e2.abs().median():.2f} m, median bias {e2.median():+.2f} m")
print("\n  per area (Spearman metres | floors E):")
for a, g in both.groupby("area"):
    if len(g) >= 20:
        print(f"    {a:34s} n={len(g):4d}  {spearmanr(g.height_m, g.height_p95_m)[0]:.3f} | {spearmanr(g.fl_E, g.height_p95_m)[0]:.3f}")
print("\n  by LiDAR band: median VLM metres / median floors E")
for lo, hi in [(0, 5), (5, 8), (8, 12), (12, 20), (20, 999)]:
    g = both[(both.height_p95_m >= lo) & (both.height_p95_m < hi)]
    if len(g): print(f"    LiDAR {lo:>2}-{hi:<3} m n={len(g):4d}  VLM {g.height_m.median():5.1f} m  floors E {g.fl_E.median():.2f}")
both.to_csv("buildscale/height_m_vs_floors_20261011.csv", index=False)
