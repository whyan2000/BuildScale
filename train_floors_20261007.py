"""10-07 floors (number of storeys): propagation from street-view VLM pseudo-labels to the whole county,
evaluated against LiDAR class 6 roof height (user 10-06: floors ground truth = LiDAR) on the 7 areas.

Labels (floors, capped at KMAX=6 = "6+"):
  OSM      building:levels (osm_levels, rounded, >= 1)                         ~7k buildings
  street   VLM floor answer (vlm_floors_20261002.py, Qwen3.6-27B), county run 879301
           (vlm_floors_sv_street_20261006_shard*.jsonl, 35,355) + the 7-area street run 868994 (1,010).
           Argmax answer; "unseen" -> no label. Expected value E[floors] over 1..9, 10+ (=10) kept for scoring.
Models (LightGBM, same features as train_route_v3_20261006: geometry + neighbourhood group A; same params,
2 km-cell early-stopping split). Ordinal output = expected floors, E = 1 + sum_k P(y > k), k = 1..KMAX-1.
  osmonly     cumulative (Frank-Hall) binaries P(y > k) on OSM labels
  naive       cumulative binaries on OSM else street pseudo-label
  2st         two stages (as the use model): stage 1 P(y > 1) on all naive labels (one storey vs more);
              stage 2 continuation ratios P(y > k | y > k-1), k = 2..KMAX-1, each fitted only on rows with y >= k
              -> the upper classes are learnt from the multi-storey buildings only.
              P(y > k) = prod_{j <= k} P(y > j | y > j-1)
  full VLM    street E[floors] where the building has a street answer, else satellite E[floors] (7-area run only)
  hybfb_<m>   street VLM E[floors] if the answer is a number, else model m (as hybfb in the use table)
Leave-one-area-out over the 7 areas (training = all labelled county buildings outside the held-out area).
Scored: 7 areas, LiDAR class 6 p95 present, edge_clipped == 0, human code not NB/BADFP.
Metrics: Spearman(E, LiDAR p95) [main; rank only, no metres-per-storey assumption], Kendall tau-b,
  AUROC for LiDAR p95 >= 7 m (multi-storey proxy; 10-06: VLM 1 storey median 5.0 m, 2 storeys 7.2 m),
  per-area Spearman. Paired bootstrap (1,000) CI of the Spearman difference vs naive.
  Known limit: class 6 misses most high-rise roofs (downtown), so tall buildings are under-measured.
Outputs: buildscale/floors_20261007/{eval_s<seed>.txt, metrics_s<seed>.csv, preds_s<seed>.csv.gz}
Run from the project root:  python buildscale/train_floors_20261007.py --seed 0
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, kendalltau, rankdata
from sklearn.metrics import roc_auc_score

T0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ARGS = ap.parse_args()
OUTD = "buildscale/floors_20261007/"
os.makedirs(OUTD, exist_ok=True)
SFX = f"s{ARGS.seed}"
KMAX, NBOOT, MULTI_M = 6, 1000, 7.0
F = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
AREAS = ["downtown_orlando", "industrial_orlando_central_park", "random_c2", "random_c3",
         "residential_long_lake", "tourism_universal", "ucf_main_campus"]
LOG = OUTD + f"eval_{SFX}.txt"
open(LOG, "w").close()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time()-T0:5.0f}s] {s}", flush=True)
    with open(LOG, "a") as f:
        print(s, file=f)


# ---------------------------------------------------------------- data (features as train_route_v3_20261006)
feat = pd.read_csv("features/building_features_county_20260922.csv.gz", low_memory=False)
feat = feat[feat.meta_in_orange == 1].reset_index(drop=True)
FCOLS = list(feat.columns[feat.columns.get_loc("f_area_m2"):feat.columns.get_loc("f_nbr500_area_rank") + 1])
bid = feat.building_id.to_numpy(); N = len(bid)
nbr = pd.read_csv("features/nbr_features_20261005.csv.gz").set_index("building_id").reindex(bid)
ACOLS = [c for c in nbr.columns if c.startswith("ctxA_")]
X = np.c_[feat[FCOLS].to_numpy(np.float32), nbr[ACOLS].to_numpy(np.float32)]
xy = feat[["f_x_utm", "f_y_utm"]].to_numpy()
cell_id = (xy[:, 0] // 2000).astype(int) * 100000 + (xy[:, 1] // 2000).astype(int)
lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv").set_index("building_id")
area_of = pd.Series(bid).map(lid.area).to_numpy()
area_of = np.where(np.isin(area_of, AREAS), area_of, None)
in_area = pd.notna(area_of)
lid_h = pd.Series(bid).map(lid.height_p95_m.where(lid.edge_clipped == 0)).to_numpy(float)

osm = pd.read_csv("buildscale/osm_labels_20261001_v2.csv").set_index("building_id").reindex(bid)
y_osm = np.clip(np.round(osm.osm_levels.to_numpy(float)), 1, KMAX)
y_osm = np.where(np.isnan(y_osm), 0, y_osm).astype(int)                      # 0 = no label

st_ans = np.full(N, "", object); st_E = np.full(N, np.nan); sa_E = np.full(N, np.nan)
pos = {b: i for i, b in enumerate(bid)}


def e_floors(r):
    p = np.array([r["p_height"].get(k, 0) for k in F]); s = p.sum()
    return (p @ np.arange(1, 11)) / s if s > 0 else np.nan


for f in sorted(glob.glob("vlm_floors_sv_street_20261006_shard*.jsonl") + glob.glob("vlm_floors_7areas_20261002_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l); i = pos.get(r["building_id"])
        if i is None:
            continue
        if r["modality"] == "street":
            st_ans[i] = r["height"]; st_E[i] = e_floors(r) if r["height"] in F else np.nan
        else:
            sa_E[i] = e_floors(r)
has_st = st_ans != ""
st_num = np.isin(st_ans, F)
y_pl = np.where(st_num, np.clip([F.index(a) + 1 if a in F else 0 for a in st_ans], 0, KMAX), 0).astype(int)
y_naive = np.where(y_osm > 0, y_osm, y_pl)

hum = {}
for f in glob.glob("annotation_20261002/labels_dump_20261006/*/labels/*.json"):
    hum[int(f.split("/")[-1][:-5])] = json.load(open(f))["code"]
bad = np.array([hum.get(b) in ("NB", "BADFP") for b in bid])
scored = in_area & ~np.isnan(lid_h) & ~bad
log(f"# train_floors_20261007.py seed {ARGS.seed}  KMAX {KMAX}")
log(f"in-county {N}; OSM floors {int((y_osm > 0).sum())}; street answers {int(has_st.sum())} (number {int(st_num.sum())}); "
    f"naive labels {int((y_naive > 0).sum())}")
log("  naive label distribution: " + str({k: int((y_naive == k).sum()) for k in range(1, KMAX + 1)}))
log(f"scored (7 areas, LiDAR, not NB/BADFP) {int(scored.sum())}; with street answer {int((scored & has_st).sum())}")

# ---------------------------------------------------------------- GBM
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=ARGS.seed,
              objective="binary")


def fit_bin(tr, y, te):
    """P(y = 1) for te; class-balanced (inverse sqrt) weights, 2 km-cell validation split, early stopping."""
    if tr.sum() < 20 or y[tr].min() == y[tr].max():
        return np.full(te.sum(), y[tr].mean() if tr.any() else 0.0)
    ug = np.unique(cell_id[tr])
    val_cells = set(np.random.RandomState(ARGS.seed).choice(ug, max(1, len(ug) // 10), replace=False))
    v = np.array([g in val_cells for g in cell_id[tr]])
    yt = y[tr].astype(int); cnt = np.bincount(yt, minlength=2).astype(float)
    w = (cnt ** -0.5)[yt]; w = w / w.mean()
    Xt = X[tr]
    m = lgb.train(PARAMS, lgb.Dataset(Xt[~v], yt[~v], weight=w[~v]), 2000,
                  valid_sets=[lgb.Dataset(Xt[v], yt[v], weight=w[v])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(X[te], num_iteration=m.best_iteration)


def cumulative(tr, y, te):
    """Frank-Hall: P(y > k) fitted independently; monotone by running min; E = 1 + sum."""
    P = np.c_[[fit_bin(tr & (y > 0), (y > k).astype(int), te) for k in range(1, KMAX)]].T
    P = np.minimum.accumulate(P, axis=1)
    return 1 + P.sum(1)


def two_stage(tr, y, te):
    """continuation ratio: P(y > k | y > k-1) fitted on rows with y >= k."""
    surv = np.ones(te.sum()); E = np.ones(te.sum())
    for k in range(1, KMAX):
        surv = surv * fit_bin(tr & (y >= k), (y > k).astype(int), te)
        E += surv
    return E


E = {g: np.full(N, np.nan) for g in ["osmonly", "naive", "2st"]}
for a in AREAS:
    t1 = time.time()
    te = (area_of == a); out = ~te
    E["osmonly"][te] = cumulative(out, y_osm, te)
    E["naive"][te] = cumulative(out, y_naive, te)
    E["2st"][te] = two_stage(out, y_naive, te)
    log(f"  fold {a}: {int(te.sum())} buildings ({time.time()-t1:.0f}s)")

E["full VLM"] = np.where(st_num, st_E, sa_E)
for g in ["naive", "2st"]:
    E[f"hybfb_{g}"] = np.where(st_num, st_E, E[g])
ORDER = ["osmonly", "naive", "2st", "full VLM", "hybfb_naive", "hybfb_2st"]


# ---------------------------------------------------------------- metrics
def spear(p, h, w=None):
    if w is None:
        return spearmanr(p, h)[0]
    rp, rh = rankdata(p), rankdata(h)
    mp, mh = np.average(rp, weights=w), np.average(rh, weights=w)
    return np.average((rp - mp) * (rh - mh), weights=w) / np.sqrt(
        np.average((rp - mp) ** 2, weights=w) * np.average((rh - mh) ** 2, weights=w))


rng = np.random.default_rng(1000 + ARGS.seed)
GROUPS = [("all", scored), ("with SV", scored & st_num), ("no SV", scored & ~st_num)]
rows = []
for tag, m in GROUPS:
    h = lid_h[m]; n = int(m.sum()); multi = (h >= MULTI_M).astype(int)
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(NBOOT)])
    log(f"\n## floors [{tag}] n={n}  LiDAR p95 >= {MULTI_M:g} m: {multi.sum()} ({multi.mean()*100:.1f}%)")
    log(f"  {'method':12s} Spearman [95% CI]      Kendall  AUROC(>={MULTI_M:g}m)  dSpearman vs naive [CI]   (missing preds)")
    bs, mts = {}, {}
    for name in ORDER:
        p = E[name][m]
        ok = ~np.isnan(p)
        pf = np.where(ok, p, np.nanmedian(p) if ok.any() else 1.0)   # missing (full VLM, no answer) -> median
        bs[name] = np.array([spear(pf, h, w) for w in W])
        mts[name] = dict(spearman=spear(pf, h), kendall=kendalltau(pf, h)[0],
                         auroc=roc_auc_score(multi, pf) if 0 < multi.sum() < n else np.nan, missing=int((~ok).sum()))
    for name in ORDER:
        mt = mts[name]
        d = bs[name] - bs["naive"]
        dpt = mt["spearman"] - mts["naive"]["spearman"]
        log(f"  {name:12s} {mt['spearman']:.3f} [{np.percentile(bs[name], 2.5):.3f},{np.percentile(bs[name], 97.5):.3f}]  "
            f"{mt['kendall']:.3f}    {mt['auroc']:.3f}         {dpt:+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]   ({mt['missing']})")
        for k, v in mt.items():
            rows.append(dict(seed=ARGS.seed, group=tag, area="all", method=name, metric=k, value=v))
        rows.append(dict(seed=ARGS.seed, group=tag, area="all", method=name, metric="dSpearman_vs_naive", value=dpt))

log("\n## per-area Spearman (all scored buildings)")
log("  " + f"{'area':34s} {'n':>5s} " + " ".join(f"{n:>11s}" for n in ORDER))
for a in AREAS:
    m = scored & (area_of == a); h = lid_h[m]
    vals = []
    for name in ORDER:
        p = E[name][m]; p = np.where(np.isnan(p), np.nanmedian(p) if (~np.isnan(p)).any() else 1.0, p)
        vals.append(spear(p, h)); rows.append(dict(seed=ARGS.seed, group="all", area=a, method=name, metric="spearman", value=vals[-1]))
    log(f"  {a:34s} {int(m.sum()):5d} " + " ".join(f"{v:11.3f}" for v in vals))

log("\n## median LiDAR p95 (m) by rounded prediction, all scored buildings")
for name in ["naive", "2st", "hybfb_2st"]:
    r = np.clip(np.round(E[name][scored]), 1, KMAX)
    log(f"  {name:12s} " + "  ".join(f"{int(k)}:{np.median(lid_h[scored][r == k]):.1f}m(n={(r == k).sum()})" for k in np.unique(r)))

pd.DataFrame(rows).to_csv(OUTD + f"metrics_{SFX}.csv", index=False)
out = {"building_id": bid[in_area], "area": area_of[in_area], "lidar_p95": lid_h[in_area], "scored": scored[in_area],
       "street_ans": st_ans[in_area], "osm_floors": y_osm[in_area]}
for name in ORDER:
    out[name] = E[name][in_area]
pd.DataFrame(out).to_csv(OUTD + f"preds_{SFX}.csv.gz", index=False, float_format="%.4f")

# county-wide two-stage model (all labels, no held-out area) -> floors for every building, for the dataset
if ARGS.seed == 0:
    allm = np.ones(N, bool)
    E_cty = two_stage(allm, y_naive, allm)
    final = np.where(st_num, st_E, E_cty)
    pd.DataFrame({"building_id": bid, "E_floors_2st": E_cty, "E_floors_final": final,
                  "src": np.where(st_num, "street_vlm", "gbm_2st")}).to_csv(
        OUTD + "county_floors_s0.csv.gz", index=False, float_format="%.4f")
    log("county floors written (hybfb_2st: street VLM where answered, else two-stage GBM)")
log(f"\ndone in {time.time()-T0:.0f}s")
