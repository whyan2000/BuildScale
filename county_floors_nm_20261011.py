"""10-11 county-wide floors with the 10-11 best variant hybq_Nm (7-area LOAO, 5 seeds: Spearman 0.553 vs hybq_2st 0.509;
no-street-view buildings 0.543 vs 0.490). Copy of county_floors_hybq_20261008.py; the only change is the feature
matrix: geometry + group A + Nm (masked neighbour heights, see train_floors_v2_20261011.py):
  known floors kf = OSM levels else street VLM E[floors]; stats over the 50 nearest OTHER known buildings;
  for labelled rows, known neighbours closer than r_i are hidden, r_i drawn from the unlabelled buildings'
  nearest-known distance (train/test distance shift: labelled 22 m vs unlabelled 350 m median).
  The non-street GBM (E_gbm_all) is now refitted here on all labels (10-08 read it from floors_20261007).
Extra column E_hybfb = raw street VLM E where answered else GBM (the quantile mapping compresses high-rise
buildings, 10-08 note; dataset users who need absolute floors of tall buildings should use this one).
--tau t (10-11 evening): confidence filter as Nm_fall in train_floors_v2_20261011 -- VLM floor labels (no OSM levels)
with token p_max < t are dropped from every stage; output/log names get the suffix _fall (t = 0.5, fixed in advance).
User 10-11: the dataset uses E_hybfb (street VLM raw value kept, so high-rise buildings are not compressed).
Check 1 of the 10-08 script (area vs cell mapping) is dropped; check 2 compares the 10-08 and 10-11 county files.
Outputs: buildscale/floors_county_20261011/{county_floors_hybq_Nm_20261011.csv.gz, log_20261011.txt}
Run from the project root.
Original 10-08 docstring follows.
10-08 county-wide floors with the best evaluated variant, hybq_2st (10-07: Spearman 0.509 vs hybfb_2st 0.488).

hybq in the 7-area evaluation (aggregate_floors_20261007.py): inside each held-out area, the street-VLM E[floors]
of the buildings with a numeric street answer is quantile-mapped onto the distribution of the (out-of-fold)
two-stage GBM E of the SAME buildings -> VLM ordering, GBM scale. No LiDAR / human label used.
County version:
  * GBM for buildings WITHOUT a numeric street answer: the all-label two-stage model of train_floors_20261007.py
    (floors_20261007/county_floors_s0.csv.gz, E_floors_2st; unchanged).
  * GBM target distribution for the street buildings must not have seen their own VLM label (in the evaluation
    it was out-of-fold by area) -> 5-fold spatial cross-fitting: 2 km cells randomly assigned to 5 folds, the
    two-stage model is trained outside the fold and predicts the street buildings inside it.
  * Mapping region = 2 km cell (the evaluation areas are 2 x 2 km); cells with < MIN_N numeric street answers
    use one county-wide mapping instead.
Checks: (1) held-out: on the 7-area LOAO predictions (floors_20261007/preds_s*.csv.gz) compare mapping per area
(as 10-07) with mapping per 2 km cell inside the area (fallback: whole area) -> Spearman vs LiDAR, 5 seeds;
(2) not held-out (sanity only): Spearman of the county files vs LiDAR on the scored 7-area buildings.
Outputs: buildscale/floors_county_20261008/{county_floors_hybq_20261008.csv.gz, log_20261008.txt}
Run from the project root.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

T0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("--tau", type=float, default=0.0); ARGS = ap.parse_args()
SUF = "_fall" if ARGS.tau > 0 else ""
OUTD = "buildscale/floors_county_20261011/"
os.makedirs(OUTD, exist_ok=True)
LOG = OUTD + f"log_20261011{SUF}.txt"
open(LOG, "w").close()
KMAX, NFOLD, MIN_N, SEED = 6, 5, 30, 0
F = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"]
AREAS = ["downtown_orlando", "industrial_orlando_central_park", "random_c2", "random_c3",
         "residential_long_lake", "tourism_universal", "ucf_main_campus"]


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time()-T0:5.0f}s] {s}", flush=True)
    with open(LOG, "a") as f:
        print(s, file=f)


# ---------------------------------------------------------------- data (as train_floors_20261007)
feat = pd.read_csv("features/building_features_county_20260922.csv.gz", low_memory=False)
feat = feat[feat.meta_in_orange == 1].reset_index(drop=True)
FCOLS = list(feat.columns[feat.columns.get_loc("f_area_m2"):feat.columns.get_loc("f_nbr500_area_rank") + 1])
bid = feat.building_id.to_numpy(); N = len(bid)
nbr = pd.read_csv("features/nbr_features_20261005.csv.gz").set_index("building_id").reindex(bid)
ACOLS = [c for c in nbr.columns if c.startswith("ctxA_")]
X = np.c_[feat[FCOLS].to_numpy(np.float32), nbr[ACOLS].to_numpy(np.float32)]
xy = feat[["f_x_utm", "f_y_utm"]].to_numpy()
cell_id = (xy[:, 0] // 2000).astype(int) * 100000 + (xy[:, 1] // 2000).astype(int)
cell_of = pd.Series(cell_id, index=bid)

osm = pd.read_csv("buildscale/osm_labels_20261001_v2.csv").set_index("building_id").reindex(bid)
y_osm = np.clip(np.round(osm.osm_levels.to_numpy(float)), 1, KMAX)
y_osm = np.where(np.isnan(y_osm), 0, y_osm).astype(int)

st_ans = np.full(N, "", object); st_E = np.full(N, np.nan); st_pm = np.full(N, np.nan)
pos = {b: i for i, b in enumerate(bid)}
for f in sorted(glob.glob("vlm_floors_sv_street_20261006_shard*.jsonl") + glob.glob("vlm_floors_7areas_20261002_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l); i = pos.get(r["building_id"])
        if i is None or r["modality"] != "street":
            continue
        st_ans[i] = r["height"]
        if r["height"] in F:
            p = np.array([r["p_height"].get(k, 0) for k in F]); st_E[i] = (p @ np.arange(1, 11)) / p.sum(); st_pm[i] = p.max() / p.sum()
st_num = np.isin(st_ans, F)
y_pl = np.where(st_num, np.clip([F.index(a) + 1 if a in F else 0 for a in st_ans], 0, KMAX), 0).astype(int)
y_naive = np.where(y_osm > 0, y_osm, y_pl)
log(f"in-county {N}; numeric street answers {int(st_num.sum())}; naive labels {int((y_naive > 0).sum())}")


# ---------------------------------------------------------------- Nm features (as train_floors_v2_20261011, seed 0)
def nfeat(dd, vv):
    """same statistics as NF from (distance, value) rows sorted by distance; inf distance = no neighbour."""
    fin = np.isfinite(dd); vz = np.where(fin, vv, np.nan)
    with np.errstate(all="ignore"):
        out = [dd[:, 0], np.nanmean(vz[:, :5], 1), np.nanmedian(vz[:, :5], 1), np.nanmean(vz[:, :10], 1),
               np.nanmedian(vz[:, :10], 1),
               np.nansum(vz[:, :10] / (dd[:, :10] + 5), 1) / np.nansum(np.where(fin[:, :10], 1 / (dd[:, :10] + 5), np.nan), 1)]
        for r in (100, 250):
            w = dd <= r; n = w.sum(1)
            out += [n.astype(float), np.where(n > 0, np.where(w, vz, 0).sum(1) / np.maximum(n, 1), np.nan),
                    np.where(n > 0, np.where(w, vz, -np.inf).max(1), np.nan),
                    np.where(n > 0, ((vz > 1.5) & w).sum(1) / np.maximum(n, 1), np.nan)]
    return np.c_[out].T.astype(np.float32)



kf = np.where(y_osm > 0, np.clip(osm.osm_levels.to_numpy(float), 1, 10), np.where(st_num, st_E, np.nan))
kn = ~np.isnan(kf); kidx = np.flatnonzero(kn)
tree = cKDTree(xy[kn])
dd, ii = tree.query(xy, k=51)
gi = kidx[ii]
dd = np.where(gi == np.arange(N)[:, None], np.inf, dd)
order = np.argsort(dd, 1, kind="stable")[:, :50]
dd = np.take_along_axis(dd, order, 1); vv = kf[np.take_along_axis(gi, order, 1)]
XN = nfeat(dd, vv)
rng_m = np.random.default_rng(700 + SEED)
r_i = rng_m.choice(dd[~kn, 0], kn.sum())
d2, i2 = tree.query(xy[kn], k=400)
g2 = kidx[i2]
d2 = np.where((g2 == kidx[:, None]) | (d2 < r_i[:, None]), np.inf, d2)
o2 = np.argsort(d2, 1, kind="stable")[:, :50]
d2 = np.take_along_axis(d2, o2, 1); v2 = kf[np.take_along_axis(g2, o2, 1)]
XNm = XN.copy(); XNm[kn] = nfeat(d2, v2)
del dd, ii, gi, vv, d2, i2, g2, v2
X = np.c_[X, XNm]
log(f"Nm features added; X {X.shape}; known floors {int(kn.sum())}")

# ---------------------------------------------------------------- GBM (as train_floors_20261007, seed 0)
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=SEED,
              objective="binary")


def fit_bin(tr, y, te):
    if tr.sum() < 20 or y[tr].min() == y[tr].max():
        return np.full(te.sum(), y[tr].mean() if tr.any() else 0.0)
    ug = np.unique(cell_id[tr])
    val_cells = set(np.random.RandomState(SEED).choice(ug, max(1, len(ug) // 10), replace=False))
    v = np.array([g in val_cells for g in cell_id[tr]])
    yt = y[tr].astype(int); cnt = np.bincount(yt, minlength=2).astype(float)
    w = (cnt ** -0.5)[yt]; w = w / w.mean()
    Xt = X[tr]
    m = lgb.train(PARAMS, lgb.Dataset(Xt[~v], yt[~v], weight=w[~v]), 2000,
                  valid_sets=[lgb.Dataset(Xt[v], yt[v], weight=w[v])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict(X[te], num_iteration=m.best_iteration)


def two_stage(tr, y, te):
    surv = np.ones(te.sum()); E = np.ones(te.sum())
    for k in range(1, KMAX):
        surv = surv * fit_bin(tr & (y >= k) & keep_conf, (y > k).astype(int), te)
        E += surv
    return E


keep_conf = (y_osm > 0) | ~st_num | (st_pm >= ARGS.tau)
log(f"confidence filter tau {ARGS.tau}: dropped {int((~keep_conf).sum())} VLM floor labels")
t1 = time.time()
allm = np.ones(N, bool)
E_all = two_stage(allm & (y_naive > 0), y_naive, allm)
log(f"all-label two-stage GBM with Nm: done ({time.time()-t1:.0f}s)")

# ---------------------------------------------------------------- cross-fitted GBM E for the street buildings
ucell = np.unique(cell_id)
fold_of_cell = dict(zip(ucell, np.random.default_rng(SEED).integers(0, NFOLD, len(ucell))))
fold = np.array([fold_of_cell[c] for c in cell_id])
E_oof = np.full(N, np.nan)
for k in range(NFOLD):
    t1 = time.time()
    te = (fold == k) & st_num
    E_oof[te] = two_stage((fold != k) & (y_naive > 0), y_naive, te)
    log(f"  fold {k}: predicted {int(te.sum())} street buildings ({time.time()-t1:.0f}s)")


def qmap(src, tgt):
    """rank of src within its group -> quantile of tgt (same buildings)."""
    r = pd.Series(src).rank(pct=True).to_numpy()
    return np.quantile(tgt, np.clip(r, 0, 1))


E_map = np.full(N, np.nan)
region = np.full(N, "", object)
idx_all = np.where(st_num)[0]
cnt = pd.Series(cell_id[idx_all]).value_counts()
big = set(cnt.index[cnt >= MIN_N])
for c in big:
    ii = idx_all[cell_id[idx_all] == c]
    E_map[ii] = qmap(st_E[ii], E_oof[ii]); region[ii] = "cell"
small = idx_all[~np.isin(cell_id[idx_all], list(big))]
if len(small):
    q = qmap(st_E[idx_all], E_oof[idx_all])                  # county-wide mapping, applied to the small cells
    E_map[small] = q[np.isin(idx_all, small)]; region[small] = "county"
log(f"mapping: {len(big)} cells with >= {MIN_N} street answers cover {int((region == 'cell').sum())} buildings; "
    f"{int((region == 'county').sum())} buildings in smaller cells use the county-wide mapping")

final = np.where(st_num, E_map, E_all)
out = pd.DataFrame({"building_id": bid, "E_floors_final": final, "floors_round": np.clip(np.round(final), 1, None).astype(int),
                    "src": np.where(st_num, "street_vlm_qmap", "gbm_2st_Nm"), "map_region": region,
                    "E_hybfb": np.where(st_num, st_E, E_all),
                    "E_vlm_raw": st_E, "E_gbm_oof": E_oof, "E_gbm_all": E_all})
out.to_csv(OUTD + f"county_floors_hybq_Nm{SUF}_20261011.csv.gz", index=False, float_format="%.4f")
log(f"written {OUTD}county_floors_hybq_Nm{SUF}_20261011.csv.gz ({len(out)} rows)")
log("  E_floors_final rounded distribution: " + str(out.floors_round.clip(upper=KMAX).value_counts().sort_index().to_dict()))
log("  street buildings: raw VLM E mean %.3f -> mapped %.3f (GBM OOF mean %.3f)" % (
    np.nanmean(st_E[st_num]), np.nanmean(E_map[st_num]), np.nanmean(E_oof[st_num])))

# ---------------------------------------------------------------- check 2: not held-out sanity on the county files
lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv").set_index("building_id")
area_of = pd.Series(bid).map(lid.area).to_numpy()
lid_h = pd.Series(bid).map(lid.height_p95_m.where(lid.edge_clipped == 0)).to_numpy(float)
hum = {}
for f in glob.glob("annotation_20261002/labels_dump_20261006/*/labels/*.json"):
    hum[int(f.split("/")[-1][:-5])] = json.load(open(f))["code"]
bad = np.array([hum.get(b) in ("NB", "BADFP") for b in bid])
sc = np.isin(area_of, AREAS) & ~np.isnan(lid_h) & ~bad
log(f"\n## check 2 (NOT held-out: the county GBM saw the 7 areas' OSM / VLM labels), scored n={int(sc.sum())}")
old = pd.read_csv("buildscale/floors_county_20261008/county_floors_hybq_20261008.csv.gz").set_index("building_id").reindex(bid)
for name, x in [("10-08 hybq_2st file", old.E_floors_final.to_numpy(float)), ("10-11 hybq_Nm (this file)", final),
                ("10-11 hybfb_Nm (E_hybfb)", np.where(st_num, st_E, E_all))]:
    log(f"  {name:32s} Spearman all {spearmanr(x[sc], lid_h[sc])[0]:.3f} | with SV {spearmanr(x[sc & st_num], lid_h[sc & st_num])[0]:.3f} | no SV {spearmanr(x[sc & ~st_num], lid_h[sc & ~st_num])[0]:.3f}")
log("  street VLM >= 6 floors: n=%d, hybq median %.2f, hybfb median %.2f" % ((st_E >= 6).sum(), np.median(final[st_E >= 6]), np.median(st_E[st_E >= 6])))
log(f"\ndone in {time.time()-T0:.0f}s")
