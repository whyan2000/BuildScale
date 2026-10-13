"""10-11: county-wide out-of-fold BuildScale use output, to be fed into the floors model as features (user 10-11:
"feed the use results back into the height model").
Same model as train_route_v3_20261006 (budget 1): R fitted once on the targeted labels; two-stage LightGBM
(stage 1 RES vs non-RES on all naive labels; stage 2 non-RES class, pseudo-label rows need R >= 0.5);
final = hybfb: street VLM token distribution where the VLM answered a major, else the two-stage GBM.
Out-of-fold so a building's own use label never feeds its own feature: 12 folds = the 7 areas (one fold each) +
the rest of the county split by 2 km cell into 5 folds (as osm_p_ans in cheap_signals_20261006).
Only seed 0 (one feature table for all floors seeds).
Output: buildscale/use_oof_county_20261011.csv.gz  (building_id, src, U_<major> x 8 = final, G_<major> x 8 = GBM only)
Run from the project root (CPU, ~15 min)."""
import glob
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

T0 = time.time()
SEED, TAU = 0, 0.5
MAJORS = ["RES", "COM", "IND", "AGR", "REL", "GOV", "EDU", "ACC"]
NONRES = MAJORS[1:]
SIGNALS = ["tok_use_margin", "xm_p_sat_ans", "osm_p_ans"]
AREAS = ["downtown_orlando", "industrial_orlando_central_park", "random_c2", "random_c3",
         "residential_long_lake", "tourism_universal", "ucf_main_campus"]


def log(*a):
    print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)


feat = pd.read_csv("features/building_features_county_20260922.csv.gz", low_memory=False)
feat = feat[feat.meta_in_orange == 1].reset_index(drop=True)
FCOLS = list(feat.columns[feat.columns.get_loc("f_area_m2"):feat.columns.get_loc("f_nbr500_area_rank") + 1])
bid = feat.building_id.to_numpy(); N = len(bid)
nbr = pd.read_csv("features/nbr_features_20261005.csv.gz").set_index("building_id").reindex(bid)
ACOLS = [c for c in nbr.columns if c.startswith("ctxA_")]
X = np.c_[feat[FCOLS].to_numpy(np.float32), nbr[ACOLS].to_numpy(np.float32)]
xy = feat[["f_x_utm", "f_y_utm"]].to_numpy()
cell_id = (xy[:, 0] // 2000).astype(int) * 100000 + (xy[:, 1] // 2000).astype(int)
lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv")
area_of = pd.Series(bid).map(lid.set_index("building_id").area).to_numpy()
area_of = np.where(np.isin(area_of, AREAS), area_of, None)

osm = pd.read_csv("buildscale/osm_labels_20261001_v2.csv", keep_default_na=False).set_index("building_id").reindex(bid)
osm_major = osm.use_major.fillna("").to_numpy()


def pvec(d):
    v = np.array([float((d or {}).get(k, 0.0)) for k in MAJORS])
    return v / v.sum() if v.sum() > 0 else np.full(len(MAJORS), 1 / len(MAJORS))


pos = {b: i for i, b in enumerate(bid)}
st_ans = np.full(N, "", object); st_P = np.full((N, len(MAJORS)), np.nan)
for f in sorted(glob.glob("vlm_hazus_sv_20261001_shard*.jsonl") + glob.glob("vlm_hazus_areas_20261001_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["modality"] == "street" and r["building_id"] in pos:
            i = pos[r["building_id"]]; st_ans[i] = r["use_major"]; st_P[i] = pvec(r.get("p_use"))
pl_major = np.where(np.isin(st_ans, MAJORS), st_ans, "")
st_pmax = np.where(pl_major != "", np.nanmax(np.nan_to_num(st_P, nan=0), 1), np.nan)
naive_major = np.where(osm_major != "", osm_major, pl_major)
src = np.where(osm_major != "", "osm", np.where(pl_major != "", "pl", ""))

sig = pd.read_csv("buildscale/cheap_signals_20261006.csv.gz").set_index("building_id").reindex(bid)
S = sig[SIGNALS].to_numpy(float)

# R exactly as fit_R_hard() in train_route_v3_20261006
_smp = pd.read_csv("annotation_hard_20261006/sample_hard_20261006.csv").set_index("building_id")
_lab = {int(f.split("/")[-1][:-5]): json.load(open(f)) for f in glob.glob("annotation_hard_20261006/labels_dump_20261006/labels/*.json")}
_ok = [b for b in _smp.index if b in _lab and _lab[b]["code"] not in ("NB", "BADFP", "UNK")]
hard_ref = {b: ("COM" if _lab[b]["major"] == "IND" else _lab[b]["major"]) for b in _ok}
hard_w = {b: 1 / _smp.pi[b] for b in _ok}
R = np.full(N, np.nan)
hm = np.array([b in hard_ref for b in bid]) & (pl_major != "")
hy = np.array([pl_major[i] == hard_ref[b] if hm[i] else False for i, b in enumerate(bid)]).astype(int)
hw = np.array([hard_w.get(b, 0.0) for b in bid])
for grp, gm in [("RES", pl_major == "RES"), ("nonRES", np.isin(pl_major, NONRES))]:
    tr = hm & gm
    Xs = S.copy(); mu = np.nanmean(Xs[tr], 0); sd = np.nanstd(Xs[tr], 0) + 1e-9
    Xs = np.where(np.isnan(Xs), mu, Xs); Xs = (Xs - mu) / sd
    lr = LogisticRegression(C=1.0).fit(Xs[tr], hy[tr], sample_weight=hw[tr] / hw[tr].mean())
    R[gm] = lr.predict_proba(Xs[gm])[:, 1]
    log(f"R[{grp}] fitted on {int(tr.sum())} targeted labels ({int((hy[tr] == 0).sum())} errors)")

PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=SEED)


def fit_predict(tr, y, nc, te):
    ug = np.unique(cell_id[tr])
    val_cells = set(np.random.RandomState(SEED).choice(ug, max(1, len(ug) // 10), replace=False))
    v = np.array([g in val_cells for g in cell_id[tr]])
    cnt = np.bincount(y[tr], minlength=nc).astype(float)
    w = (np.where(cnt > 0, cnt, 1) ** -0.5)[y[tr]]; w = w / w.mean()
    Xt, yt = X[tr], y[tr]
    obj = dict(objective="binary") if nc == 2 else dict(objective="multiclass", num_class=nc)
    m = lgb.train(dict(PARAMS, **obj), lgb.Dataset(Xt[~v], yt[~v], weight=w[~v]), 2000,
                  valid_sets=[lgb.Dataset(Xt[v], yt[v], weight=w[v])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = m.predict(X[te], num_iteration=m.best_iteration)
    return np.c_[1 - p, p] if nc == 2 else p


def enc(lab, cls):
    idx = {c: i for i, c in enumerate(cls)}
    return np.array([idx.get(v, -1) for v in lab], dtype=int)


y_naive = enc(naive_major, MAJORS)
y_bin = np.where(y_naive < 0, -1, (y_naive > 0).astype(int))
y_nr = enc(naive_major, NONRES)
keep_R = ~((src == "pl") & (y_nr >= 0)) | (R >= TAU)

# 12 folds: 7 areas + 5 cell folds of the rest
fold = np.full(N, -1)
for k, a in enumerate(AREAS):
    fold[area_of == a] = k
rest = fold < 0
ucell = np.unique(cell_id[rest])
cfold = dict(zip(ucell, np.random.RandomState(SEED).permutation(len(ucell)) % 5))
fold[rest] = [7 + cfold[c] for c in cell_id[rest]]
G = np.zeros((N, len(MAJORS)))
for k in range(12):
    t1 = time.time()
    te = fold == k; out = ~te
    p1 = fit_predict(out & (y_bin >= 0), y_bin, 2, te)[:, 1]
    p2 = fit_predict(out & (y_nr >= 0) & keep_R, y_nr, len(NONRES), te)
    G[te] = np.c_[1 - p1, p1[:, None] * p2]
    log(f"fold {k}: {int(te.sum())} buildings ({time.time()-t1:.0f}s)")
U = np.where((pl_major != "")[:, None], np.nan_to_num(st_P, nan=1 / 8), G)
df = pd.DataFrame({"building_id": bid, "src": np.where(pl_major != "", "street_vlm", "gbm_2st_R")})
for j, c in enumerate(MAJORS):
    df[f"U_{c}"] = U[:, j]
for j, c in enumerate(MAJORS):
    df[f"G_{c}"] = G[:, j]
df.to_csv("buildscale/use_oof_county_20261011.csv.gz", index=False, float_format="%.4f")
log("final argmax distribution:", pd.Series(np.array(MAJORS)[U.argmax(1)]).value_counts().to_dict())
log(f"done in {time.time()-T0:.0f}s")
