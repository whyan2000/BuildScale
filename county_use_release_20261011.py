"""10-11: county-wide BuildScale use prediction (10-08 gap 1: train_route_v3 only wrote 7-area predictions) and the merged
release table (use + floors). Copy of use_oof_county_20261011.py with the 12-fold loop replaced by ONE all-label fit
(same model as the paper: R from targeted labels; two-stage LightGBM, stage 2 drops pseudo-labels with R < 0.5; seed 0).
Final use (= BuildScale inference, hybfb_2st_R): street VLM major where it answered a major, else the two-stage GBM.
Release table: OSM first (10-01 principle: label enhancement = completion, OSM is never overwritten), then the model.
  use_major  = OSM use_major > BuildScale (VLM > GBM);   use_src  = osm / street_vlm / gbm
  floors     = OSM building:levels (>= 1) > E_hybfb of floors_county_20261011/county_floors_hybq_Nm_fall_20261011.csv.gz
               (user 10-11: street VLM raw value kept, else two-stage GBM with masked neighbour heights + p_max filter)
  floors_src = osm / street_vlm / gbm;  plus model probabilities P_<major>, R of the street pseudo-label, raw columns.
Outputs: buildscale/dataset_20261011/{county_use_buildscale_20261011.csv.gz, orange_buildings_use_floors_20261011.csv.gz, log_20261011.txt}
Run from the project root."""
import glob
import os
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


LOGF = []


def log(*a):
    s = " ".join(str(x) for x in a); LOGF.append(s)
    print(f"[{time.time()-T0:5.0f}s]", s, flush=True)
    os.makedirs("buildscale/dataset_20261011/", exist_ok=True)
    open("buildscale/dataset_20261011/log_20261011.txt", "w").write("\n".join(LOGF) + "\n")


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

allm = np.ones(N, bool)
t1 = time.time()
p1 = fit_predict(allm & (y_bin >= 0), y_bin, 2, allm)[:, 1]
p2 = fit_predict(allm & (y_nr >= 0) & keep_R, y_nr, len(NONRES), allm)
G = np.c_[1 - p1, p1[:, None] * p2]
log(f"all-label two-stage GBM: {time.time()-t1:.0f}s; stage-2 rows kept {int((allm & (y_nr >= 0) & keep_R).sum())} of {int((y_nr >= 0).sum())}")
U = np.where((pl_major != "")[:, None], np.nan_to_num(st_P, nan=1 / 8), G)
import os
OUT = "buildscale/dataset_20261011/"
os.makedirs(OUT, exist_ok=True)
use_model = np.array(MAJORS)[U.argmax(1)]
df = pd.DataFrame({"building_id": bid, "use_model": use_model,
                   "use_model_src": np.where(pl_major != "", "street_vlm", "gbm"), "R_street": R})
for j, c in enumerate(MAJORS):
    df[f"P_{c}"] = U[:, j]
df.to_csv(OUT + "county_use_buildscale_20261011.csv.gz", index=False, float_format="%.4f")
log("model use distribution: " + str(pd.Series(use_model).value_counts().to_dict()))

fl = pd.read_csv("buildscale/floors_county_20261011/county_floors_hybq_Nm_fall_20261011.csv.gz").set_index("building_id").reindex(bid)
lv = osm.osm_levels.replace("", np.nan).astype(float).to_numpy()
has_lv = ~np.isnan(lv) & (lv >= 1)
rel = pd.DataFrame({"building_id": bid,
                    "use_major": np.where(osm_major != "", osm_major, use_model),
                    "use_src": np.where(osm_major != "", "osm", np.where(pl_major != "", "street_vlm", "gbm")),
                    "floors": np.where(has_lv, lv, fl.E_hybfb.to_numpy(float)),
                    "floors_src": np.where(has_lv, "osm", np.where(fl.src.to_numpy() == "street_vlm_qmap", "street_vlm", "gbm")),
                    "use_model": use_model, "floors_model": fl.E_hybfb.to_numpy(float),
                    "osm_use_major": osm_major, "osm_levels": lv})
rel["floors_round"] = np.clip(np.round(rel.floors), 1, None).astype(int)
assert rel.floors.notna().all() and (rel.use_major != "").all()
rel.to_csv(OUT + "orange_buildings_use_floors_20261011.csv.gz", index=False, float_format="%.3f")
log(f"release table {len(rel)} rows")
log("  use_src: " + str(rel.use_src.value_counts().to_dict()) + " | floors_src: " + str(rel.floors_src.value_counts().to_dict()))
log("  use_major: " + str(rel.use_major.value_counts().to_dict()))
log("  floors_round (6+ pooled): " + str(rel.floors_round.clip(upper=6).value_counts().sort_index().to_dict()))
log(f"  max floors {rel.floors.max():.1f}; buildings >= 4 floors {int((rel.floors >= 3.5).sum())}")
for c in ["use", "floors"]:
    m = rel[f"{c}_src"] == "osm"
    agree = (rel.use_model[m] == rel.use_major[m]).mean() if c == "use" else np.corrcoef(rel.floors_model[m], rel.floors[m])[0, 1]
    log(f"  where OSM has {c}: model agrees/corr with OSM {agree:.3f} (n={int(m.sum())}; not a held-out test, model saw these labels)")
log(f"done in {time.time()-T0:.0f}s")
