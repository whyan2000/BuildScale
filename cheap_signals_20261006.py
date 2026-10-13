# 10-06 copy: human labels -> labels_dump_20261006 (all human IND relabelled COM, user decision 10-06); outputs renamed *_20261006, 10-05 files untouched.
"""BuildScale C1 (10-05): cheap reliability signals -- no new VLM inference.

For every building with a street-view pseudo-label (867636 + 867633 street answers, Hazus 3-bin run of
10-01) compute candidate reliability signals, then score how well each one separates correct from wrong
pseudo-labels on the human-labelled buildings of the 7 evaluation areas (random_c1 dropped, 10-05).

Signals (column prefix -> meaning; "higher = more reliable" unless noted in DIRECTION):
  tok_*     token confidence of the street answer: use p_max / margin (top1-top2) / entropy (over the 8
            majors, "unseen" removed and renormalised); height p_max / margin / entropy (L/M/H); sub p_max
  conf_*    split-conformal set size for the use major (alpha 0.1), calibrated on the human labels of the
            OTHER areas (leave-one-area-out); county buildings outside the areas use all 7 areas
  xm_*      street vs satellite (867636 / 867633 sat answers), use only (satellite cannot judge height,
            10-02): agree (argmax), p_sat(street answer), JS divergence
  osm_*     agreement with an OSM-only LightGBM (geometry + group A context features, labels = OSM only)
            that never saw the building: 12 folds = the 7 areas + 5 folds of 2 km cells for the rest;
            osm_p_ans = P_osm(street answer), osm_agree = argmax agreement
  nb_*      neighbourhood agreement (100 m, self excluded): share of neighbours' street pseudo-labels with
            the same major; share of neighbours' OSM labels with the same major
  meta_*    capture metadata: camera distance (m), ray occlusion share (09-23), readable sign
Also: propensity P(has street view | geometry + group A), out-of-fold on the same 12 folds, for ALL
in-county buildings (for selection-bias weighting later; not a reliability signal).
Evaluation: AUROC of each signal for "pseudo-label correct" vs human labels (major; fine where both a human
fine label and a VLM fine class exist; major restricted to buildings the VLM called non-residential).
Outputs: buildscale/cheap_signals_20261006.csv.gz (pseudo-labelled buildings), features/propensity_sv_20261006.csv.gz,
         buildscale/cheap_signals_eval_20261006.txt
Run from the project root.
"""
import glob
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

T0 = time.time()
MAJORS = ["RES", "COM", "IND", "AGR", "REL", "GOV", "EDU", "ACC"]
HB = ["L", "M", "H"]
AREAS = ["downtown_orlando", "industrial_orlando_central_park", "random_c2", "random_c3",
         "residential_long_lake", "tourism_universal", "ucf_main_campus"]
ALPHA, SEED, R_NB = 0.1, 0, 100.0
OUT_TXT = "buildscale/cheap_signals_eval_20261006.txt"
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time()-T0:5.0f}s] {s}", flush=True)
    lines.append(s)


def norm(d, keys):
    v = np.array([float((d or {}).get(k, 0.0)) for k in keys])
    return v / v.sum() if v.sum() > 0 else np.full(len(keys), np.nan)


def ent(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(-1)


def top2(p):
    s = np.sort(p, -1)
    return s[..., -1], s[..., -1] - s[..., -2]


# ---------------------------------------------------------------- inputs
feat = pd.read_csv("features/building_features_county_20260922.csv.gz", low_memory=False)
feat = feat[feat.meta_in_orange == 1].reset_index(drop=True)
FCOLS = list(feat.columns[feat.columns.get_loc("f_area_m2"):feat.columns.get_loc("f_nbr500_area_rank") + 1])
bid = feat.building_id.to_numpy(); N = len(bid); row = {b: i for i, b in enumerate(bid)}
nbr = pd.read_csv("features/nbr_features_20261005.csv.gz").set_index("building_id").reindex(bid)
ACOLS = [c for c in nbr.columns if c.startswith("ctxA_")]
X = np.c_[feat[FCOLS].to_numpy(np.float32), nbr[ACOLS].to_numpy(np.float32)]
xy = feat[["f_x_utm", "f_y_utm"]].to_numpy()
has_sv = feat.meta_has_streetview_match.to_numpy() == 1

lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv")
area_of = pd.Series(bid).map(lid.set_index("building_id").area).to_numpy()
area_of = np.where(np.isin(area_of, AREAS), area_of, None)

recs = {"street": {}, "sat": {}}
for f in sorted(glob.glob("vlm_hazus_sv_20261001_shard*.jsonl") + glob.glob("vlm_hazus_areas_20261001_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        recs[r["modality"]][r["building_id"]] = r
st = {b: r for b, r in recs["street"].items() if b in row}
log(f"street pseudo-labels in county: {len(st)}; satellite answers: {len(recs['sat'])}")

# ---------------------------------------------------------------- OSM-only model + propensity, 12 folds
lab = pd.read_csv("buildscale/osm_labels_20261001_v2.csv", keep_default_na=False).set_index("building_id").reindex(bid)
y_osm = np.array([MAJORS.index(v) if v in MAJORS else -1 for v in lab.use_major.fillna("")])
cell = (xy[:, 0] // 2000).astype(int) * 100000 + (xy[:, 1] // 2000).astype(int)
ucell = np.unique(cell)
cfold = dict(zip(ucell, np.random.RandomState(SEED).randint(0, 5, len(ucell))))
fold = np.array([AREAS.index(a) if a is not None else 7 + cfold[c] for a, c in zip(area_of, cell)])
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=SEED)


def oof(y, nc, weight_balanced=True):
    P = np.zeros((N, nc))
    for k in range(12):
        te = fold == k
        tr = (~te) & (y >= 0)
        cnt = np.bincount(y[tr], minlength=nc).astype(float)
        w = (np.where(cnt > 0, cnt, 1) ** -0.5)[y[tr]] if weight_balanced else np.ones(tr.sum())
        obj = dict(objective="multiclass", num_class=nc) if nc > 2 else dict(objective="binary")
        m = lgb.train(dict(PARAMS, **obj), lgb.Dataset(X[tr], y[tr], weight=w / w.mean()), 400)
        p = m.predict(X[te])
        P[te] = p if nc > 2 else np.c_[1 - p, p]
    return P


t1 = time.time()
P_osm = oof(y_osm, len(MAJORS))
log(f"OSM-only OOF model (12 folds, 400 trees) {time.time()-t1:.0f}s")
t1 = time.time()
P_sv = oof(has_sv.astype(int), 2, weight_balanced=False)[:, 1]
pd.DataFrame({"building_id": bid, "p_has_sv": P_sv.astype(np.float32)}).to_csv(
    "features/propensity_sv_20261006.csv.gz", index=False, float_format="%.4f")
log(f"propensity OOF {time.time()-t1:.0f}s; AUROC(has_sv) {roc_auc_score(has_sv, P_sv):.3f}; "
    f"mean p in SV {P_sv[has_sv].mean():.3f} / non-SV {P_sv[~has_sv].mean():.3f}")

# ---------------------------------------------------------------- per pseudo-label signals
man = {json.loads(l)["building_id"]: json.loads(l) for l in open("all_crops_v3/manifest.jsonl")}
occ = pd.read_csv("occlusion_v3_20260923.csv").set_index("building_id").occlusion
rows = []
for b, r in st.items():
    i = row[b]
    pu = norm(r.get("p_use"), MAJORS); ph = norm(r.get("p_height"), HB)
    s = recs["sat"].get(b); ps = norm(s.get("p_use") if s else None, MAJORS)
    ans = r["use_major"] if r["use_major"] in MAJORS else None
    d = dict(building_id=b, area=area_of[i], ans_major=ans or r["use_major"], ans_fine=r.get("use_fine") or "",
             ans_height=r["height"],
             tok_use_pmax=top2(pu)[0], tok_use_margin=top2(pu)[1], tok_use_entropy=ent(pu),
             tok_h_pmax=top2(ph)[0], tok_h_margin=top2(ph)[1], tok_h_entropy=ent(ph),
             tok_sub_pmax=max((r.get("p_sub") or {"_": np.nan}).values()),
             tok_pmass_use=r.get("pmass_use"),
             meta_dist_m=man.get(b, {}).get("distance_m", np.nan), meta_occlusion=occ.get(b, np.nan),
             meta_sign=float((r.get("sign") or "none").strip().lower() not in ("none", "")))
    if ans:
        j = MAJORS.index(ans)
        d["xm_agree"] = float(np.nanargmax(ps) == j) if np.isfinite(ps).all() else np.nan
        d["xm_p_sat_ans"] = ps[j]
        m = 0.5 * (pu + ps)
        d["xm_js"] = 0.5 * ((pu * np.log(np.clip(pu, 1e-12, 1) / np.clip(m, 1e-12, 1))).sum() +
                            (ps * np.log(np.clip(ps, 1e-12, 1) / np.clip(m, 1e-12, 1))).sum())
        d["osm_p_ans"] = P_osm[i, j]
        d["osm_agree"] = float(P_osm[i].argmax() == j)
    rows.append(d)
S = pd.DataFrame(rows).set_index("building_id")
S["pu"] = [norm(st[b].get("p_use"), MAJORS) for b in S.index]

# neighbourhood agreement (100 m, self excluded)
pl_idx = np.array([row[b] for b in S.index])
pl_ans = S.ans_major.to_numpy()
tree_pl = cKDTree(xy[pl_idx])
nb_same, nb_n = np.full(len(S), np.nan), np.zeros(len(S))
for k, nbrs in enumerate(tree_pl.query_ball_point(xy[pl_idx], R_NB, workers=8)):
    nbrs = [q for q in nbrs if q != k]
    nb_n[k] = len(nbrs)
    if nbrs:
        nb_same[k] = np.mean(pl_ans[nbrs] == pl_ans[k])
S["nb_pl_same"] = nb_same; S["nb_pl_n"] = nb_n
um = lab.use_major.fillna("").to_numpy()
osm_lab = np.where(um != "")[0]
tree_osm = cKDTree(xy[osm_lab])
osm_same = np.full(len(S), np.nan)
for k, nbrs in enumerate(tree_osm.query_ball_point(xy[pl_idx], R_NB, workers=8)):
    nbrs = [osm_lab[q] for q in nbrs if osm_lab[q] != pl_idx[k]]
    if nbrs:
        osm_same[k] = np.mean(um[nbrs] == pl_ans[k])
S["nb_osm_same"] = osm_same
log(f"signals computed for {len(S)} pseudo-labels")

# ---------------------------------------------------------------- human labels + conformal (LOAO)
hum = []
for f in glob.glob("annotation_20261002/labels_dump_20261006/*/labels/*.json"):
    d = json.load(open(f))
    hum.append(dict(building_id=int(f.split("/")[-1][:-5]), h_code=d["code"], h_major=d["major"]))
hum = pd.DataFrame(hum).set_index("building_id")
S = S.join(hum)
ok = S.h_major.notna() & ~S.h_code.isin(["UNK", "NB", "BADFP"]) & S.ans_major.isin(MAJORS)
S["correct_major"] = np.where(ok, (S.ans_major == S.h_major).astype(float), np.nan)
fine_h = ~S.h_code.isin(["COM", "GOV", "IND", "UNK", "NB", "BADFP"]) & S.h_code.notna()
fine_v = S.ans_fine.ne("") | S.ans_major.eq("ACC")
vf = np.where(S.ans_major == "ACC", "ACC", S.ans_fine)
S["correct_fine"] = np.where(fine_h & fine_v & ok, (vf == S.h_code).astype(float), np.nan)


def conformal_size(cal_mask, apply_mask):
    pu = np.stack(S.loc[cal_mask, "pu"].values)
    y = np.array([MAJORS.index(v) for v in S.loc[cal_mask, "h_major"]])
    sc = 1 - pu[np.arange(len(y)), y]
    n = len(sc)
    q = np.quantile(sc, min(1, np.ceil((n + 1) * (1 - ALPHA)) / n), method="higher")
    pa = np.stack(S.loc[apply_mask, "pu"].values)
    return (1 - pa <= q).sum(1), q


S["conf_setsize"] = np.nan
cal_all = ok
for a in AREAS:
    am = S.area == a
    if am.sum():
        S.loc[am, "conf_setsize"], q = conformal_size(ok & ~am, am)
rest = S.area.isna()
S.loc[rest, "conf_setsize"], q_all = conformal_size(ok, rest)
log(f"conformal alpha={ALPHA}: q (all areas) {q_all:.3f}; set size distribution (county) "
    + " ".join(f"{k}:{v}" for k, v in S.conf_setsize.value_counts().sort_index().items()))

# ---------------------------------------------------------------- evaluation
SIG = ["tok_use_pmax", "tok_use_margin", "tok_use_entropy", "tok_h_pmax", "tok_h_margin", "tok_sub_pmax",
       "conf_setsize", "xm_agree", "xm_p_sat_ans", "xm_js", "osm_p_ans", "osm_agree",
       "nb_pl_same", "nb_osm_same", "meta_dist_m", "meta_occlusion", "meta_sign"]
LOWER_BETTER = {"tok_use_entropy", "conf_setsize", "xm_js", "meta_dist_m", "meta_occlusion"}


def auroc_table(target, mask, tag):
    m = mask & S[target].notna()
    y = S.loc[m, target].to_numpy()
    log(f"\n## {tag}: n={m.sum()}, correct {int(y.sum())}, wrong {int((1-y).sum())} "
        f"(accuracy {y.mean():.3f})")
    if len(set(y)) < 2:
        log("  only one class, skipped"); return
    for s in SIG:
        v = S.loc[m, s].to_numpy(float)
        f = np.isfinite(v)
        if f.sum() < 20 or len(set(y[f])) < 2 or np.nanstd(v) == 0:
            log(f"  {s:16s} n/a (n={f.sum()})"); continue
        sgn = -1 if s in LOWER_BETTER else 1
        au = roc_auc_score(y[f], sgn * v[f])
        rng = np.random.default_rng(SEED); bs = []
        for _ in range(500):
            i = rng.integers(0, f.sum(), f.sum())
            if len(set(y[f][i])) == 2:
                bs.append(roc_auc_score(y[f][i], sgn * v[f][i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        log(f"  {s:16s} AUROC {au:.3f} [{lo:.3f},{hi:.3f}]  n={f.sum()}"
            + ("" if f.all() else f" (missing {int((~f).sum())})"))


in_area = S.area.notna()
auroc_table("correct_major", in_area, "use major, all human-labelled pseudo-labels (7 areas)")
auroc_table("correct_major", in_area & (S.ans_major != "RES"), "use major, VLM said non-RES")
auroc_table("correct_major", in_area & (S.ans_major == "RES"), "use major, VLM said RES")
auroc_table("correct_fine", in_area, "use fine (VLM gave a fine class and human has one)")

log("\n## Confusion: VLM street major (rows) vs human major (cols), 7 areas")
c = pd.crosstab(S.loc[in_area & ok, "ans_major"], S.loc[in_area & ok, "h_major"])
log("  " + c.to_string().replace("\n", "\n  "))
log("\n## Correct rate by conformal set size (7 areas)")
log("  " + S.loc[in_area & ok].groupby("conf_setsize").correct_major.agg(["size", "mean"]).to_string().replace("\n", "\n  "))

S.drop(columns=["pu"]).to_csv("buildscale/cheap_signals_20261006.csv.gz", float_format="%.5g")
open(OUT_TXT, "w").write("\n".join(lines) + "\n")
log(f"wrote buildscale/cheap_signals_20261006.csv.gz, {OUT_TXT}; done")
