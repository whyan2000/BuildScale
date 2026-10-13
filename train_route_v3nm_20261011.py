"""10-11: train_route_v3_20261006.py + neighbour-USE features with neighbour masking (as Nm for floors, which raised
no-street floors Spearman 0.490 -> 0.543). New group B' (--feats):
  known use = OSM use_major else street VLM major (not "unseen"); over the 50 nearest OTHER known buildings:
  distance to nearest; share of each of the 8 majors among k10 and within 250 m (+ count within 250 m);
  inverse-distance-weighted k10 shares.
  A    = the 10-06 main line (geometry + group A)       -> should reproduce route_v3_20261006 at budget 1
  AB   = + B' unmasked
  ABm  = + B' masked: for labelled rows, known neighbours closer than r_i are hidden, r_i drawn from the unlabelled
         buildings' nearest-known distance (labelled rows otherwise see neighbours from the same street/panorama).
Budget 1 only. Outputs: --outd (default buildscale/route_v3nm_20261011/<feats>/).
Original v3 docstring follows.
v3 (10-06 evening): R is fitted ONCE on the 599 targeted labels outside the 7 areas
(annotation_hard_20261006; 574 usable, 222 VLM errors), with 1/pi weights, separately for VLM RES / non-RES answers,
on the 3 cheap signals. No evaluation-area label is used for R any more, so R is the same in every fold.
At budget b < 1 the targeted labels are kept only for called buildings (labels need a VLM answer).
Everything else as v2 below. Outputs: buildscale/route_v3_20261006/.

v2 (10-06 afternoon) of train_route_20261006.py -- changes, after the 5-seed v1 results:
  * non-RES R = Platt calibration of the token p_max only (v1's 3-signal model did not transfer to held-out
    areas: AUROC 0.666 < raw p_max 0.719); RES R unchanged (3 signals). A group with < 10 rows or a single
    outcome falls back to R = p_max.
  * new routing cmp_<gbm>: VLM answer if R >= the GBM's own max probability, else the GBM (v1's route_ with a
    fixed TAU sent low-R minority-class answers to a GBM that never predicts those classes -> macro-F1 dropped).
  * --budget b: VLM-call budget experiment, simulated offline. A random share b (seeded) of the buildings with a
    street-view answer counts as "called"; uncalled buildings have no pseudo-label, no signals and no VLM answer
    at test time (also inside the 7 areas). R is fitted only on called, human-labelled buildings. Cost = calls
    (street + satellite answer per called building), not kept rows. b = 1 reproduces the full setting.
Outputs: buildscale/route_v2_20261006/{eval,metrics,preds}_b<budget>_s<seed>.*
Original v1 docstring follows.

BuildScale routing (10-06, user decision): reliability R decides WHEN TO TRUST THE VLM, not how to weight training.

Use major only (6 human classes after IND->COM). Leave-one-area-out over the 7 areas; for each held-out area a:
  R    class-conditional reliability P(street pseudo-label correct | signals), fitted on the human labels of the
       OTHER 6 areas only (nested -> no leakage into the scored area). Two logistic models by the VLM's predicted
       group (RES / non-RES; non-RES pooled + class indicators would overfit 159 rows / 30 errors).
       Signals (cheap, no new inference; buildscale/cheap_signals_20261006.csv.gz):
         tok_use_margin, xm_p_sat_ans, osm_p_ans  (osm_p_ans is out-of-fold w.r.t. the 7 areas)
       Applied to every street pseudo-label in the county.
  GBM  same LightGBM / features (geometry + group A) / class weights / early stopping as train_ctx_20261005:
         osmonly_A      single stage, OSM labels
         naive_A        single stage, OSM else street pseudo-label
         2st_nofilt     stage 1 RES vs non-RES on the naive labels (all pseudo-labels);
                        stage 2 non-RES class among non-RES naive labels;  P = [p1 | (1-p1) * p2]
         2st_pmax       stage 2 drops non-RES pseudo-label rows with low token p_max, keeping exactly as many
                        rows as 2st_R keeps (same-budget control)
         2st_R          stage 2 drops non-RES pseudo-label rows with R < TAU
  Routing (buildings with a street-view answer only; everything else = the GBM):
         hybfb_<gbm>    street VLM answer unless it is "unseen" (10-06 hybrid+fb)
         route_<gbm>    street VLM answer if R >= TAU, else the GBM
       Probability vector of a routed VLM answer: R on the VLM class, (1 - R) spread over the other classes
       in proportion to the GBM probabilities. hybfb uses the VLM's own token distribution.
  full VLM / sat-only: VLM token distribution over the 8 majors ("unseen" answers stay errors).
TAU = 0.5 fixed in advance ("more likely right than wrong"); a TAU sweep is logged as a risk-coverage table.
Metrics on all scored buildings (human major, NB/BADFP/UNK excluded): macro-F1 (classes present), accuracy,
top-label ECE (15 equal-width bins), multiclass Brier (8 majors), accuracy at confidence >= 0.9 + its coverage.
Paired bootstrap (1,000, same resamples) of macro-F1 differences.
Outputs: buildscale/route_20261006/{eval_s<seed>.txt, metrics_s<seed>.csv, preds_s<seed>.csv.gz}
Run from the project root:  python buildscale/train_route_20261006.py --seed 0
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

T0 = time.time()
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--budget", type=float, default=1.0)
ap.add_argument("--feats", default="A", choices=["A", "AB", "ABm"]); ap.add_argument("--outd", default="")
ARGS = ap.parse_args()
OUTD = ARGS.outd or f"buildscale/route_v3nm_20261011/{ARGS.feats}/"
SFX = f"b{ARGS.budget:g}_s{ARGS.seed}"
os.makedirs(OUTD, exist_ok=True)
TAU, NBOOT, NBIN = 0.5, 1000, 15
MAJORS = ["RES", "COM", "IND", "AGR", "REL", "GOV", "EDU", "ACC"]
NONRES = MAJORS[1:]
SIGNALS = ["tok_use_margin", "xm_p_sat_ans", "osm_p_ans"]
AREAS = ["downtown_orlando", "industrial_orlando_central_park", "random_c2", "random_c3",
         "residential_long_lake", "tourism_universal", "ucf_main_campus"]
LOG = OUTD + f"eval_{SFX}.txt"
open(LOG, "w").close()


def log(*a):
    s = " ".join(str(x) for x in a)
    print(f"[{time.time()-T0:5.0f}s] {s}", flush=True)
    with open(LOG, "a") as f:
        print(s, file=f)


# ---------------------------------------------------------------- data (as train_weighted_seeds_20261006)
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
in_area = pd.notna(area_of)

osm = pd.read_csv("buildscale/osm_labels_20261001_v2.csv", keep_default_na=False).set_index("building_id").reindex(bid)
osm_major = osm.use_major.fillna("").to_numpy()


def pvec(d):
    v = np.array([float((d or {}).get(k, 0.0)) for k in MAJORS])
    return v / v.sum() if v.sum() > 0 else np.full(len(MAJORS), 1 / len(MAJORS))


street, sat = {}, {}          # building_id -> (answer, prob vector over MAJORS)
for f in sorted(glob.glob("vlm_hazus_sv_20261001_shard*.jsonl") + glob.glob("vlm_hazus_areas_20261001_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        (street if r["modality"] == "street" else sat)[r["building_id"]] = (r["use_major"], pvec(r.get("p_use")))
pos = {b: i for i, b in enumerate(bid)}
st_ans = np.full(N, "", object); st_P = np.full((N, len(MAJORS)), np.nan)
sa_ans = np.full(N, "missing", object); sa_P = np.full((N, len(MAJORS)), 1 / len(MAJORS))
for b, (a, p) in street.items():
    if b in pos:
        st_ans[pos[b]] = a; st_P[pos[b]] = p
for b, (a, p) in sat.items():
    if b in pos:
        sa_ans[pos[b]] = a; sa_P[pos[b]] = p
has_st_all = st_ans != ""
st_ans_all = st_ans.copy()                  # full VLM / sat-only reference points ignore the budget
called = has_st_all & (np.random.default_rng(500 + ARGS.seed).random(N) < ARGS.budget)
st_ans = np.where(called, st_ans, "")
has_st = called
pl_major = np.where(np.isin(st_ans, MAJORS), st_ans, "")
st_pmax = np.where(called, np.nanmax(np.where(called[:, None], st_P, 0), 1), np.nan)
COST_S = dict(street=1.93, sat=1.41)        # median seconds per answer on one H100 (867636)
log_cost = (f"budget {ARGS.budget:g}: called {int(called.sum())} / {int(has_st_all.sum())} street-view buildings "
            f"= {called.sum() / N * 100:.2f}% of the county; est. {called.sum() * (COST_S['street'] + COST_S['sat']) / 3600:.1f} GPU h")

naive_major = np.where(osm_major != "", osm_major, pl_major)
src = np.where(osm_major != "", "osm", np.where(pl_major != "", "pl", ""))

sig = pd.read_csv("buildscale/cheap_signals_20261006.csv.gz").set_index("building_id").reindex(bid)
S = sig[SIGNALS].to_numpy(float)
assert (sig.ans_major.fillna("").to_numpy()[pl_major != ""] == pl_major[pl_major != ""]).all(), "signals / answers mismatch"

hum = []
for f in glob.glob("annotation_20261002/labels_dump_20261006/*/labels/*.json"):
    d = json.load(open(f))
    hum.append(dict(building_id=int(f.split("/")[-1][:-5]), code=d["code"], major=d["major"]))
hum = pd.DataFrame(hum).set_index("building_id").reindex(bid)
ref = hum.major.where(~hum.code.isin(["UNK", "NB", "BADFP"])).to_numpy()
scored = in_area & pd.notna(ref)
log(f"# train_route_v3nm_20261011.py feats {ARGS.feats} seed {ARGS.seed}  budget {ARGS.budget:g}  TAU {TAU}  RES signals {SIGNALS}, non-RES logit(p_max)")
log(log_cost)
log(f"in-county {N}; street pseudo-labels {int((pl_major != '').sum())}; scored human labels {int(scored.sum())} "
    f"(with street answer {int((scored & has_st).sum())})")


# ---------------------------------------------------------------- R: class-conditional, nested leave-one-area-out
def fit_R(train_mask):
    """two logistic models (VLM said RES / non-RES); returns R for every street pseudo-label in the county."""
    R = np.full(N, np.nan)
    for grp, gm in [("RES", pl_major == "RES"), ("nonRES", np.isin(pl_major, NONRES))]:
        tr = train_mask & gm & scored & (pl_major != "")
        y = (pl_major[tr] == ref[tr]).astype(int)
        if tr.sum() < 10 or y.min() == y.max():
            R[gm] = st_pmax[gm]; continue           # too few called + labelled buildings: raw p_max
        if grp == "RES":
            Xs = S.copy()
        else:                                       # v2: Platt on the token p_max only
            pm = np.clip(st_pmax, 1e-4, 1 - 1e-4)
            Xs = np.log(pm / (1 - pm))[:, None]
        mu = np.nanmean(Xs[tr], 0); sd = np.nanstd(Xs[tr], 0) + 1e-9
        Xs = np.where(np.isnan(Xs), mu, Xs); Xs = (Xs - mu) / sd
        lr = LogisticRegression(C=1.0).fit(Xs[tr], y)
        R[gm] = lr.predict_proba(Xs[gm])[:, 1]
    return R


# v3: R from the targeted labels (outside all areas), 1/pi weights, 3 signals for both groups
_smp = pd.read_csv("annotation_hard_20261006/sample_hard_20261006.csv").set_index("building_id")
_lab = {int(f.split("/")[-1][:-5]): json.load(open(f)) for f in glob.glob("annotation_hard_20261006/labels_dump_20261006/labels/*.json")}
_ok = [b for b in _smp.index if b in _lab and _lab[b]["code"] not in ("NB", "BADFP", "UNK")]
hard_ref = {b: ("COM" if _lab[b]["major"] == "IND" else _lab[b]["major"]) for b in _ok}
hard_w = {b: 1 / _smp.pi[b] for b in _ok}


def fit_R_hard():
    R = np.full(N, np.nan)
    hm = np.array([b in hard_ref for b in bid]) & (pl_major != "")
    hy = np.array([pl_major[i] == hard_ref[b] if hm[i] else False for i, b in enumerate(bid)]).astype(int)
    hw = np.array([hard_w.get(b, 0.0) for b in bid])
    for grp, gm in [("RES", pl_major == "RES"), ("nonRES", np.isin(pl_major, NONRES))]:
        tr = hm & gm
        if tr.sum() < 10 or hy[tr].min() == hy[tr].max():
            R[gm] = st_pmax[gm]; continue
        Xs = S.copy()
        mu = np.nanmean(Xs[tr], 0); sd = np.nanstd(Xs[tr], 0) + 1e-9
        Xs = np.where(np.isnan(Xs), mu, Xs); Xs = (Xs - mu) / sd
        lr = LogisticRegression(C=1.0).fit(Xs[tr], hy[tr], sample_weight=hw[tr] / hw[tr].mean())
        R[gm] = lr.predict_proba(Xs[gm])[:, 1]
        log(f"R[{grp}] fitted on {int(tr.sum())} targeted labels ({int((hy[tr] == 0).sum())} errors)")
    return R


R_all = fit_R_hard()
R_oof = np.where(in_area, R_all, np.nan)
R_fold = {a: R_all for a in AREAS}

log("\n## R calibration on the scored street pseudo-labels (out-of-area R vs raw token p_max)")
for grp, gm in [("all", pl_major != ""), ("VLM RES", pl_major == "RES"), ("VLM non-RES", np.isin(pl_major, NONRES))]:
    m = scored & gm
    y = (pl_major[m] == ref[m]).astype(int)
    if m.sum() == 0 or y.min() == y.max():
        log(f"  {grp:12s} n={m.sum():4d} (no AUROC: empty or one outcome)"); continue
    for name, c in [("p_max", st_pmax[m]), ("R", R_oof[m])]:
        bins = np.minimum((c * NBIN).astype(int), NBIN - 1)
        ece = sum(abs(y[bins == b].mean() - c[bins == b].mean()) * (bins == b).sum() for b in range(NBIN) if (bins == b).any()) / len(y)
        log(f"  {grp:12s} n={m.sum():4d} acc {y.mean():.3f}  {name:5s} AUROC {roc_auc_score(y, c):.3f}  "
            f"ECE {ece:.3f}  Brier {np.mean((c - y) ** 2):.4f}  mean {c.mean():.3f}")



# ---------------------------------------------------------------- 10-11: neighbour-use features B' (optionally masked)
def nuse(dd, gg):
    """dd: (n, 50) distances sorted (inf = none); gg: (n, 50) class index of the neighbour."""
    fin = np.isfinite(dd)
    oh = (gg[..., None] == np.arange(len(MAJORS))) & fin[..., None]
    n10 = np.maximum(fin[:, :10].sum(1), 1)[:, None]
    w = np.where(fin[:, :10], 1 / (dd[:, :10] + 5), 0)
    r = (dd <= 250) & fin; n250 = r.sum(1)
    out = [dd[:, 0][:, None], oh[:, :10].sum(1) / n10,
           (oh[:, :10] * w[..., None]).sum(1) / np.maximum(w.sum(1), 1e-9)[:, None],
           n250[:, None].astype(float),
           np.where(n250[:, None] > 0, (oh & r[..., None]).sum(1) / np.maximum(n250, 1)[:, None], np.nan)]
    return np.concatenate(out, 1).astype(np.float32)


if ARGS.feats != "A":
    ku = np.array([MAJORS.index(v) if v in MAJORS else -1 for v in naive_major])
    kn = ku >= 0; kidx = np.flatnonzero(kn)
    tree = cKDTree(xy[kn])
    dd, ii = tree.query(xy, k=51); gi = kidx[ii]
    dd = np.where(gi == np.arange(N)[:, None], np.inf, dd)
    o = np.argsort(dd, 1, kind="stable")[:, :50]
    dd = np.take_along_axis(dd, o, 1); gi = np.take_along_axis(gi, o, 1)
    XB = nuse(dd, ku[gi])
    msg = f"B' features: known use {int(kn.sum())}; labelled rows nearest known median {np.median(dd[kn, 0]):.0f} m, unlabelled {np.median(dd[~kn, 0]):.0f} m"
    if ARGS.feats == "ABm":
        r_i = np.random.default_rng(700 + ARGS.seed).choice(dd[~kn, 0], kn.sum())
        d2, i2 = tree.query(xy[kn], k=400); g2 = kidx[i2]
        d2 = np.where((g2 == kidx[:, None]) | (d2 < r_i[:, None]), np.inf, d2)
        o2 = np.argsort(d2, 1, kind="stable")[:, :50]
        d2 = np.take_along_axis(d2, o2, 1); g2 = np.take_along_axis(g2, o2, 1)
        XB[kn] = nuse(d2, ku[g2])
        msg += f"; masked: labelled rows nearest visible median {np.median(d2[:, 0][np.isfinite(d2[:, 0])]):.0f} m"
        del d2, i2, g2
    del dd, ii, gi
    X = np.c_[X, XB]
    log(msg + f"; X {X.shape}")

# ---------------------------------------------------------------- GBM
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8, verbose=-1, seed=ARGS.seed)


def fit_predict(tr, y, nc, te):
    ug = np.unique(cell_id[tr])
    val_cells = set(np.random.RandomState(ARGS.seed).choice(ug, max(1, len(ug) // 10), replace=False))
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


y_osm, y_naive = enc(osm_major, MAJORS), enc(naive_major, MAJORS)
y_bin = np.where(y_naive < 0, -1, (y_naive > 0).astype(int))          # 1 = non-RES
y_nr = enc(naive_major, NONRES)
GBMS = ["osmonly_A", "naive_A", "2st_nofilt", "2st_pmax", "2st_R"]
P = {g: np.zeros((N, len(MAJORS))) for g in GBMS}
kept_log = []
for a in AREAS:
    t1 = time.time()
    te = area_of == a; out = ~te
    P["osmonly_A"][te] = fit_predict(out & (y_osm >= 0), y_osm, 8, te)
    P["naive_A"][te] = fit_predict(out & (y_naive >= 0), y_naive, 8, te)
    p1 = fit_predict(out & (y_bin >= 0), y_bin, 2, te)[:, 1]
    R = R_fold[a]
    nr_pl = (src == "pl") & (y_nr >= 0)
    keep_R = ~nr_pl | (R >= TAU)
    n_keep = int((nr_pl & keep_R & out).sum())
    cand = np.sort(st_pmax[nr_pl & out])[::-1]
    thr = cand[max(n_keep - 1, 0)] if len(cand) else 0.0                   # same budget by token p_max
    keep_pm = ~nr_pl | (st_pmax >= thr)
    for g, keep in [("2st_nofilt", np.ones(N, bool)), ("2st_pmax", keep_pm), ("2st_R", keep_R)]:
        p2 = fit_predict(out & (y_nr >= 0) & keep, y_nr, len(NONRES), te)
        P[g][te] = np.c_[1 - p1, p1[:, None] * p2]
    kept_log.append(f"  {a}: non-RES pseudo-label rows (training) {int((nr_pl & out).sum())}, kept by R {n_keep}, "
                    f"p_max threshold for same budget {thr:.3f}  ({time.time()-t1:.0f}s)")
log("\n## stage-2 filtering")
for s in kept_log:
    log(s)

# ---------------------------------------------------------------- final predictions (label + prob vector)
FINAL = {}
for g in GBMS:
    FINAL[g] = (np.array(MAJORS, object)[P[g].argmax(1)], P[g])
full_ans = np.where(has_st_all, st_ans_all, sa_ans)
full_P = np.where(has_st_all[:, None], np.nan_to_num(st_P, nan=1 / 8), sa_P)
FINAL["full VLM"] = (full_ans, full_P)
FINAL["sat-only"] = (sa_ans, sa_P)
usable = pl_major != ""


def route(g, use_vlm, vlm_P_fn):
    lab, Pg = FINAL[g]
    lab = np.where(use_vlm, pl_major, lab).astype(object)
    Pr = Pg.copy()
    Pr[use_vlm] = vlm_P_fn(use_vlm, Pg)
    return lab, Pr


def r_dist(mask, Pg):
    """R on the VLM class, (1 - R) over the other classes in proportion to the GBM."""
    out = Pg[mask].copy()
    j = enc(pl_major[mask], MAJORS)
    out[np.arange(len(j)), j] = 0
    out = out / out.sum(1, keepdims=True) * (1 - R_oof[mask])[:, None]
    out[np.arange(len(j)), j] = R_oof[mask]
    return out


for g in ["naive_A", "2st_R"]:
    FINAL[f"hybfb_{g}"] = route(g, usable, lambda m, Pg: np.nan_to_num(st_P[m], nan=1 / 8))
    FINAL[f"route_{g}"] = route(g, usable & (R_oof >= TAU), r_dist)
    FINAL[f"cmp_{g}"] = route(g, usable & (R_oof >= FINAL[g][1].max(1)), r_dist)   # v2: more reliable source wins


# ---------------------------------------------------------------- metrics
ALLCLS = MAJORS + ["unseen", "missing"]
CI = {c: i for i, c in enumerate(ALLCLS)}
K = len(ALLCLS)


def counts(g_idx, p_idx, w):
    """per-class tp / fp / fn with sample multiplicities w (bootstrap)."""
    cm = np.bincount(g_idx * K + p_idx, weights=w, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); return tp, cm.sum(0) - tp, cm.sum(1) - tp


def macro_f1(g_idx, p_idx, present, w=None):
    w = np.ones(len(g_idx)) if w is None else w
    tp, fp, fn = counts(g_idx, p_idx, w)
    d = 2 * tp + fp + fn
    f = np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0)
    return f[present].mean()


def metrics(lab, Pm, g):
    gi = np.array([CI[x] for x in g]); pi = np.array([CI.get(x, CI["unseen"]) for x in lab])
    present = sorted(set(gi))
    conf = np.where(pi < 8, Pm[np.arange(len(pi)), np.minimum(pi, 7)], np.nan)
    conf = np.nan_to_num(conf, nan=0.0)
    correct = (pi == gi).astype(float)
    bins = np.minimum((conf * NBIN).astype(int), NBIN - 1)
    ece = sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).sum()
              for b in range(NBIN) if (bins == b).any()) / len(gi)
    oh = np.zeros_like(Pm); oh[np.arange(len(gi)), gi] = 1
    hc = conf >= 0.9
    return dict(macroF1=macro_f1(gi, pi, present), acc=correct.mean(), ECE=ece,
                Brier=np.mean(((Pm - oh) ** 2).sum(1)),
                hc_acc=correct[hc].mean() if hc.any() else np.nan, hc_cov=hc.mean()), gi, pi, present


rng = np.random.default_rng(1000 + ARGS.seed)
GROUPS = [("all", scored), ("with SV", scored & has_st_all), ("no SV", scored & ~has_st_all),   # v2: groups ignore the budget
          ("human non-RES", scored & (ref != "RES"))]
ORDER = ["osmonly_A", "naive_A", "2st_nofilt", "2st_pmax", "2st_R", "full VLM", "sat-only",
         "hybfb_naive_A", "route_naive_A", "cmp_naive_A", "hybfb_2st_R", "route_2st_R", "cmp_2st_R"]
rows = []
for tag, m in GROUPS:
    g = ref[m]; n = int(m.sum())
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(NBOOT)])
    log(f"\n## use major [{tag}] n={n}  " + " ".join(f"{c}:{(g == c).sum()}" for c in sorted(set(g))))
    log(f"  {'method':15s} macroF1 [95% CI]         acc    ECE    Brier   acc@p>=.9 (cov)   dF1 vs naive_A [CI]   dF1 vs hybfb_naive_A [CI]")
    base = {}
    res = {}
    for name in ORDER:
        lab, Pm = FINAL[name]
        mt, gi, pi, present = metrics(lab[m], Pm[m], g)
        bs = np.array([macro_f1(gi, pi, present, w) for w in W])
        res[name] = (mt, bs)
    for name in ORDER:
        mt, bs = res[name]
        d1 = bs - res["naive_A"][1]; d2 = bs - res["hybfb_naive_A"][1]
        ds = lambda d, ref_n: (f"{mt['macroF1'] - res[ref_n][0]['macroF1']:+.3f} "
                               f"[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]")
        log(f"  {name:15s} {mt['macroF1']:.3f} [{np.percentile(bs, 2.5):.3f},{np.percentile(bs, 97.5):.3f}]  "
            f"{mt['acc']:.3f}  {mt['ECE']:.3f}  {mt['Brier']:.4f}  {mt['hc_acc']:.3f} ({mt['hc_cov']*100:4.1f}%)  "
            f"{ds(d1, 'naive_A')}  {ds(d2, 'hybfb_naive_A')}")
        for k, v in mt.items():
            rows.append(dict(seed=ARGS.seed, group=tag, method=name, metric=k, value=v))
        rows.append(dict(seed=ARGS.seed, group=tag, method=name, metric="dF1_vs_naive_A", value=mt["macroF1"] - res["naive_A"][0]["macroF1"]))
        rows.append(dict(seed=ARGS.seed, group=tag, method=name, metric="dF1_vs_hybfb_naive_A", value=mt["macroF1"] - res["hybfb_naive_A"][0]["macroF1"]))

# ---------------------------------------------------------------- risk-coverage of routing (TAU sweep), GBM = 2st_R
log("\n## TAU sweep, buildings with a usable street answer (route_2st_R): share sent to VLM, VLM accuracy there, "
    "GBM accuracy on the rest, overall macro-F1 on all scored buildings")
m_sv = scored & usable
for tau in ([0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] if m_sv.any() else []):
    to_vlm = usable & (R_oof >= tau)
    lab = np.where(to_vlm, pl_major, FINAL["2st_R"][0]).astype(object)
    gi = np.array([CI[x] for x in ref[scored]]); pi = np.array([CI.get(x, CI["unseen"]) for x in lab[scored]])
    v = m_sv & to_vlm; r = m_sv & ~to_vlm
    log(f"  TAU {tau:4.2f}  to VLM {v.sum() / m_sv.sum() * 100:5.1f}%  VLM acc {np.mean(pl_major[v] == ref[v]) if v.any() else np.nan:.3f}  "
        f"GBM acc on rest {np.mean(FINAL['2st_R'][0][r] == ref[r]) if r.any() else np.nan:.3f}  "
        f"VLM acc on rest {np.mean(pl_major[r] == ref[r]) if r.any() else np.nan:.3f}  "
        f"macroF1(all) {macro_f1(gi, pi, sorted(set(gi))):.3f}")

pd.DataFrame(rows).assign(budget=ARGS.budget).to_csv(OUTD + f"metrics_{SFX}.csv", index=False)
out = {"building_id": bid[in_area], "area": area_of[in_area], "ref": ref[in_area], "R": R_oof[in_area]}
for name in ORDER:
    out[name] = FINAL[name][0][in_area]
pd.DataFrame(out).to_csv(OUTD + f"preds_{SFX}.csv.gz", index=False, float_format="%.4f")
log(f"\ndone in {time.time()-T0:.0f}s")
