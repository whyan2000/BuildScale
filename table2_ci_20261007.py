"""10-07 Table 2 confidence intervals (paper open item: paired-bootstrap 95% intervals for BuildScale minus naive).

Input: route_v3_20261006/preds_b1_s{0..4}.csv.gz (budget 1, the Table 2 run 910327).
For every resample the SAME building indices are used for all methods and all 5 seeds; the statistic is the
seed-averaged macro-F1 (as in Table 2), so the interval covers building sampling and the seed mean together.
  * building bootstrap: 2,000 resamples, stratified by area (keeps each area's size)
  * area bootstrap (sensitivity, 7 areas -> clusters): 2,000 resamples of the 7 areas with replacement
Macro-F1 exactly as train_route_v3_20261006 (classes present in the reference of the scored set; "unseen" and
"missing" count as wrong). Groups: all / imaged (street answer exists) / unimaged.
Output: buildscale/table2_ci_20261007.txt
Run from the project root.
"""
import glob
import json

import numpy as np
import pandas as pd

D = "buildscale/route_v3_20261006/"
NB = 2000
MAJORS = ["RES", "COM", "IND", "AGR", "REL", "GOV", "EDU", "ACC"]
ALLCLS = MAJORS + ["unseen", "missing"]
CI = {c: i for i, c in enumerate(ALLCLS)}
K = len(ALLCLS)
METHODS = [("osmonly_A", "OSM-only"), ("sat-only", "Satellite-only VLM"), ("full VLM", "Full VLM"),
           ("naive_A", "Naive propagation"), ("2st_nofilt", "Two-stage (no filter)"),
           ("2st_pmax", "Two-stage + p_max filter"), ("2st_R", "Two-stage + R filter"),
           ("hybfb_2st_R", "BuildScale"), ("route_2st_R", "Ablation: R >= 0.5 routing"),
           ("cmp_2st_R", "Ablation: R vs propagated confidence")]
COMPARE = [("hybfb_2st_R", "naive_A"), ("hybfb_2st_R", "full VLM"), ("hybfb_2st_R", "hybfb_naive_A"),
           ("hybfb_2st_R", "2st_R"), ("2st_R", "2st_nofilt"), ("2st_R", "2st_pmax"), ("2st_R", "naive_A"),
           ("naive_A", "osmonly_A")]
out = open("buildscale/table2_ci_20261007.txt", "w")


def log(s=""):
    print(s, flush=True); print(s, file=out)


street = set()
for f in glob.glob("vlm_hazus_sv_20261001_shard*.jsonl") + glob.glob("vlm_hazus_areas_20261001_shard*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["modality"] == "street":
            street.add(r["building_id"])

P = [pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob(D + "preds_b1_s*.csv.gz"))]
base = P[0][["building_id", "area", "ref"]]
for p in P:
    assert (p.building_id.to_numpy() == base.building_id.to_numpy()).all()
sc = (base.ref.isin(MAJORS)).to_numpy()            # scored = human major present (NB/BADFP/UNK were blanked)
imaged = base.building_id.isin(street).to_numpy()
names = sorted({m for m, _ in METHODS} | {a for c in COMPARE for a in c})


def cm_counts(gi, pi):
    return np.bincount(gi * K + pi, minlength=K * K).reshape(K, K)


def f1_from_cm(cm, present):
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return np.where(d > 0, 2 * tp / np.where(d > 0, d, 1), 0)[present].mean()


log(f"# table2_ci_20261007.py  budget 1 (run 910327), {len(P)} seeds, {NB} resamples; statistic = seed-mean macro-F1")
rng = np.random.default_rng(20261007)
for tag, gm in [("all", sc), ("imaged", sc & imaged), ("unimaged", sc & ~imaged)]:
    idx = np.where(gm)[0]
    g = np.array([CI[x] for x in base.ref.to_numpy()[idx]])
    present = sorted(set(g))
    pred = {m: [np.array([CI.get(x, CI["unseen"]) for x in p[m].to_numpy()[idx]]) for p in P] for m in names}
    # per-building one-hot contributions -> fast resampled confusion matrices: code = g*K + p
    codes = {m: [g * K + pp for pp in pred[m]] for m in names}
    point = {m: np.mean([f1_from_cm(np.bincount(c, minlength=K * K).reshape(K, K), present) for c in codes[m]]) for m in names}
    areas = base.area.to_numpy()[idx]
    ua = np.unique(areas)
    by_area = {a: np.where(areas == a)[0] for a in ua}
    res = {"bld": {m: np.zeros(NB) for m in names}, "area": {m: np.zeros(NB) for m in names}}
    for b in range(NB):
        sb = np.concatenate([rng.choice(by_area[a], len(by_area[a])) for a in ua])           # stratified by area
        sa = np.concatenate([by_area[a] for a in rng.choice(ua, len(ua))])                   # cluster (area)
        for kind, s in [("bld", sb), ("area", sa)]:
            for m in names:
                res[kind][m][b] = np.mean([f1_from_cm(np.bincount(c[s], minlength=K * K).reshape(K, K), present)
                                           for c in codes[m]])
    log(f"\n## {tag}  n={len(idx)}  classes present {[ALLCLS[i] for i in present]}")
    log(f"  {'method':38s} macro-F1  [95% building CI]   [95% area CI]")
    for m, lab in METHODS:
        q = lambda v: f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]"
        log(f"  {lab:38s} {point[m]:.3f}    {q(res['bld'][m])}   {q(res['area'][m])}")
    log(f"  {'difference':38s} dF1     [95% building CI]    P(d<=0)   [95% area CI]     P(d<=0)")
    for a, c in COMPARE:
        d = point[a] - point[c]; db = res["bld"][a] - res["bld"][c]; da = res["area"][a] - res["area"][c]
        q = lambda v: f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]"
        log(f"  {a + ' - ' + c:38s} {d:+.3f}  {q(db)}  {np.mean(db <= 0):.3f}    {q(da)}  {np.mean(da <= 0):.3f}")
out.close()
