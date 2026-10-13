"""10-06: targeted labels (annotation_hard_20261006, 599 buildings outside the 7 areas) ->
(1) observed VLM error yield per stratum / per flag count; (2) refit R with these labels (1/pi weights) and
score it on the 7 evaluation areas (942 street pseudo-labels with human labels; never used for fitting here).
Human IND -> COM (10-06 decision). NB / BADFP / UNK excluded.
Run from the project root."""
import glob, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
NR = ["COM", "IND", "AGR", "REL", "GOV", "EDU", "ACC"]
S = pd.read_csv("buildscale/cheap_signals_20261006.csv.gz", low_memory=False).set_index("building_id")
smp = pd.read_csv("annotation_hard_20261006/sample_hard_20261006.csv").set_index("building_id")
lab = {}
for f in glob.glob("annotation_hard_20261006/labels_dump_20261006/labels/*.json"):
    d = json.load(open(f)); lab[int(f.split("/")[-1][:-5])] = d
L = smp.join(pd.DataFrame({b: dict(code=d["code"], major=d.get("major"), unsure=d.get("unsure", False)) for b, d in lab.items()}).T)
print("labelled", L.code.notna().sum(), "/", len(L), "| codes:", L.code.value_counts().head(12).to_dict())
L = L[~L.code.isin(["NB", "BADFP", "UNK"]) & L.code.notna()].copy()
L["hmajor"] = L.major.replace({"IND": "COM"})
L["correct"] = (L.ans_major == L.hmajor).astype(int)
print(f"usable {len(L)}; unsure {int(L.unsure.sum())}")
print("\n## observed VLM error rate (street major vs human), targeted sample")
for k, g in L.groupby("stratum"):
    print(f"  stratum {k:6s} n={len(g):3d} errors={int((1-g.correct).sum()):3d} rate={1-g.correct.mean():.2f}  pi={g.pi.iloc[0]:.4f}")
for k, g in L.groupby("nflag"):
    print(f"  flags {k}  n={len(g):3d} errors={int((1-g.correct).sum()):3d} rate={1-g.correct.mean():.2f}")
w = 1 / L.pi
print(f"  total errors {int((1-L.correct).sum())} / {len(L)}; weighted county error rate among street PLs outside areas: {np.average(1-L.correct, weights=w):.3f}")
print("  human major distribution:", L.hmajor.value_counts().to_dict())
print("  VLM said -> human (errors only):", L[L.correct == 0].groupby(["ans_major", "hmajor"]).size().sort_values(ascending=False).head(10).to_dict())

# ---- R refit on targeted labels, scored on the 7 areas
E = S[S.correct_major.notna() & S.ans_major.isin(["RES"] + NR)].copy()
T = S.loc[L.index].assign(y=L.correct.values, w=w.values / w.mean())
SIG3 = ["tok_use_margin", "xm_p_sat_ans", "osm_p_ans"]
def feats(D, cols):
    if cols == ["logit_pmax"]:
        p = np.clip(D.tok_use_pmax.to_numpy(float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p))[:, None]
    return D[cols].to_numpy(float)
def ece(c, y, nb=15):
    b = np.minimum((c * nb).astype(int), nb - 1)
    return sum(abs(y[b == i].mean() - c[b == i].mean()) * (b == i).sum() for i in range(nb) if (b == i).any()) / len(y)
print("\n## R fitted on the targeted labels (1/pi weights), scored on the 7 areas")
for grp, gT, gE in [("VLM RES", T.ans_major == "RES", E.ans_major == "RES"), ("VLM non-RES", T.ans_major.isin(NR), E.ans_major.isin(NR))]:
    tt, ee = T[gT], E[gE]; ye = ee.correct_major.astype(int).to_numpy()
    print(f"  [{grp}] fit n={len(tt)} (errors {int((1-tt.y).sum())}); eval n={len(ee)} (errors {int((1-ye).sum())})")
    print(f"    raw p_max          AUROC {roc_auc_score(ye, ee.tok_use_pmax):.3f}  ECE {ece(ee.tok_use_pmax.to_numpy(), ye):.3f}  Brier {np.mean((ee.tok_use_pmax - ye) ** 2):.4f}")
    for name, cols in [("Platt p_max", ["logit_pmax"]), ("3 signals", SIG3), ("3 signals + nb", SIG3 + ["nb_pl_same"])]:
        Xt, Xe = feats(tt, cols), feats(ee, cols)
        mu = np.nanmean(Xt, 0); sd = np.nanstd(Xt, 0) + 1e-9
        Zt = (np.where(np.isnan(Xt), mu, Xt) - mu) / sd; Ze = (np.where(np.isnan(Xe), mu, Xe) - mu) / sd
        for wt in ["w", "unw"]:
            m = LogisticRegression(C=1.0).fit(Zt, tt.y, sample_weight=tt.w if wt == "w" else None)
            r = m.predict_proba(Ze)[:, 1]
            print(f"    {name:18s} {wt:3s} AUROC {roc_auc_score(ye, r):.3f}  ECE {ece(r, ye):.3f}  Brier {np.mean((r - ye) ** 2):.4f}")
