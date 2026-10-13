"""BuildScale (10-05): neighbourhood ("context") features for the downstream LightGBM.

User decision 10-05: the geometry-only model cannot learn the minority use classes (EDU / REL / GOV / IND
recall ~0 on buildings without street view), whatever the labels -> add neighbourhood features.
Three OSM data types, each used once (10-05 discussion):

Group A -- OSM only (allowed for every method, incl. OSM-only):
  A1 sites: closed OSM areas WITHOUT a building tag (osm_context_areas_orange_county_20261005.json).
     Areas that carry building=* are the building itself (= the source of our OSM use labels: 258 of 408
     place_of_worship areas, 343 of 398 hotels, all townhalls ...) -> dropped, otherwise label leakage.
     Per category: ar_in_<cat> (building centroid inside, 0/1) and ar_d_<cat> (centroid -> nearest area of
     that category, 0 inside, capped at DMAX).
     Note: amenity sites are often co-tagged landuse=public (bulk public-land import) -> coverage may lean
     towards public facilities.
  A2 POIs: NODES only from osm_pois_orange_county_20260922.json (its ways are buildings or sites, already
     covered by A1 / A3). Mapped to a Hazus major with the 10-01 OSM label rules (match_other).
     poi_<MAJOR>_<r> / poi_all_<r>: counts within r m of the centroid, EXCLUDING nodes inside the
     building's own footprint (those nodes are a source of its own OSM label).
  A3 OSM labels of OTHER buildings (buildscale/osm_labels_20261001_v2.csv), within r m, self excluded:
     osm_n_<r> (labelled for use), osm_f_<MAJOR>_<r> (share), osm_nh_<r> (labelled for height),
     osm_fMH_<r> (share 4+ stories); osm_d_<MAJOR> = distance to the nearest other building with that
     OSM major label (capped at DMAX).
Group B -- VLM street-view pseudo-labels of OTHER buildings (naive / BuildScale only):
     867636 + 867633 street answers (Hazus 3-bin run of 10-01), soft probabilities, self excluded:
     pl_n_<r>, pl_p_<MAJOR>_<r> (mean P(major), renormalised without "unseen"), pl_pMH_<r>
     (mean P(4+ stories), renormalised over L/M/H), pl_sign_<r> (share with a readable sign);
     pl_d_<MAJOR> = distance to the nearest other building whose argmax major is MAJOR.
Columns are prefixed ctxA_ / ctxB_. Fractions with an empty denominator are NaN (LightGBM handles it).
Output: features/nbr_features_20261005.csv.gz (all 364,437 in-county buildings)
Run from the project root on a compute node (~20-30 GB RAM for the 300 m neighbour matrix).
"""
import glob
import json
import os
import sys
import time
from collections import Counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.ops import linemerge, polygonize, unary_union

sys.path.insert(0, "buildscale")
from extract_geometry_features import load_buildings          # noqa: E402
from osm_labels_20261001 import match_other                    # noqa: E402

T0 = time.time()
AREAS_FILE = "osm_context_areas_orange_county_20261005.json"
POIS_FILE = "osm_pois_orange_county_20260922.json"
OUT = "features/nbr_features_20261005.csv.gz"
RADII = [100, 300]
DMAX = 3000.0
MAJORS = ["RES", "COM", "IND", "AGR", "REL", "GOV", "EDU", "ACC"]
D_MAJORS = ["COM", "IND", "REL", "GOV", "EDU", "ACC"]
TO_UTM = Transformer.from_crs(4326, 32617, always_xy=True)


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def area_cat(t):
    """Site category of an OSM area (no building tag), or None."""
    a, lu, le, to = t.get("amenity"), t.get("landuse"), t.get("leisure"), t.get("tourism")
    if a in ("college", "university"):
        return "univ"
    if a in ("school", "kindergarten") or lu == "education":
        return "school"
    if a == "place_of_worship" or lu == "religious":
        return "rel"
    if a in ("townhall", "courthouse", "police", "fire_station", "prison"):
        return "gov"
    if a in ("hospital", "clinic"):
        return "hosp"
    if a == "parking":
        return "parking"
    if lu == "military" or "military" in t:
        return "mil"
    if t.get("aeroway") == "aerodrome":
        return "airport"
    if to in ("theme_park", "hotel", "resort", "attraction", "zoo") or le == "resort":
        return "tour"
    if le in ("park", "golf_course", "stadium", "sports_centre", "pitch") or lu == "recreation_ground":
        return "rec"
    if lu in ("cemetery",) or a == "grave_yard":
        return "cem"
    if lu in ("farmland", "farmyard", "orchard", "meadow", "plant_nursery", "greenhouse_horticulture"):
        return "farm"
    if lu in ("residential", "commercial", "retail", "industrial", "public"):
        return {"residential": "res", "commercial": "com", "retail": "retail", "industrial": "ind",
                "public": "public"}[lu]
    return None


AREA_CATS = ["res", "com", "retail", "ind", "public", "school", "univ", "rel", "gov", "hosp", "tour", "rec",
             "farm", "cem", "parking", "airport", "mil"]


def area_polygon(e):
    def ring(geom):
        x, y = TO_UTM.transform([p["lon"] for p in geom], [p["lat"] for p in geom])
        return list(zip(x, y))
    if e["type"] == "way":
        g = e.get("geometry")
        if not g or len(g) < 4 or g[0] != g[-1]:
            return None
        p = shapely.Polygon(ring(g))
        return p if p.is_valid else p.buffer(0)
    lines = [shapely.LineString(ring(m["geometry"])) for m in e.get("members", [])
             if m.get("role") == "outer" and m.get("geometry") and len(m["geometry"]) >= 2]
    if not lines:
        return None
    polys = list(polygonize(linemerge(lines)))
    return unary_union(polys) if polys else None


# ---------------------------------------------------------------- buildings
feat = pd.read_csv("features/building_features_county_20260922.csv.gz",
                   usecols=["building_id", "f_x_utm", "f_y_utm", "meta_in_orange"])
feat = feat[feat.meta_in_orange == 1].reset_index(drop=True)
bid = feat.building_id.to_numpy()
N = len(bid)
xy = feat[["f_x_utm", "f_y_utm"]].to_numpy()
pts = shapely.points(xy[:, 0], xy[:, 1])
ids_all, polys_all, _ = load_buildings()
pos = {b: i for i, b in enumerate(ids_all)}
fp = [polys_all[pos[b]] for b in bid]
log(f"in-county buildings {N}")
out = pd.DataFrame({"building_id": bid})

# ---------------------------------------------------------------- A1 sites
els = json.load(open(AREAS_FILE))["elements"]
cat_polys = {c: [] for c in AREA_CATS}
n_drop = Counter()
for e in els:
    t = e.get("tags", {})
    c = area_cat(t)
    if c is None:
        continue
    if "building" in t:
        n_drop[c] += 1
        continue
    p = area_polygon(e)
    if p is not None and not p.is_empty and p.area > 0:
        cat_polys[c].append(p)
log("sites kept: " + " ".join(f"{c}:{len(v)}" for c, v in cat_polys.items()))
log("sites dropped (building tag): " + " ".join(f"{c}:{n}" for c, n in n_drop.items()))
for c in AREA_CATS:
    ps = cat_polys[c]
    if not ps:
        out[f"ctxA_ar_in_{c}"] = 0
        out[f"ctxA_ar_d_{c}"] = DMAX
        continue
    tree = STRtree(ps)
    _, d = tree.query_nearest(pts, return_distance=True, all_matches=False)
    d = np.minimum(d, DMAX)
    out[f"ctxA_ar_in_{c}"] = (d == 0).astype(np.int8)
    out[f"ctxA_ar_d_{c}"] = d.astype(np.float32)
log("A1 done: share inside " + " ".join(f"{c}:{out[f'ctxA_ar_in_{c}'].mean()*100:.1f}%" for c in AREA_CATS))

# ---------------------------------------------------------------- A2 POI nodes
pe = [e for e in json.load(open(POIS_FILE))["elements"] if e["type"] == "node"]
px, py = TO_UTM.transform([e["lon"] for e in pe], [e["lat"] for e in pe])
pmaj = np.array([match_other(e.get("tags", {}))[0] or "" for e in pe])
log(f"POI nodes {len(pe)}: " + " ".join(f"{m}:{(pmaj == m).sum()}" for m in MAJORS[:-1]) +
    f" unmapped:{(pmaj == '').sum()}")
# own-footprint POIs: (poi index, building index) pairs
pi, bi = STRtree(fp).query(shapely.points(px, py), predicate="intersects")
own = pd.DataFrame({"b": bi, "m": pmaj[pi]})
pxy = np.c_[px, py]
btree_xy = xy
for m in ["all"] + MAJORS[:-1]:
    sel = np.ones(len(pe), bool) if m == "all" else pmaj == m
    if sel.sum() == 0:
        continue
    tr = cKDTree(pxy[sel])
    own_m = own if m == "all" else own[own.m == m]
    own_cnt = np.bincount(own_m.b.to_numpy(), minlength=N)
    for r in RADII:
        cnt = tr.query_ball_point(btree_xy, r, return_length=True, workers=4)
        out[f"ctxA_poi_{m}_{r}"] = np.maximum(cnt - own_cnt, 0).astype(np.int32)
log(f"A2 done; POI nodes inside own footprint: {len(own)} (excluded)")

# ---------------------------------------------------------------- neighbour matrices (self excluded)
btree = cKDTree(xy)
D = btree.sparse_distance_matrix(btree, max(RADII), output_type="coo_matrix")
keep = D.data > 0                                              # self (and exact duplicates) drop out
A = {}
for r in RADII:
    k = keep & (D.data <= r)
    A[r] = sp.csr_matrix((np.ones(k.sum(), np.float32), (D.row[k], D.col[k])), shape=(N, N))
    log(f"neighbour matrix r={r}: {A[r].nnz} pairs, median {np.median(np.diff(A[r].indptr)):.0f} per building")
del D, keep


def frac_feats(prefix, M, has, names, r):
    """M: N x k numeric (rows of unlabelled buildings = 0), has: N indicator. Mean over neighbours."""
    n = A[r] @ has
    s = A[r] @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        f = s / n[:, None]
    out[f"{prefix}_n_{r}"] = n.astype(np.int32)
    for j, c in enumerate(names):
        out[f"{prefix}_{c}_{r}"] = np.where(n > 0, f[:, j], np.nan).astype(np.float32)


def nearest_feats(prefix, labels, names):
    for c in names:
        sel = np.where(labels == c)[0]
        if len(sel) < 2:
            out[f"{prefix}_d_{c}"] = DMAX
            continue
        d, j = cKDTree(xy[sel]).query(xy, k=2, workers=4)
        self_hit = sel[j[:, 0]] == np.arange(N)
        dd = np.where(self_hit, d[:, 1], d[:, 0])
        out[f"{prefix}_d_{c}"] = np.minimum(dd, DMAX).astype(np.float32)


# ---------------------------------------------------------------- A3 OSM labels of other buildings
lab = pd.read_csv("buildscale/osm_labels_20261001_v2.csv", keep_default_na=False).set_index("building_id").reindex(bid)
um = lab.use_major.fillna("").to_numpy()
hb = lab.height_bin.fillna("").to_numpy()
Mu = np.stack([(um == m) for m in MAJORS], 1).astype(np.float32)
hu = (um != "").astype(np.float32)
Mh = np.isin(hb, ["M", "H"]).astype(np.float32)[:, None]
hh = (hb != "").astype(np.float32)
for r in RADII:
    frac_feats("ctxA_osm", Mu, hu, [f"f_{m}" for m in MAJORS], r)
    n = A[r] @ hh
    out[f"ctxA_osm_nh_{r}"] = n.astype(np.int32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[f"ctxA_osm_fMH_{r}"] = np.where(n > 0, (A[r] @ Mh)[:, 0] / n, np.nan).astype(np.float32)
nearest_feats("ctxA_osm", um, D_MAJORS)
log("A3 done")

# ---------------------------------------------------------------- B VLM street pseudo-labels of other buildings
pl = {}
for f in sorted(glob.glob("vlm_hazus_sv_20261001_shard*.jsonl") + glob.glob("vlm_hazus_areas_20261001_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["modality"] == "street":
            pl[r["building_id"]] = r
row = {b: i for i, b in enumerate(bid)}
Pu = np.zeros((N, len(MAJORS)), np.float32)
Pmh = np.zeros((N, 1), np.float32)
Sg = np.zeros((N, 1), np.float32)
has_u = np.zeros(N, np.float32)
has_h = np.zeros(N, np.float32)
argmax_u = np.array([""] * N, dtype=object)
for b, r in pl.items():
    i = row.get(b)
    if i is None:
        continue
    pu = r.get("p_use")
    if pu and r["use_major"] in MAJORS:
        v = np.array([pu.get(m, 0.0) for m in MAJORS])
        if v.sum() > 0:
            Pu[i] = v / v.sum(); has_u[i] = 1; argmax_u[i] = MAJORS[int(v.argmax())]
            Sg[i] = float((r.get("sign") or "none").strip().lower() not in ("none", ""))
    ph = r.get("p_height")
    if ph and r["height"] in ("L", "M", "H"):
        v = np.array([ph.get(k, 0.0) for k in ("L", "M", "H")])
        if v.sum() > 0:
            Pmh[i] = (v[1] + v[2]) / v.sum(); has_h[i] = 1
log(f"B: street pseudo-labels in county {int(has_u.sum())} (use) / {int(has_h.sum())} (height)")
for r in RADII:
    frac_feats("ctxB_pl", np.c_[Pu, Sg], has_u, [f"p_{m}" for m in MAJORS] + ["sign"], r)
    n = A[r] @ has_h
    with np.errstate(invalid="ignore", divide="ignore"):
        out[f"ctxB_pl_pMH_{r}"] = np.where(n > 0, (A[r] @ Pmh)[:, 0] / n, np.nan).astype(np.float32)
nearest_feats("ctxB_pl", argmax_u, D_MAJORS)
log("B done")

out.to_csv(OUT, index=False, float_format="%.4g")
log(f"wrote {OUT}: {out.shape[0]} x {out.shape[1]-1} features "
    f"(A {sum(c.startswith('ctxA_') for c in out)}, B {sum(c.startswith('ctxB_') for c in out)})")
