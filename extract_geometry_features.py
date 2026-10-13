#!/usr/bin/env python3
"""
Footprint geometry / neighbourhood / OSM-tag features for EVERY building in
Orange County (step 2 of the 09-21 pipeline: geometry -> VLM pseudolabel
distillation, no graph).

Hard rule: every feature must be computable for buildings that have NO
street-view photo (the 92.3% we want to predict), so nothing derived from
Mapillary images/matches goes in as a feature. Street-view match status is
written only as a meta_ column for splitting/eval.

Also excluded from features (they ARE the target, or leak it):
  building:levels, height, building:height, roof:levels, building:min_level,
  min_height, min_level, max_level, levels, building:levels:underground

Output CSV (gzip), one row per building, columns:
  building_id
  f_*      -- features
  meta_*   -- non-feature info (GT levels where OSM has them, has_match, ...)

Buildings loaded the same way as match_full.py (UTM 17N, ways + relation
outer members, buffer(0) for invalid rings), so building ids line up with
matches_50m.jsonl / pseudolabels.jsonl.
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from shapely.ops import unary_union

BUILDINGS_FILE = "osm_buildings_orange_county_20260918.json"
MATCHES_FILE = "mapillary_metadata/matches_50m.jsonl"

LEAKY_TAGS = {
    "building:levels", "height", "building:height", "roof:levels",
    "building:min_level", "min_height", "min_level", "max_level", "levels",
    "building:levels:underground",
}
NEIGHBOR_RADII_M = (50, 100, 250, 500)
KNN_K = 5
WORKERS = 1  # set from --workers; login nodes cap threads per user
TOP_BUILDING_TYPES = 15  # rest of building=* values collapse to "other"

# Orlando downtown (Church St & Orange Ave), just a fixed reference point
DOWNTOWN_LONLAT = (-81.3792, 28.5383)

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
to_wgs = Transformer.from_crs("EPSG:32617", "EPSG:4326", always_xy=True)


def load_buildings(bbox=None):
    with open(BUILDINGS_FILE) as f:
        elements = json.load(f)["elements"]

    if bbox is not None:
        minlon, minlat, maxlon, maxlat = bbox
        elements = [el for el in elements if "bounds" in el
                    and el["bounds"]["minlon"] >= minlon and el["bounds"]["maxlon"] <= maxlon
                    and el["bounds"]["minlat"] >= minlat and el["bounds"]["maxlat"] <= maxlat]

    lons, lats = [], []
    way_specs, rel_member_specs = [], []
    for ei, el in enumerate(elements):
        if el["type"] == "way":
            geom = el.get("geometry")
            if not geom or len(geom) < 4:
                continue
            start = len(lons)
            for pt in geom:
                lons.append(pt["lon"])
                lats.append(pt["lat"])
            way_specs.append((ei, start, len(lons)))
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") != "outer" or not m.get("geometry"):
                    continue
                mgeom = m["geometry"]
                if len(mgeom) < 3:
                    continue
                start = len(lons)
                for pt in mgeom:
                    lons.append(pt["lon"])
                    lats.append(pt["lat"])
                rel_member_specs.append((ei, start, len(lons)))

    xs, ys = to_utm.transform(np.array(lons), np.array(lats))

    ways_by_element = {ei: list(zip(xs[s:e], ys[s:e])) for ei, s, e in way_specs}
    rel_members_by_element = {}
    for ei, s, e in rel_member_specs:
        rel_members_by_element.setdefault(ei, []).append(list(zip(xs[s:e], ys[s:e])))

    def make_polygon(coords):
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if poly.area > 0 else None
        except Exception:
            return None

    ids, polys, tags_list = [], [], []
    for ei, el in enumerate(elements):
        if el["type"] == "way":
            coords = ways_by_element.get(ei)
            if not coords:
                continue
            poly = make_polygon(coords)
        else:
            members = rel_members_by_element.get(ei)
            if not members:
                continue
            parts = [p for p in (make_polygon(c) for c in members) if p is not None]
            if not parts:
                continue
            poly = unary_union(parts)
        if poly is None or poly.is_empty:
            continue
        ids.append(el["id"])
        polys.append(poly)
        tags_list.append(el.get("tags", {}) or {})
    return np.array(ids, dtype=np.int64), np.array(polys, dtype=object), tags_list


def parse_levels(v):
    """Numeric building:levels -> float, else NaN (same spirit as select_baseline_sample.py)."""
    if v is None:
        return np.nan
    try:
        x = float(str(v).strip().split(";")[0])
        return x if 0 < x < 200 else np.nan
    except ValueError:
        return np.nan


def shape_features(polys):
    area = shapely.area(polys)
    perim = shapely.length(polys)
    hull_area = shapely.area(shapely.convex_hull(polys))
    mrr = shapely.minimum_rotated_rectangle(polys)

    n_parts = shapely.get_num_geometries(polys)
    n_vertices = np.zeros(len(polys), dtype=np.int32)
    n_holes = np.zeros(len(polys), dtype=np.int32)
    long_side = np.zeros(len(polys))
    short_side = np.zeros(len(polys))
    orient = np.zeros(len(polys))
    for i, (p, r) in enumerate(zip(polys, mrr)):
        parts = p.geoms if p.geom_type == "MultiPolygon" else [p]
        n_vertices[i] = sum(len(q.exterior.coords) - 1 for q in parts)
        n_holes[i] = sum(len(q.interiors) for q in parts)
        c = np.asarray(r.exterior.coords) if r.geom_type == "Polygon" else None
        if c is None or len(c) < 4:
            continue
        e1, e2 = c[1] - c[0], c[2] - c[1]
        l1, l2 = np.hypot(*e1), np.hypot(*e2)
        long_vec = e1 if l1 >= l2 else e2
        long_side[i], short_side[i] = max(l1, l2), min(l1, l2)
        # compass-style angle of the long axis, folded to [0, 180)
        orient[i] = math.degrees(math.atan2(long_vec[0], long_vec[1])) % 180

    mrr_area = long_side * short_side
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = {
            "f_area_m2": area,
            "f_log_area": np.log1p(area),
            "f_perimeter_m": perim,
            "f_compactness": 4 * np.pi * area / perim ** 2,
            "f_solidity": area / hull_area,
            "f_rectangularity": np.where(mrr_area > 0, area / mrr_area, np.nan),
            "f_mrr_long_m": long_side,
            "f_mrr_short_m": short_side,
            "f_elongation": np.where(long_side > 0, short_side / long_side, np.nan),
            "f_orient_deg": orient,
            # deviation from the nearest cardinal axis (street-grid alignment)
            "f_orient_mod90_dev": np.minimum(orient % 90, 90 - orient % 90),
            "f_n_vertices": n_vertices,
            "f_n_parts": n_parts,
            "f_n_holes": n_holes,
        }
    return feats


def neighbor_features(cx, cy, area):
    """Centroid-based neighbourhood stats, self excluded."""
    xy = np.column_stack([cx, cy])
    tree = cKDTree(xy)
    n = len(xy)
    feats = {}

    k = min(KNN_K + 1, n)
    d, idx = tree.query(xy, k=k, workers=WORKERS)
    feats["f_nn1_dist_m"] = d[:, 1] if k > 1 else np.full(n, np.nan)
    feats[f"f_knn{KNN_K}_mean_dist_m"] = d[:, 1:].mean(axis=1) if k > 1 else np.full(n, np.nan)
    feats["f_nn1_area_m2"] = area[idx[:, 1]] if k > 1 else np.full(n, np.nan)

    log_area = np.log1p(area)
    chunk = 20000
    for r in NEIGHBOR_RADII_M:
        cnt = np.zeros(n, dtype=np.int32)
        area_sum = np.zeros(n)
        mean_log_area = np.full(n, np.nan)
        max_area = np.full(n, np.nan)
        frac_smaller = np.full(n, np.nan)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            lists = tree.query_ball_point(xy[s:e], r=r, workers=WORKERS)
            for j, nb in enumerate(lists):
                i = s + j
                nb = np.asarray(nb, dtype=np.int64)
                nb = nb[nb != i]
                cnt[i] = len(nb)
                if len(nb) == 0:
                    continue
                a = area[nb]
                area_sum[i] = a.sum()
                mean_log_area[i] = log_area[nb].mean()
                max_area[i] = a.max()
                frac_smaller[i] = (a < area[i]).mean()
        circle = np.pi * r ** 2
        feats[f"f_nbr{r}_count"] = cnt
        feats[f"f_nbr{r}_coverage"] = (area_sum + area) / circle
        feats[f"f_nbr{r}_mean_log_area"] = mean_log_area
        feats[f"f_nbr{r}_max_area_m2"] = max_area
        feats[f"f_nbr{r}_area_rank"] = frac_smaller
        print(f"    radius {r}m done (median count {np.median(cnt):.0f})", flush=True)
    return feats


def tag_features(tags_list):
    btype = pd.Series([str(t.get("building", "yes")) for t in tags_list])
    top = btype.value_counts().index[:TOP_BUILDING_TYPES]
    feats = {
        "f_building_type": np.where(btype.isin(top), btype, "other"),
        "f_has_name": np.array([("name" in t) for t in tags_list], dtype=np.int8),
        "f_has_addr": np.array([any(k.startswith("addr:") for k in t) for t in tags_list], dtype=np.int8),
        "f_has_amenity": np.array([("amenity" in t) for t in tags_list], dtype=np.int8),
        "f_has_shop": np.array([("shop" in t) for t in tags_list], dtype=np.int8),
        "f_has_office": np.array([("office" in t) for t in tags_list], dtype=np.int8),
        "f_has_tourism": np.array([("tourism" in t) for t in tags_list], dtype=np.int8),
        # number of non-leaky tags: a proxy for how carefully the building was mapped
        "f_n_tags": np.array([sum(k not in LEAKY_TAGS for k in t) for t in tags_list], dtype=np.int16),
    }
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                    help="only buildings fully inside this box (for quick tests)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    global WORKERS
    WORKERS = args.workers

    t0 = time.time()
    ids, polys, tags_list = load_buildings(args.bbox)
    print(f"[{time.time()-t0:.0f}s] buildings: {len(ids)}", flush=True)

    feats = shape_features(polys)
    print(f"[{time.time()-t0:.0f}s] shape features done", flush=True)

    cent = shapely.centroid(polys)
    cx, cy = shapely.get_x(cent), shapely.get_y(cent)
    lon, lat = to_wgs.transform(cx, cy)
    dx, dy = to_utm.transform(*DOWNTOWN_LONLAT)
    feats.update({
        "f_x_utm": cx, "f_y_utm": cy,
        "f_dist_downtown_m": np.hypot(cx - dx, cy - dy),
    })

    feats.update(neighbor_features(cx, cy, feats["f_area_m2"]))
    print(f"[{time.time()-t0:.0f}s] neighbour features done", flush=True)

    feats.update(tag_features(tags_list))

    matched = set()
    with open(MATCHES_FILE) as f:
        for line in f:
            matched.add(json.loads(line)["building_id"])

    df = pd.DataFrame({"building_id": ids, **feats})
    df["meta_lon"], df["meta_lat"] = lon, lat
    df["meta_has_streetview_match"] = np.isin(ids, list(matched)).astype(np.int8)
    df["meta_levels_gt"] = [parse_levels(t.get("building:levels")) for t in tags_list]
    df["meta_height_tag"] = [t.get("height") for t in tags_list]

    df.to_csv(args.out, index=False, compression="gzip" if args.out.endswith(".gz") else None)
    print(f"[{time.time()-t0:.0f}s] wrote {args.out}: {df.shape[0]} rows x {df.shape[1]} cols", flush=True)
    print(f"  with street-view match: {df.meta_has_streetview_match.sum()}, "
          f"with numeric levels GT: {df.meta_levels_gt.notna().sum()}", flush=True)


if __name__ == "__main__":
    main()
