#!/usr/bin/env python3
"""
Data-driven choice of LiDAR validation rectangles (09-22 decision: download
EPT only for a few typical areas instead of the whole county).

Windows: 2 km x 2 km squares in UTM 17N, sliding with 1 km stride over the
county. Per window we compute building stats (features/building_features_*),
street-view coverage, OSM levels GT, and OSM land-use composition
(osm_landuse_orange_county_20260922.json). Each area class is then ranked by
a fixed, pre-declared rule (below) -- chosen BEFORE looking at any model
output, so the areas can't be accused of being cherry-picked.

Classes and rules (all require >= MIN_BLDG buildings and >= MIN_NOMATCH
share of buildings without a street-view match, since the point of the LiDAR
check is to validate on buildings the VLM never saw):
  downtown     rank by built-up coverage (footprint area / window area)
  campus       share of buildings inside amenity=university/college >= 0.4,
               rank by that share
  residential  share inside landuse=residential >= 0.8 and building-count
               closest to the median of such windows (a "typical" one, not
               the densest)
  industrial   share inside landuse=industrial >= 0.5, rank by the NUMBER of
               buildings inside industrial land use
  tourism      share inside tourism=theme_park/resort/hotel or leisure=resort
               >= 0.3, rank by the NUMBER of buildings inside those areas
  (v1 ranked these two by share; that picked sparse windows, and the single
  huge Walt Disney World resort polygon makes share=1.0 everywhere inside it)
  random       uniform draw (seed 0) among windows with >= MIN_BLDG buildings

v2 (09-22, county fix): the OSM building file was downloaded by a bbox
rectangle, so 122,644 of its 487,081 buildings (25%) lie outside Orange County
(Seminole/Osceola/Lake...), where the Orange EPT has no points and street view
was not harvested. v2 only counts buildings inside the Orange County boundary
(OSM relation 389011, orange_county_boundary_20260922.json) and only allows
windows with >= MIN_IN_COUNTY of their area inside the county.

Output: lidar/area_candidates_county_20260922.csv (top-K per class,
non-overlapping within a class) + printed table. v1 output
(area_candidates_20260922.csv) is kept.
"""
import json

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

FEATURES = "features/building_features_20260922.csv.gz"
LANDUSE = "osm_landuse_orange_county_20260922.json"
COUNTY = "orange_county_boundary_20260922.json"
OUT = "lidar/area_candidates_county_20260922.csv"
WIN, STRIDE = 2000.0, 1000.0
MIN_BLDG, MIN_NOMATCH = 150, 0.3
MIN_IN_COUNTY = 0.95
TOP_K = 3

TO_UTM = Transformer.from_crs(4326, 32617, always_xy=True)
TO_LL = Transformer.from_crs(32617, 4326, always_xy=True)


def ring_utm(geom):
    xs, ys = TO_UTM.transform([p["lon"] for p in geom], [p["lat"] for p in geom])
    return list(zip(xs, ys))


def element_polys(el):
    """Polygons of a way / multipolygon relation (outer members only)."""
    if el["type"] == "way" and "geometry" in el and len(el["geometry"]) >= 4:
        return [Polygon(ring_utm(el["geometry"]))]
    if el["type"] == "relation":
        outs = [m for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
        # outer members are often split ways; merge them into rings
        lines = [shapely.LineString(ring_utm(m["geometry"])) for m in outs if len(m["geometry"]) >= 2]
        if lines:
            return list(shapely.get_parts(shapely.polygonize(lines)).tolist())
    return []


def category(tags):
    lu, am, tu, le = tags.get("landuse"), tags.get("amenity"), tags.get("tourism"), tags.get("leisure")
    if am in ("university", "college"):
        return "campus"
    if tu in ("theme_park", "resort", "hotel") or le == "resort":
        return "tourism"
    if lu == "residential":
        return "residential"
    if lu == "industrial":
        return "industrial"
    if lu in ("commercial", "retail"):
        return "commercial"
    return None


def county_polygon():
    el = json.load(open(COUNTY))["elements"][0]
    lines = [shapely.LineString(ring_utm(m["geometry"])) for m in el["members"]
             if m.get("role") == "outer" and "geometry" in m]
    return shapely.union_all(shapely.get_parts(shapely.polygonize(lines)))


def main():
    d = pd.read_csv(FEATURES, usecols=["building_id", "f_x_utm", "f_y_utm", "f_area_m2",
                                       "meta_has_streetview_match", "meta_levels_gt"])
    county = county_polygon()
    shapely.prepare(county)
    in_c = shapely.contains_xy(county, d.f_x_utm.to_numpy(), d.f_y_utm.to_numpy())
    print(f"buildings inside Orange County: {in_c.sum()} / {len(d)}", flush=True)
    d = d[in_c].reset_index(drop=True)
    x, y = d.f_x_utm.to_numpy(), d.f_y_utm.to_numpy()
    pts = shapely.points(x, y)

    els = json.load(open(LANDUSE))["elements"]
    cat_polys, cat_names = {}, {}
    for el in els:
        c = category(el.get("tags", {}))
        if c is None:
            continue
        for p in element_polys(el):
            p = p.buffer(0)
            if p.is_empty:
                continue
            cat_polys.setdefault(c, []).append(p)
            cat_names.setdefault(c, []).append((p, el["tags"].get("name")))
    inside = {}
    for c, polys in cat_polys.items():
        u = unary_union(polys)
        shapely.prepare(u)
        inside[c] = shapely.contains(u, pts)
        print(f"landuse {c}: {len(polys)} polygons, {inside[c].sum()} buildings inside", flush=True)

    x0 = np.floor(x.min() / STRIDE) * STRIDE
    y0 = np.floor(y.min() / STRIDE) * STRIDE
    ix = ((x - x0) // STRIDE).astype(int)
    iy = ((y - y0) // STRIDE).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1

    # 1 km cell aggregates, then 2x2 sums = 2 km windows
    def cellsum(v):
        g = np.zeros((nx, ny))
        np.add.at(g, (ix, iy), v)
        return g

    def win(g):
        return g[:-1, :-1] + g[1:, :-1] + g[:-1, 1:] + g[1:, 1:]

    one = np.ones(len(d))
    lev = d.meta_levels_gt.to_numpy()
    W = {
        "n_bldg": win(cellsum(one)),
        "built_area": win(cellsum(d.f_area_m2.to_numpy())),
        "n_match": win(cellsum(d.meta_has_streetview_match.to_numpy().astype(float))),
        "n_gt": win(cellsum((~np.isnan(lev)).astype(float))),
        "n_gt3": win(cellsum(np.nan_to_num(lev) >= 3)),
        "n_gt6": win(cellsum(np.nan_to_num(lev) >= 6)),
    }
    for c, m in inside.items():
        W["n_in_" + c] = win(cellsum(m.astype(float)))

    gx, gy = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    t = pd.DataFrame({k: v.ravel() for k, v in W.items()})
    t["minx"] = x0 + gx.ravel() * STRIDE
    t["miny"] = y0 + gy.ravel() * STRIDE
    t = t[t.n_bldg >= MIN_BLDG].copy()
    t["in_county"] = [box(a, b, a + WIN, b + WIN).intersection(county).area / (WIN * WIN)
                      for a, b in zip(t.minx, t.miny)]
    t = t[t.in_county >= MIN_IN_COUNTY].copy()
    t["coverage"] = t.built_area / (WIN * WIN)
    t["match_share"] = t.n_match / t.n_bldg
    t["mean_area"] = t.built_area / t.n_bldg
    for c in inside:
        t["sh_" + c] = t["n_in_" + c] / t.n_bldg
    elig = t[1 - t.match_share >= MIN_NOMATCH]
    print(f"{len(t)} windows with >= {MIN_BLDG} buildings, {len(elig)} also pass no-match share", flush=True)

    rules = {
        "downtown": (elig, "coverage", False),
        "campus": (elig[elig.sh_campus >= 0.4], "sh_campus", False),
        "industrial": (elig[elig.sh_industrial >= 0.5], "n_in_industrial", False),
        "tourism": (elig[elig.sh_tourism >= 0.3], "n_in_tourism", False),
    }
    res = elig[elig.sh_residential >= 0.8].copy()
    res["dist_median"] = (res.n_bldg - res.n_bldg.median()).abs()
    rules["residential"] = (res, "dist_median", True)

    picks = []
    for cls, (df, key, asc) in rules.items():
        chosen = []
        for _, r in df.sort_values(key, ascending=asc).iterrows():
            if all(abs(r.minx - c.minx) >= WIN or abs(r.miny - c.miny) >= WIN for c in chosen):
                chosen.append(r)
            if len(chosen) == TOP_K:
                break
        picks += [dict(r, cls=cls, rank=i + 1) for i, r in enumerate(chosen)]
    rnd = t.sample(TOP_K, random_state=0)
    picks += [dict(r, cls="random", rank=i + 1) for i, (_, r) in enumerate(rnd.iterrows())]

    out = pd.DataFrame(picks)
    lon, lat = TO_LL.transform(out.minx + WIN / 2, out.miny + WIN / 2)
    out["center_lat"], out["center_lon"] = np.round(lat, 5), np.round(lon, 5)
    w, s = TO_LL.transform(out.minx, out.miny)
    e, n = TO_LL.transform(out.minx + WIN, out.miny + WIN)
    out["bbox_wsen"] = [f"{a:.5f},{b:.5f},{c:.5f},{dd:.5f}" for a, b, c, dd in zip(w, s, e, n)]

    # label each window with the names of the largest named land-use polygons in it
    def names(r):
        bb = box(r.minx, r.miny, r.minx + WIN, r.miny + WIN)
        found = []
        for c, lst in cat_names.items():
            for p, nm in lst:
                if nm and p.intersects(bb):
                    found.append((p.intersection(bb).area, nm))
        found.sort(reverse=True)
        seen = []
        for _, nm in found:
            if nm not in seen:
                seen.append(nm)
            if len(seen) == 3:
                break
        return "; ".join(seen)

    out["osm_names"] = out.apply(names, axis=1)
    cols = ["cls", "rank", "center_lat", "center_lon", "in_county", "n_bldg", "coverage", "mean_area",
            "match_share", "n_gt", "n_gt3", "n_gt6", "sh_residential", "sh_industrial",
            "sh_commercial", "sh_campus", "sh_tourism", "minx", "miny", "bbox_wsen", "osm_names"]
    out = out[cols]
    out.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(out.drop(columns=["minx", "miny", "bbox_wsen"]).round(3).to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
