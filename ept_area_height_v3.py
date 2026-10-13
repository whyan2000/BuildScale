#!/usr/bin/env python3
"""
Per-building LiDAR height v3 (09-28): remove tree canopy from the roof points.

Why: v2 (ept_area_height.py) takes classes {1, 6} as roof points because high-rise roofs in
FL_Peninsular_FDEM_Orange_2018 are left in class 1. But this dataset has NO vegetation classes
(3/4/5) at all -- trees are class 1 too, so canopy over a roof enters the height. 09-28 audit: 9
visually single-storey houses under oaks had v2 p95 10.5-16 m, class-6-only 3-8 m.

Point-cloud-only cue used here (no VLM output is used for any choice): canopy is partly penetrated
by the laser -> multiple returns; hard roofs give single returns (in a Long Lake node: class 1
68% multi-return, class 6 15%). Candidate roof point sets, all computed in one pass:
  v2   : class 1 or 6                                     (reproduces v2)
  c6   : class 6 only
  sr   : class 6, plus class 1 with number_of_returns == 1
  srp  : sr, restricted to "planar" 1 m cells (z range of the cell's sr points <= PLANAR_M, >= 2 pts)
For each: n points and p50/p95 height above the DTM. Same DTM, footprint erosion (0.5 m) and area
box logic as v2; building set = centroid inside the area box.

Usage (from the project root):
  python3 lidar/ept_area_height_v3.py --areas lidar/selected_areas_county_20260922.csv --tag final8_county
Outputs lidar/ept_areas/<area>_heights_v3_20260928.csv and <tag>_heights_v3_20260928.csv.
"""
import argparse
import glob
import sys
import time

import laspy
import numpy as np
import pandas as pd
import shapely
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

sys.path.insert(0, "lidar"); sys.path.insert(0, ".")
import ept_area_height as E

VER, DATE = "v3", "20260928"
PLANAR_CELL_M, PLANAR_M = 1.0, 0.5
SETS = ["v2", "c6", "sr", "srp"]


def stats(h, prefix):
    if len(h) == 0:
        return {f"n_{prefix}": 0, f"p50_{prefix}": np.nan, f"p95_{prefix}": np.nan}
    return {f"n_{prefix}": len(h), f"p50_{prefix}": float(np.median(h)), f"p95_{prefix}": float(np.percentile(h, 95))}


def area_height(r, ids, polys, cent_x, cent_y):
    t0 = time.time()
    bx = E.area_box_utm(r); bm = E.area_box_utm(r, E.MARGIN_M)
    sel = (cent_x >= bx[0]) & (cent_x < bx[2]) & (cent_y >= bx[1]) & (cent_y < bx[3])
    a_ids, a_polys = ids[sel], polys[sel]
    b = shapely.bounds(a_polys)
    edge = (b[:, 0] < bm[0]) | (b[:, 2] > bm[2]) | (b[:, 1] < bm[1]) | (b[:, 3] > bm[3])
    minx, miny, maxx, maxy = bm
    W, H = int(np.ceil((maxx - minx) / E.ID_CELL_M)), int(np.ceil((maxy - miny) / E.ID_CELL_M))
    shrunk = [q if not q.is_empty else p for p, q in zip(a_polys, shapely.buffer(a_polys, -E.ERODE_M))]
    id_rast = features.rasterize(((p, i + 1) for i, p in enumerate(shrunk)), out_shape=(H, W),
                                 transform=from_origin(minx, maxy, E.ID_CELL_M, E.ID_CELL_M), fill=0, dtype="int32")
    gW, gH = int(np.ceil((maxx - minx) / E.GROUND_CELL_M)), int(np.ceil((maxy - miny) / E.GROUND_CELL_M))
    g_sum, g_cnt = np.zeros(gH * gW), np.zeros(gH * gW)
    parts = []
    for fn in sorted(glob.glob(f"{E.OUT_DIR}/{r.area}/nodes/*.laz")):
        las = laspy.read(fn)
        cls = np.asarray(las.classification)
        m = (cls == E.CLASS_GROUND) | np.isin(cls, E.ROOF_CLASSES)
        if not m.any():
            continue
        x, y = E.M3857_TO_UTM.transform(np.asarray(las.x)[m], np.asarray(las.y)[m])
        z, cls, nr = np.asarray(las.z)[m], cls[m], np.asarray(las.number_of_returns)[m]
        k = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)
        x, y, z, cls, nr = x[k], y[k], z[k], cls[k], nr[k]
        g = cls == E.CLASS_GROUND
        gi = (((maxy - y[g]) // E.GROUND_CELL_M).clip(0, gH - 1).astype(np.int64) * gW
              + ((x[g] - minx) // E.GROUND_CELL_M).clip(0, gW - 1).astype(np.int64))
        g_sum += np.bincount(gi, weights=z[g], minlength=gH * gW)
        g_cnt += np.bincount(gi, minlength=gH * gW)
        bb = ~g
        row = ((maxy - y[bb]) // E.ID_CELL_M).clip(0, H - 1).astype(np.int64)
        col = ((x[bb] - minx) // E.ID_CELL_M).clip(0, W - 1).astype(np.int64)
        bid = id_rast[row, col]; kk = bid > 0
        parts.append(pd.DataFrame({"i": bid[kk] - 1, "x": x[bb][kk], "y": y[bb][kk], "z": z[bb][kk],
                                   "c": cls[bb][kk], "nr": nr[bb][kk]}))
    dtm = np.full(gH * gW, np.nan); ok = g_cnt > 0; dtm[ok] = g_sum[ok] / g_cnt[ok]; dtm = dtm.reshape(gH, gW)
    if (~ok).any():
        _, (ri, ci) = ndimage.distance_transform_edt(np.isnan(dtm), return_indices=True); dtm = dtm[ri, ci]
    P = pd.concat(parts, ignore_index=True)
    P["h"] = P.z - dtm[((maxy - P.y) // E.GROUND_CELL_M).clip(0, gH - 1).astype(int),
                       ((P.x - minx) // E.GROUND_CELL_M).clip(0, gW - 1).astype(int)]
    P["sr"] = (P.c == 6) | ((P.c == 1) & (P.nr == 1))
    # planar 1 m cells among sr points
    S = P[P.sr].copy()
    S["cell"] = ((maxy - S.y) // PLANAR_CELL_M).astype(np.int64) * 10**7 + ((S.x - minx) // PLANAR_CELL_M).astype(np.int64)
    cs = S.groupby(["i", "cell"]).z.agg(["min", "max", "size"])
    planar = cs[((cs["max"] - cs["min"]) <= PLANAR_M) & (cs["size"] >= 2)].index
    S = S.set_index(["i", "cell"]); S["planar"] = S.index.isin(planar); S = S.reset_index()

    rows = []
    gP = dict(tuple(P.groupby("i"))); gS = dict(tuple(S.groupby("i")))
    for i in range(len(a_ids)):
        rec = {"building_id": a_ids[i], "area": r.area, "edge_clipped": int(edge[i]),
               "footprint_area_m2": float(shapely.area(a_polys[i]))}
        p = gP.get(i)
        if p is None:
            for s in SETS: rec.update(stats([], s))
            rec.update(n_c1=0, c1_multi_share=np.nan, c6_share=np.nan, planar_cell_share=np.nan)
        else:
            s = gS.get(i)
            rec.update(stats(p.h.to_numpy(), "v2"))
            rec.update(stats(p.h[p.c == 6].to_numpy(), "c6"))
            rec.update(stats(p.h[p.sr].to_numpy(), "sr"))
            rec.update(stats(s.h[s.planar].to_numpy() if s is not None else [], "srp"))
            n1 = int((p.c == 1).sum())
            rec.update(n_c1=n1, c1_multi_share=float((p.nr[p.c == 1] > 1).mean()) if n1 else np.nan,
                       c6_share=float((p.c == 6).mean()),
                       planar_cell_share=float(s.drop_duplicates(["cell"]).planar.mean()) if s is not None else np.nan)
        rows.append(rec)
    res = pd.DataFrame(rows)
    print(f"{r.area}: {len(res)} buildings, {int((res.n_v2 > 0).sum())} with pts, {len(P)/1e6:.1f}M bldg pts, "
          f"{time.time() - t0:.0f}s", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", required=True)
    ap.add_argument("--tag", default="final8_county")
    a = ap.parse_args()
    areas = pd.read_csv(a.areas)
    from extract_geometry_features import load_buildings
    ids, polys, _ = load_buildings()
    c = shapely.centroid(polys); cx, cy = shapely.get_x(c), shapely.get_y(c)
    out = []
    for r in areas.itertuples():
        res = area_height(r, ids, polys, cx, cy)
        res.to_csv(f"{E.OUT_DIR}/{r.area}_heights_{VER}_{DATE}.csv", index=False)
        out.append(res)
    allr = pd.concat(out)
    meta = pd.read_csv(E.FEATURES, usecols=["building_id", "meta_has_streetview_match", "meta_levels_gt"])
    allr = allr.merge(meta, on="building_id", how="left")
    fn = f"{E.OUT_DIR}/{a.tag}_heights_{VER}_{DATE}.csv"
    allr.to_csv(fn, index=False)
    print(f"wrote {fn} ({len(allr)} rows)")


if __name__ == "__main__":
    main()
