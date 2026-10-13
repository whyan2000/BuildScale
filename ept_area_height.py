#!/usr/bin/env python3
"""
Per-building LiDAR height for a few selected validation areas, read from the
USGS 3DEP EPT (Entwine octree) copy of FL_Peninsular_FDEM_Orange_2018 on AWS
(usgs-lidar-public) instead of the 243 GB of rockyweb LAZ tiles.

09-22 decisions behind this:
  - only typical areas are downloaded (areas chosen by select_lidar_areas.py,
    listed in selected_areas_20260922.csv), not the whole county
  - EPT: nodes are small (tens of KB - 1 MB), so an area's download is just the
    set of octree nodes whose XY box intersects the area; resumable per node
  - EPT coords are EPSG:3857, Z in metres (checked on a downtown node: ground
    ~33.6, i.e. Orlando's ~30 m elevation; ftUS would give ~10 m)

Same height method as lidar_building_height.py (which stays for the LAZ
tiles): class 2 ground -> gridded DTM with nearest fill; above-ground points
-> building id via a rasterized footprint-id grid; roof Z percentiles minus
DTM. Here everything is in UTM 17N metres.

v2 (09-22): v1 used class 6 only and capped downtown heights at ~17 m. In this
dataset high-rise roofs are class 1 (unclassified), class 6 only covers low
roofs (e.g. building 500079228, OSM 21 levels: class 6 ~8 m above ground,
class 1 p95 ~57 m). v2 uses classes ROOF_CLASSES = {1, 6} (noise 7/18
excluded) and rasterizes footprints shrunk by ERODE_M so overhanging tree
crowns (also class 1) at the edges are mostly excluded. Outputs get a _v2
suffix; v1 outputs are kept.

Buildings belong to an area if their centroid is inside the area box; points
are kept for box + MARGIN_M, so footprints are complete unless they extend
more than MARGIN_M past the box (edge_clipped=1).

Usage (run from the project root, load_buildings uses relative paths):
  python3 lidar/ept_area_height.py download --areas lidar/selected_areas_20260922.csv
  python3 lidar/ept_area_height.py height   --areas lidar/selected_areas_20260922.csv
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import laspy
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

EPT = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/FL_Peninsular_FDEM_Orange_2018"
OUT_DIR = "lidar/ept_areas"
MARGIN_M = 30.0
GROUND_CELL_M = 1.5
ID_CELL_M = 0.3
CLASS_GROUND = 2
ROOF_CLASSES = (1, 6)
ERODE_M = 0.5
VER = "v2"
THREADS = 32
FEATURES = "features/building_features_20260922.csv.gz"

UTM_TO_3857 = Transformer.from_crs(32617, 3857, always_xy=True)
M3857_TO_UTM = Transformer.from_crs(3857, 32617, always_xy=True)


def fetch(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {err}")


def area_box_utm(r, margin=0.0):
    return r.minx - margin, r.miny - margin, r.minx + r.size_m + margin, r.miny + r.size_m + margin


def box_3857(b):
    xs, ys = UTM_TO_3857.transform([b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
    return min(xs), min(ys), max(xs), max(ys)


def nodes_for_box(q, root_bounds, cache):
    """All EPT node keys with >0 points whose XY extent intersects box q (EPSG:3857)."""
    size0 = root_bounds[3] - root_bounds[0]

    def hits(key):
        D, X, Y, _ = map(int, key.split("-"))
        s = size0 / 2 ** D
        x0, y0 = root_bounds[0] + X * s, root_bounds[1] + Y * s
        return x0 < q[2] and x0 + s > q[0] and y0 < q[3] and y0 + s > q[1]

    def hier(key):
        if key not in cache:
            cache[key] = json.loads(fetch(f"{EPT}/ept-hierarchy/{key}.json"))
        return cache[key]

    out, todo = [], ["0-0-0-0"]
    while todo:
        h = hier(todo.pop())
        for k, n in h.items():
            if not hits(k):
                continue
            if n == -1:
                todo.append(k)          # sub-hierarchy file rooted at k
            elif n > 0:
                out.append(k)
    return sorted(set(out))


def download(areas):
    info = json.loads(fetch(f"{EPT}/ept.json"))
    cache = {}
    for r in areas.itertuples():
        d = f"{OUT_DIR}/{r.area}/nodes"
        os.makedirs(d, exist_ok=True)
        t0 = time.time()
        keys = nodes_for_box(box_3857(area_box_utm(r, MARGIN_M)), info["bounds"], cache)
        todo = [k for k in keys if not os.path.exists(f"{d}/{k}.laz")]

        def get(k):
            b = fetch(f"{EPT}/ept-data/{k}.laz")
            with open(f"{d}/{k}.laz.tmp", "wb") as f:
                f.write(b)
            os.replace(f"{d}/{k}.laz.tmp", f"{d}/{k}.laz")
            return len(b)

        with ThreadPoolExecutor(THREADS) as ex:
            nb = sum(ex.map(get, todo))
        dt = time.time() - t0
        print(f"{r.area}: {len(keys)} nodes ({len(keys) - len(todo)} already there), "
              f"{nb / 1e6:.0f} MB in {dt:.0f}s ({nb / 1e6 / max(dt, 1e-9):.1f} MB/s)", flush=True)


def area_height(r, ids, polys, cent_x, cent_y):
    t0 = time.time()
    bx = area_box_utm(r)
    bm = area_box_utm(r, MARGIN_M)
    sel = (cent_x >= bx[0]) & (cent_x < bx[2]) & (cent_y >= bx[1]) & (cent_y < bx[3])
    a_ids, a_polys = ids[sel], polys[sel]
    b = shapely.bounds(a_polys)
    edge = (b[:, 0] < bm[0]) | (b[:, 2] > bm[2]) | (b[:, 1] < bm[1]) | (b[:, 3] > bm[3])

    minx, miny, maxx, maxy = bm
    W, H = int(np.ceil((maxx - minx) / ID_CELL_M)), int(np.ceil((maxy - miny) / ID_CELL_M))
    shrunk = [q if not q.is_empty else p for p, q in zip(a_polys, shapely.buffer(a_polys, -ERODE_M))]
    id_rast = features.rasterize(((p, i + 1) for i, p in enumerate(shrunk)), out_shape=(H, W),
                                 transform=from_origin(minx, maxy, ID_CELL_M, ID_CELL_M),
                                 fill=0, dtype="int32", all_touched=False)
    gW, gH = int(np.ceil((maxx - minx) / GROUND_CELL_M)), int(np.ceil((maxy - miny) / GROUND_CELL_M))
    g_sum, g_cnt = np.zeros(gH * gW), np.zeros(gH * gW)

    files = sorted(glob.glob(f"{OUT_DIR}/{r.area}/nodes/*.laz"))
    n_total = 0
    b_idx, b_x, b_y, b_z = [], [], [], []
    for fn in files:
        las = laspy.read(fn)
        n_total += len(las.points)
        cls = np.asarray(las.classification)
        roof = np.isin(cls, ROOF_CLASSES)
        m = (cls == CLASS_GROUND) | roof
        if not m.any():
            continue
        x, y = M3857_TO_UTM.transform(np.asarray(las.x)[m], np.asarray(las.y)[m])
        z, cls = np.asarray(las.z)[m], cls[m]
        k = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)
        x, y, z, cls = x[k], y[k], z[k], cls[k]

        g = cls == CLASS_GROUND
        gi = (((maxy - y[g]) // GROUND_CELL_M).clip(0, gH - 1).astype(np.int64) * gW
              + ((x[g] - minx) // GROUND_CELL_M).clip(0, gW - 1).astype(np.int64))
        g_sum += np.bincount(gi, weights=z[g], minlength=gH * gW)
        g_cnt += np.bincount(gi, minlength=gH * gW)

        bb = np.isin(cls, ROOF_CLASSES)
        row = ((maxy - y[bb]) // ID_CELL_M).clip(0, H - 1).astype(np.int64)
        col = ((x[bb] - minx) // ID_CELL_M).clip(0, W - 1).astype(np.int64)
        bid = id_rast[row, col]
        kk = bid > 0
        b_idx.append(bid[kk] - 1)
        b_x.append(x[bb][kk]); b_y.append(y[bb][kk]); b_z.append(z[bb][kk])
    t_read = time.time() - t0

    dtm = np.full(gH * gW, np.nan)
    ok = g_cnt > 0
    dtm[ok] = g_sum[ok] / g_cnt[ok]
    dtm = dtm.reshape(gH, gW)
    if (~ok).any():
        _, (ri, ci) = ndimage.distance_transform_edt(np.isnan(dtm), return_indices=True)
        dtm = dtm[ri, ci]

    b_idx = np.concatenate(b_idx); b_x = np.concatenate(b_x)
    b_y = np.concatenate(b_y); b_z = np.concatenate(b_z)
    gr = ((maxy - b_y) // GROUND_CELL_M).clip(0, gH - 1).astype(np.int64)
    gc = ((b_x - minx) // GROUND_CELL_M).clip(0, gW - 1).astype(np.int64)
    zg = dtm[gr, gc]
    df = pd.DataFrame({"i": b_idx, "h": b_z - zg, "zg": zg})
    agg = df.groupby("i").agg(
        n_points=("h", "size"),
        height_max_m=("h", "max"),
        height_p99_m=("h", lambda v: np.percentile(v, 99)),
        height_p95_m=("h", lambda v: np.percentile(v, 95)),
        height_p50_m=("h", "median"),
        ground_elev_m=("zg", "median"),
    )
    res = pd.DataFrame({"building_id": a_ids, "area": r.area, "edge_clipped": edge.astype(np.int8),
                        "footprint_area_m2": shapely.area(a_polys)})
    res = res.join(agg, how="left")
    res["n_points"] = res["n_points"].fillna(0).astype(int)
    res["pt_density_per_m2"] = res["n_points"] / res["footprint_area_m2"]
    print(f"{r.area}: {len(files)} nodes, {n_total / 1e6:.0f}M pts, {len(res)} buildings, "
          f"{(res.n_points > 0).sum()} with building pts | read {t_read:.0f}s, "
          f"total {time.time() - t0:.0f}s", flush=True)
    return res


def height(areas, tag):
    sys.path.insert(0, ".")
    from extract_geometry_features import load_buildings  # same ids/polygons as features table
    ids, polys, _ = load_buildings()
    c = shapely.centroid(polys)
    cx, cy = shapely.get_x(c), shapely.get_y(c)
    parts = []
    for r in areas.itertuples():
        res = area_height(r, ids, polys, cx, cy)
        res.to_csv(f"{OUT_DIR}/{r.area}_heights_{VER}_20260922.csv", index=False)
        parts.append(res)
    allr = pd.concat(parts)
    meta = pd.read_csv(FEATURES, usecols=["building_id", "meta_has_streetview_match", "meta_levels_gt"])
    allr = allr.merge(meta, on="building_id", how="left")
    out = f"{OUT_DIR}/{tag}_heights_{VER}_20260922.csv"
    allr.to_csv(out, index=False)
    print(f"wrote {out} ({len(allr)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["download", "height"])
    ap.add_argument("--areas", required=True)
    ap.add_argument("--only", help="comma-separated area names (default: all)")
    ap.add_argument("--tag", default="all_areas", help="name of the combined output csv")
    a = ap.parse_args()
    areas = pd.read_csv(a.areas)
    if a.only:
        areas = areas[areas.area.isin(a.only.split(","))]
    download(areas) if a.mode == "download" else height(areas, a.tag)


if __name__ == "__main__":
    main()
