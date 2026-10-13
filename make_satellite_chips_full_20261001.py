#!/usr/bin/env python3
"""BuildScale P2 (10-01): NAIP satellite chips for the VLM, full set.

Buildings: every in-county building with a street-view match (36,491; for the cross-modal agreement
signal) + all 8,125 buildings of the 8 LiDAR areas (full-VLM-inference baseline; 1,133 overlap)
-> 43,483 chips.
Chip rules identical to the 09-22 / 10-01 satellite test (make_satellite_chips.py): NAIP 2023 0.3 m
from Microsoft Planetary Computer, north-up, centred on the footprint centroid, side
max(60 m, 3 x footprint long side) capped at 250 m, resized to 512 px (LANCZOS), target footprint
outlined in red (width 3), PNG.
Speed: one STAC search for all 2023 NAIP items over the county (instead of one per building); each
building is assigned to an item whose footprint fully contains its chip box (chips crossing item
edges are mosaicked from all intersecting items with rasterio.merge); items are processed in
parallel, each COG opened once and read by windows.
Output: sat_chips_full_20261001/b<id>.png + manifest.jsonl (resumable: done buildings are skipped).
Run from the project root (CPU job, needs internet).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import rasterio
import shapely
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.windows import from_bounds

from extract_geometry_features import load_buildings
from make_satellite_chips import fetch_json, signed, STAC, NAIP_YEAR, MIN_SIDE_M, MAX_SIDE_M, OUT_PX

OUT = "sat_chips_full_20261001"
UTM_TO_LL = Transformer.from_crs(32617, 4326, always_xy=True)
G = {}  # per-worker globals


def county_items(lonlat_box):
    x0, y0, x1, y1 = lonlat_box
    body = {"collections": ["naip"], "bbox": [x0, y0, x1, y1], "limit": 1000,
            "datetime": f"{NAIP_YEAR}-01-01/{NAIP_YEAR}-12-31"}
    feats = []
    d = fetch_json(STAC, body)
    feats += d["features"]
    while any(l.get("rel") == "next" for l in d.get("links", [])):
        nxt = next(l for l in d["links"] if l.get("rel") == "next")
        d = fetch_json(nxt["href"], nxt.get("body"))
        feats += d["features"]
    return feats


def draw_chip(arr, poly_img, cx, cy, side):
    img = Image.fromarray(np.moveaxis(arr, 0, -1)).resize((OUT_PX, OUT_PX), Image.LANCZOS)
    h = side / 2
    scale = OUT_PX / side
    draw = ImageDraw.Draw(img)
    for ring in (poly_img.geoms if poly_img.geom_type == "MultiPolygon" else [poly_img]):
        draw.line([((x - (cx - h)) * scale, ((cy + h) - y) * scale) for x, y in ring.exterior.coords],
                  fill=(255, 0, 0), width=3)
    return img


def chip_box(poly_img, long_m):
    c = poly_img.centroid
    side = float(np.clip(3 * long_m, MIN_SIDE_M, MAX_SIDE_M))
    return c.x, c.y, side


def work(task):
    """task = (item_hrefs, item_ids, [(building_id, wkb_utm, long_m, groups), ...])"""
    hrefs, item_ids, rows = task
    out = []
    srcs = [rasterio.open(signed(h)) for h in hrefs]
    try:
        crs = srcs[0].crs
        to_img = Transformer.from_crs("EPSG:32617", crs, always_xy=True)
        for b, wkb, long_m, groups in rows:
            poly = shapely.from_wkb(wkb)
            poly_img = shapely.transform(poly, lambda c: np.column_stack(to_img.transform(c[:, 0], c[:, 1])))
            cx, cy, side = chip_box(poly_img, long_m)
            h = side / 2
            for attempt in range(3):
                try:
                    if len(srcs) == 1:
                        arr = srcs[0].read([1, 2, 3], window=from_bounds(cx - h, cy - h, cx + h, cy + h,
                                                                         srcs[0].transform),
                                           boundless=True, fill_value=0)
                    else:
                        arr, _ = merge(srcs, bounds=(cx - h, cy - h, cx + h, cy + h), indexes=[1, 2, 3],
                                       res=srcs[0].res)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5)
            fn = f"b{b}.png"
            draw_chip(arr, poly_img, cx, cy, side).save(os.path.join(OUT, fn))
            out.append({"building_id": int(b), "crop_file": fn, "chip_side_m": round(side, 1),
                        "naip_items": item_ids, "groups": groups,
                        "black_frac": round(float((arr.sum(0) == 0).mean()), 4)})
    finally:
        for s in srcs:
            s.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N buildings (test)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "manifest.jsonl")
    done = {json.loads(l)["building_id"] for l in open(man_path)} if os.path.exists(man_path) else set()

    feat = pd.read_csv("features/building_features_county_20260922.csv.gz", low_memory=False,
                       usecols=["building_id", "meta_in_orange", "meta_has_streetview_match", "f_mrr_long_m"])
    lid = pd.read_csv("lidar/ept_areas/final8_county_heights_c6_20260928.csv")
    sv = set(feat.building_id[(feat.meta_in_orange == 1) & (feat.meta_has_streetview_match == 1)])
    ar = set(lid.building_id)
    target = sorted(sv | ar)
    if a.limit:
        target = target[:a.limit]
    long_m = dict(zip(feat.building_id, feat.f_mrr_long_m))
    todo = [b for b in target if b not in done]
    print(f"target {len(target)} (street view {len(sv)}, LiDAR areas {len(ar)}), done {len(done)}, todo {len(todo)}",
          flush=True)
    if not todo:
        return

    ids, polys, _ = load_buildings()
    pos = {b: i for i, b in enumerate(ids)}
    tp = [polys[pos[b]] for b in todo]

    # chip boxes in lon/lat (approximate: UTM box -> lon/lat bbox) for item assignment
    cen = shapely.centroid(tp)
    sides = np.clip(3 * np.array([long_m[b] for b in todo]), MIN_SIDE_M, MAX_SIDE_M)
    x, y = shapely.get_x(cen), shapely.get_y(cen)
    boxes_utm = shapely.box(x - sides / 2, y - sides / 2, x + sides / 2, y + sides / 2)
    boxes_ll = shapely.transform(boxes_utm, lambda c: np.column_stack(UTM_TO_LL.transform(c[:, 0], c[:, 1])))
    b0 = shapely.total_bounds(boxes_ll)
    items = county_items(b0)
    print(f"NAIP {NAIP_YEAR} items over the target area: {len(items)}", flush=True)
    geoms = [shapely.geometry.shape(it["geometry"]) for it in items]
    tree = shapely.STRtree(geoms)

    groups = {}
    for k, (b, box) in enumerate(zip(todo, boxes_ll)):
        cand = tree.query(box, predicate="intersects")
        full = [j for j in cand if geoms[j].contains(box)]
        if full:
            # among items that fully contain the chip, prefer the finest gsd, then deepest inside
            j = min(full, key=lambda j: (items[j]["properties"]["gsd"],
                                         -geoms[j].exterior.distance(box.centroid)))
            key = (j,)
        elif len(cand):
            key = tuple(sorted(cand))
        else:
            print(f"  {b}: no NAIP item", flush=True)
            continue
        g = ["sv"] * (b in sv) + ["lidar_area"] * (b in ar)
        groups.setdefault(key, []).append((b, shapely.to_wkb(tp[k]), long_m[b], g))
    n_multi = sum(len(v) for k, v in groups.items() if len(k) > 1)
    print(f"{len(groups)} item groups; {n_multi} chips need a mosaic of several items", flush=True)

    # split big groups so the pool stays balanced
    tasks = []
    for key, rows in groups.items():
        hrefs = [items[j]["assets"]["image"]["href"] for j in key]
        iids = [items[j]["id"] for j in key]
        for s in range(0, len(rows), 200):
            tasks.append((hrefs, iids, rows[s:s + 200]))
    t0, n = time.time(), 0
    with open(man_path, "a") as man, Pool(a.workers) as pool:
        for res in pool.imap_unordered(work, tasks):
            for r in res:
                man.write(json.dumps(r) + "\n")
            man.flush()
            n += len(res)
            if n % 2000 < len(res):
                print(f"  {n}/{len(todo)} chips, {time.time() - t0:.0f}s", flush=True)
    print(f"done {n} chips in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
