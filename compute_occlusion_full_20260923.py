#!/usr/bin/env python3
"""Footprint ray-cast occlusion for every in-county crop (09-23, user decision:
use this instead of a full Grounding DINO run, 工作记录_20260923【二十二】).
Same algorithm and constants as pilot_multitask_prep.py step 4: N_RAYS rays from the
camera across the target's azimuth span (clipped to the crop's hfov 90), a ray is
occluded if another OSM footprint is hit before the target. Only buildings block --
trees/vegetation are not in OSM, so tree occlusion is NOT detected.
Output: occlusion_v3_20260923.csv (building_id, image_id, occlusion, occl_rays)
"""
import json, math, os, sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from multiprocessing import Pool
import numpy as np, pandas as pd, shapely
from pyproj import Transformer
from shapely import STRtree
from extract_geometry_features import load_buildings

IMAGES = "mapillary_metadata/images_20260918_dedup.jsonl"
HFOV, N_RAYS, RAY_LEN = 90.0, 41, 80.0
TO_UTM = Transformer.from_crs(4326, 32617, always_xy=True)
ids, polys, _ = load_buildings()
idx_of = {b: i for i, b in enumerate(ids)}
tree = STRtree(polys)
polys = np.asarray(polys, dtype=object)


def one(p):
    cx, cy = p["cx"], p["cy"]
    poly = polys[idx_of[p["building_id"]]]
    yaw = p["bearing_deg"]
    xy = np.asarray(shapely.get_coordinates(poly))
    daz = (np.degrees(np.arctan2(xy[:, 0] - cx, xy[:, 1] - cy)) - yaw + 180) % 360 - 180
    lo_c, hi_c = max(daz.min(), -HFOV / 2), min(daz.max(), HFOV / 2)
    near = tree.query(shapely.Point(cx, cy).buffer(RAY_LEN))
    cam = shapely.Point(cx, cy)
    occ = valid = 0
    for a in np.linspace(lo_c, hi_c, N_RAYS):
        r = math.radians(yaw + a)
        ray = shapely.LineString([(cx, cy), (cx + RAY_LEN * math.sin(r), cy + RAY_LEN * math.cos(r))])
        d_t = d_o = None
        for j in near:
            g = polys[j]
            if not ray.intersects(g):
                continue
            d = cam.distance(ray.intersection(g))
            if ids[j] == p["building_id"]:
                d_t = d
            elif d_o is None or d < d_o:
                d_o = d
        if d_t is None:
            continue
        valid += 1
        occ += int(d_o is not None and d_o < d_t)
    return p["building_id"], p["image_id"], (occ / valid if valid else None), valid


def main():
    jobs = [json.loads(l) for l in open("pseudolabel_v2_jobs_20260923.jsonl")]
    keep = {j["building_id"] for j in jobs}
    man = [m for m in map(json.loads, open("all_crops_v3/manifest.jsonl")) if m["building_id"] in keep]
    need = {m["image_id"] for m in man}
    cams = {}
    for l in open(IMAGES):
        d = json.loads(l)
        if d["id"] in need:
            cams[d["id"]] = d["computed_geometry"]["coordinates"]
    todo = []
    for m in man:
        cx, cy = TO_UTM.transform(*cams[m["image_id"]])
        todo.append({"building_id": m["building_id"], "image_id": m["image_id"], "bearing_deg": m["bearing_deg"],
                     "cx": cx, "cy": cy})
    print(len(todo), "buildings", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as pool:
        res = pool.map(one, todo, chunksize=200)
    o = pd.DataFrame(res, columns=["building_id", "image_id", "occlusion", "occl_rays"])
    o.to_csv("occlusion_v3_20260923.csv", index=False)
    print("occlusion median", round(o.occlusion.median(), 3), "| none(no valid ray):", int(o.occlusion.isna().sum()),
          "| >0:", round((o.occlusion > 0).mean(), 3), "| >=0.5:", round((o.occlusion >= 0.5).mean(), 3),
          "| >=0.9:", round((o.occlusion >= 0.9).mean(), 3), flush=True)
    pil = pd.DataFrame([json.loads(l) for l in open("pilot_multitask_v3b_20260922.jsonl")])[["building_id", "occlusion"]]
    c = pil.merge(o, on="building_id", suffixes=("_pilot", "_full"))
    print("pilot check: n", len(c), "max abs diff", float((c.occlusion_pilot - c.occlusion_full).abs().max()), flush=True)


if __name__ == "__main__":
    main()
