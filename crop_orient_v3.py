#!/usr/bin/env python3
"""
v3 crops with the CORRECTED panorama orientation (09-22).

orient_test.py showed the Mapillary equirect centre column = computed_compass_angle
(SfM-corrected heading), not true north. So the panorama column-angle of a true
azimuth a is
    col_deg = (a - computed_compass_angle + 180) mod 360
and crop_perspective() (which samples src column = yaw/360 * W) is called with
yaw = col_deg instead of the true bearing. Crop geometry is unchanged from v2
(cylindrical, hfov 90, vfov 140, pitch 15, 640x1000), so boxes/projection code
only needs the same substitution.

Same building -> image selection as before (all_matches_sample.jsonl: nearest
match per building). Buildings whose image has no computed_compass_angle are
skipped and counted. Old all_crops/ is kept untouched.

Output: all_crops_v3/b<building_id>.jpg + all_crops_v3/manifest.jsonl
(adds computed_compass, yaw_col_deg)
"""
import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from functools import partial
from multiprocessing import Pool

from PIL import Image

from equirect_to_perspective import crop_perspective

PANOS = "mapillary_metadata/panorama_images"
ORIENT = "mapillary_metadata/image_orientation_20260922.jsonl"


def col_deg(bearing, ccompass):
    return (bearing - ccompass + 180.0) % 360.0


def one(r, out_dir, a):
    out = os.path.join(out_dir, f"b{r['building_id']}.jpg")
    if not os.path.exists(out):
        src = os.path.join(PANOS, f"{r['image_id']}.jpg")
        if not os.path.exists(src):
            return None
        im = Image.open(src)
        crop_perspective(im, r["yaw_col_deg"], a.pitch, a.hfov, a.out_w, a.out_h,
                         mode="cylindrical", vfov_deg=a.vfov).save(out)
    return {**r, "crop_file": os.path.basename(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="all_matches_sample.jsonl")
    ap.add_argument("--out_dir", default="all_crops_v3")
    ap.add_argument("--only_ids", help="optional file with building ids (one per line) to crop")
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--vfov", type=float, default=140.0)
    ap.add_argument("--pitch", type=float, default=15.0)
    ap.add_argument("--out_w", type=int, default=640)
    ap.add_argument("--out_h", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    cc = {}
    for l in open(ORIENT):
        d = json.loads(l)
        if d.get("computed_compass_angle") is not None:
            cc[d["id"]] = d["computed_compass_angle"]
    rows = [json.loads(l) for l in open(a.matches)]
    if a.only_ids:
        keep = {int(x) for x in open(a.only_ids) if x.strip()}
        rows = [r for r in rows if r["building_id"] in keep]
    todo, missing = [], 0
    for r in rows:
        c = cc.get(r["image_id"])
        if c is None:
            missing += 1
            continue
        todo.append({**r, "computed_compass": round(c, 3), "yaw_col_deg": round(col_deg(r["bearing_deg"], c), 3)})
    print(f"{len(rows)} buildings, {len(todo)} with computed_compass, {missing} skipped (no computed_compass)",
          flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    man = []
    with Pool(a.workers) as p:
        for i, res in enumerate(p.imap_unordered(partial(one, out_dir=a.out_dir, a=a), todo, chunksize=50)):
            if res:
                man.append(res)
            if (i + 1) % 5000 == 0:
                print(f"[{i + 1}/{len(todo)}]", flush=True)
    with open(os.path.join(a.out_dir, "manifest.jsonl"), "w") as f:
        for m in man:
            f.write(json.dumps(m) + "\n")
    print(f"wrote {len(man)} crops -> {a.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
