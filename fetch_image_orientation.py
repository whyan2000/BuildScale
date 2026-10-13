#!/usr/bin/env python3
"""
Fetch computed_compass_angle (Mapillary's SfM-corrected camera heading) +
make/model for every downloaded panorama (mapillary_metadata/matched_image_ids.txt).

Why (09-22): orient_test.py showed the equirect CENTRE column = computed_compass_angle
(median error 3.6 deg); the 09-21 "north-aligned" crop convention was wrong and the
raw compass_angle is unreliable. Crops must be regenerated with
  column = (azimuth - computed_compass_angle + 180) / 360.

Batches of 50 ids via /images?image_ids=..., a few threads, resumable (skips ids
already in the output). Output: mapillary_metadata/image_orientation_20260922.jsonl
"""
import json, os, time
from concurrent.futures import ThreadPoolExecutor
import requests

IDS = "mapillary_metadata/matched_image_ids.txt"
OUT = "mapillary_metadata/image_orientation_20260922.jsonl"
FIELDS = "id,computed_compass_angle,compass_angle,make,model,camera_type"
BATCH, THREADS = 50, 4
H = {"Authorization": f"OAuth {open('token_maplliary.txt').read().strip()}"}

ids = [l.strip() for l in open(IDS) if l.strip()]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT)}
todo = [i for i in ids if i not in done]
batches = [todo[k:k + BATCH] for k in range(0, len(todo), BATCH)]
print(f"{len(ids)} ids, {len(done)} done, {len(batches)} batches to fetch", flush=True)


def get(batch):
    for a in range(5):
        try:
            r = requests.get("https://graph.mapillary.com/images", headers=H, timeout=60,
                             params={"image_ids": ",".join(batch), "fields": FIELDS})
            if r.status_code == 200:
                return r.json().get("data", [])
            err = f"HTTP {r.status_code} {r.text[:120]}"
        except Exception as e:
            err = str(e)
        time.sleep(3 * (a + 1))
    print(f"gave up on batch starting {batch[0]}: {err}", flush=True)
    return []


n = 0
t0 = time.time()
with open(OUT, "a") as f, ThreadPoolExecutor(THREADS) as ex:
    for k, rows in enumerate(ex.map(get, batches)):
        for r in rows:
            f.write(json.dumps(r) + "\n")
        n += len(rows)
        f.flush()
        if k % 200 == 0:
            print(f"[{k}/{len(batches)}] {n} rows, {time.time() - t0:.0f}s", flush=True)
print(f"done: {n} new rows -> {OUT}", flush=True)
