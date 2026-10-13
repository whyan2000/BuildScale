#!/usr/bin/env python3
"""
Resumable, adaptive-tile metadata harvest for all Mapillary images covering
Orange County, FL.

Why adaptive: Mapillary's per-request cap isn't a clean documented area
limit in practice -- small boxes in dense downtown Orlando still get
rejected ("reduce the amount of data"), while the same box size is fine in
the suburbs. So instead of a fixed grid, this recursively quarters any tile
that either (a) errors out after retries, or (b) comes back exactly at the
page limit (signals truncation -- there may be more images than we got).
Leaf tiles are appended to a JSONL file immediately (crash-safe / resumable
-- a completed-tiles index is checked before re-querying).

Metadata only (no image bytes yet) -- id, camera_type, compass_angle,
captured_at, sequence, computed_geometry. Actual image downloads are a
separate later step, prioritized by camera_type=spherical per the user.

Output:
  mapillary_metadata/leaf_tiles_done.txt      -- one bbox tuple per completed leaf (resume marker)
  mapillary_metadata/images_20260918.jsonl    -- one JSON image record per line, append-only
  mapillary_metadata/harvest.log              -- progress log
"""
import json
import time
import os
import sys
import requests

TOKEN_FILE = "token_maplliary.txt"
OUT_DIR = "mapillary_metadata"
DONE_FILE = f"{OUT_DIR}/leaf_tiles_done.txt"
IMAGES_FILE = f"{OUT_DIR}/images_20260918.jsonl"

SOUTH, NORTH, WEST, EAST = 28.346742, 28.786278, -81.658619, -80.8631553

START_TILE = 0.02        # deg, initial grid cell size
MIN_TILE = 0.0015        # deg (~150m), recursion floor -- give up below this
PAGE_LIMIT = 100
REQUEST_DELAY = 1.0      # seconds between requests (polite pacing)
MAX_RETRIES = 4          # for genuine transient errors (network, 5xx unrelated to size)
SIZE_ERROR_RETRIES = 1   # "reduce the amount of data" looks size-driven, not transient -- split fast

FIELDS = "id,camera_type,compass_angle,captured_at,sequence,computed_geometry"
BASE = "https://graph.mapillary.com/images"

os.makedirs(OUT_DIR, exist_ok=True)

with open(TOKEN_FILE) as f:
    TOKEN = f.read().strip()
HEADERS = {"Authorization": f"OAuth {TOKEN}"}

done_tiles = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        for line in f:
            done_tiles.add(line.strip())

stats = {"requests": 0, "images": 0, "leaf_tiles": 0, "gave_up": 0}


def tile_key(s, w, n, e):
    return f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"


def fetch(s, w, n, e):
    params = {"fields": FIELDS, "bbox": f"{w},{s},{e},{n}", "limit": PAGE_LIMIT}
    attempt = 0
    while True:
        stats["requests"] += 1
        try:
            resp = requests.get(BASE, headers=HEADERS, params=params, timeout=60)
        except Exception as ex:
            attempt += 1
            print(f"  request error: {ex} (attempt {attempt})", flush=True)
            if attempt >= MAX_RETRIES:
                return None
            time.sleep(3 * attempt)
            continue
        if resp.status_code == 200:
            return resp.json().get("data", [])

        body = resp.text[:150]
        is_size_error = resp.status_code == 500 and "reduce the amount of data" in body
        limit = SIZE_ERROR_RETRIES if is_size_error else MAX_RETRIES
        attempt += 1
        print(f"  HTTP {resp.status_code}: {body} (attempt {attempt}/{limit})", flush=True)
        if attempt >= limit:
            return None
        time.sleep(1 if is_size_error else 3 * attempt)


def process_tile(s, w, n, e, depth=0):
    key = tile_key(s, w, n, e)
    if key in done_tiles:
        return

    items = fetch(s, w, n, e)
    time.sleep(REQUEST_DELAY)

    width = e - w
    if items is None or len(items) >= PAGE_LIMIT:
        # failed, or likely truncated -> split, unless at the floor
        if width / 2 < MIN_TILE:
            if items is None:
                print(f"  GIVE UP (min tile size) at {key}", flush=True)
                stats["gave_up"] += 1
                with open(DONE_FILE, "a") as f:
                    f.write(key + "\n")
                done_tiles.add(key)
                return
            # at floor and got a full page: accept what we have, can't split further usefully
        else:
            mid_lat = (s + n) / 2
            mid_lon = (w + e) / 2
            process_tile(s, w, mid_lat, mid_lon, depth + 1)
            process_tile(s, mid_lon, mid_lat, e, depth + 1)
            process_tile(mid_lat, w, n, mid_lon, depth + 1)
            process_tile(mid_lat, mid_lon, n, e, depth + 1)
            return

    # leaf: accept and persist
    if items:
        with open(IMAGES_FILE, "a") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
    stats["images"] += len(items or [])
    stats["leaf_tiles"] += 1
    with open(DONE_FILE, "a") as f:
        f.write(key + "\n")
    done_tiles.add(key)

    if stats["leaf_tiles"] % 25 == 0:
        print(f"progress: {stats['leaf_tiles']} leaf tiles, {stats['images']} images, "
              f"{stats['requests']} requests, {stats['gave_up']} gave up", flush=True)


def main():
    lat = SOUTH
    while lat < NORTH:
        lon = WEST
        n = min(lat + START_TILE, NORTH)
        while lon < EAST:
            e = min(lon + START_TILE, EAST)
            process_tile(lat, lon, n, e)
            lon = e
        lat = n

    print("\n=== HARVEST DONE ===", flush=True)
    print(stats, flush=True)


if __name__ == "__main__":
    main()
