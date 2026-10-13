#!/usr/bin/env python3
"""
Download OSM building footprints for Orange County, FL via Overpass API.

The single-shot full-county query got a 504 (dispatcher timeout, server too
busy) from the main overpass-api.de instance. This version splits the county
bbox into a grid of smaller tiles, queries each tile separately (trying a
couple of public mirrors with retry/backoff), and merges the results into
one deduplicated file.

Bbox (Orange County, FL; from Nominatim, OSM relation 389011):
  south=28.346742 north=28.786278 west=-81.658619 east=-80.8631553
"""
import json
import time
import sys
import requests

SOUTH, NORTH, WEST, EAST = 28.346742, 28.786278, -81.658619, -80.8631553
GRID_ROWS, GRID_COLS = 4, 4

ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

HEADERS = {"User-Agent": "buildscale-research/1.0 (your-email@example.com)"}

OUT_DIR = "osm_tiles"
MERGED_OUT = "osm_buildings_orange_county_20260918.json"

import os
os.makedirs(OUT_DIR, exist_ok=True)


def query_for_tile(s, w, n, e):
    return f"""[out:json][timeout:180];
(
  way["building"]({s},{w},{n},{e});
  relation["building"]({s},{w},{n},{e});
);
out geom;
"""


def fetch_tile(s, w, n, e, tile_id):
    q = query_for_tile(s, w, n, e)
    for endpoint in ENDPOINTS:
        for attempt in range(3):
            try:
                resp = requests.post(endpoint, data=q.encode("utf-8"), headers=HEADERS, timeout=200)
                if resp.status_code == 200:
                    data = resp.json()
                    n_el = len(data.get("elements", []))
                    print(f"tile {tile_id}: OK from {endpoint} ({n_el} elements)", flush=True)
                    return data
                else:
                    print(f"tile {tile_id}: {endpoint} -> HTTP {resp.status_code} (attempt {attempt+1})", flush=True)
            except Exception as ex:
                print(f"tile {tile_id}: {endpoint} -> {ex} (attempt {attempt+1})", flush=True)
            time.sleep(5 * (attempt + 1))
    print(f"tile {tile_id}: FAILED on all endpoints", flush=True)
    return None


def main():
    lat_step = (NORTH - SOUTH) / GRID_ROWS
    lon_step = (EAST - WEST) / GRID_COLS

    all_elements = {}
    failed_tiles = []

    tile_id = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            tile_id += 1
            s = SOUTH + r * lat_step
            n = SOUTH + (r + 1) * lat_step
            w = WEST + c * lon_step
            e = WEST + (c + 1) * lon_step

            tile_file = f"{OUT_DIR}/tile_{tile_id:02d}.json"
            data = fetch_tile(s, w, n, e, tile_id)
            if data is None:
                failed_tiles.append((tile_id, s, w, n, e))
                continue

            with open(tile_file, "w") as f:
                json.dump(data, f)

            for el in data.get("elements", []):
                key = (el["type"], el["id"])
                all_elements[key] = el

            time.sleep(3)

    print(f"\n=== merge: {len(all_elements)} unique building elements ===", flush=True)
    merged = {"elements": list(all_elements.values())}
    with open(MERGED_OUT, "w") as f:
        json.dump(merged, f)
    print(f"Wrote: {MERGED_OUT}", flush=True)

    if failed_tiles:
        print(f"\nFAILED TILES ({len(failed_tiles)}): {failed_tiles}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
