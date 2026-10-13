#!/usr/bin/env python3
"""
Download OSM POIs / use tags for Orange County (Overpass, county-area query),
used as WEAK ground truth for the fine building-use classes and ground-floor
retail (taxonomy_nhts_20260922.json), and for stratified pilot sampling.
Not a model feature. Nodes + ways (ways as centroids via 'out center').
Output: osm_pois_orange_county_20260922.json
"""
import json, time, requests

KEYS = ["amenity", "shop", "office", "healthcare", "leisure", "tourism", "craft",
        "social_facility", "public_transport", "railway", "aeroway", "parking"]
body = "\n".join(f'  node["{k}"](area.c);\n  way["{k}"](area.c);' for k in KEYS)
Q = f"""[out:json][timeout:900];
relation(389011);map_to_area->.c;
(
{body}
);
out center tags;
"""
EP = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
H = {"User-Agent": "buildscale-research/1.0 (your-email@example.com)"}
OUT = "osm_pois_orange_county_20260922.json"
for ep in EP:
    for i in range(3):
        try:
            r = requests.post(ep, data=Q.encode(), headers=H, timeout=1000)
            if r.status_code == 200:
                d = r.json(); json.dump(d, open(OUT, "w"))
                print(f"OK {ep}: {len(d['elements'])} elements -> {OUT}"); raise SystemExit
            print(f"{ep} HTTP {r.status_code}", flush=True)
        except (requests.RequestException, ValueError) as e:
            print(f"{ep} {e}", flush=True)
        time.sleep(15 * (i + 1))
raise SystemExit("FAILED")
