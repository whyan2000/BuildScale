#!/usr/bin/env python3
"""
Download OSM land-use / facility area polygons for Orange County, FL (Overpass) -- 10-05, for the
downstream neighbourhood ("context") features. Broader than osm_landuse_orange_county_20260922.json
(that one was only for choosing LiDAR areas and lacks schools / churches / government / hospitals).

Areas (ways + relations, out geom):
  landuse=*                                    (all values; residential, commercial, retail, industrial,
                                                religious, cemetery, farmland, military, ...)
  amenity=school|university|college|kindergarten|hospital|clinic|place_of_worship|townhall|
          courthouse|police|fire_station|prison|library|community_centre|parking
  leisure=park|golf_course|stadium|sports_centre|pitch|resort
  tourism=theme_park|hotel|resort|attraction|zoo
  aeroway=aerodrome
  military=*
Output: osm_context_areas_orange_county_20261005.json (Overpass JSON)
"""
import json
import time
import requests

S, N, W, E = 28.346742, 28.786278, -81.658619, -80.8631553
BB = f"({S},{W},{N},{E})"
SEL = [
    '["landuse"]',
    '["amenity"~"^(school|university|college|kindergarten|hospital|clinic|place_of_worship|townhall|'
    'courthouse|police|fire_station|prison|library|community_centre|parking)$"]',
    '["leisure"~"^(park|golf_course|stadium|sports_centre|pitch|resort)$"]',
    '["tourism"~"^(theme_park|hotel|resort|attraction|zoo)$"]',
    '["aeroway"="aerodrome"]',
    '["military"]',
]
Q = "[out:json][timeout:900];\n(\n" + "".join(f"  way{s}{BB};\n  relation{s}{BB};\n" for s in SEL) + ");\nout geom;\n"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "buildscale-research/1.0 (your-email@example.com)"}
OUT = "osm_context_areas_orange_county_20261005.json"

for ep in ENDPOINTS:
    for attempt in range(3):
        try:
            r = requests.post(ep, data=Q.encode(), headers=HEADERS, timeout=1000)
            if r.status_code == 200:
                d = r.json()
                json.dump(d, open(OUT, "w"))
                print(f"{ep}: {len(d['elements'])} elements -> {OUT}")
                raise SystemExit(0)
            print(f"{ep} attempt {attempt}: HTTP {r.status_code}")
        except (requests.RequestException, ValueError) as e:
            print(f"{ep} attempt {attempt}: {e}")
        time.sleep(30)
raise SystemExit("all endpoints failed")
