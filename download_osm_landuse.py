#!/usr/bin/env python3
"""
Download OSM land-use / area polygons for Orange County, FL (Overpass), used
only to characterise 1 km grid cells when choosing LiDAR validation areas
(select_lidar_areas.py). Not a model feature.

Tags: landuse=residential/industrial/commercial/retail,
      amenity=university/college, tourism=theme_park/hotel/resort,
      leisure=resort
Output: osm_landuse_orange_county_20260922.json (Overpass JSON, out geom)
"""
import json
import time
import requests

S, N, W, E = 28.346742, 28.786278, -81.658619, -80.8631553
BB = f"({S},{W},{N},{E})"
Q = f"""[out:json][timeout:600];
(
  way["landuse"~"^(residential|industrial|commercial|retail)$"]{BB};
  relation["landuse"~"^(residential|industrial|commercial|retail)$"]{BB};
  way["amenity"~"^(university|college)$"]{BB};
  relation["amenity"~"^(university|college)$"]{BB};
  way["tourism"~"^(theme_park|hotel|resort)$"]{BB};
  relation["tourism"~"^(theme_park|hotel|resort)$"]{BB};
  way["leisure"="resort"]{BB};
  relation["leisure"="resort"]{BB};
);
out geom;
"""
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
HEADERS = {"User-Agent": "buildscale-research/1.0 (your-email@example.com)"}
OUT = "osm_landuse_orange_county_20260922.json"

for ep in ENDPOINTS:
    for attempt in range(3):
        try:
            r = requests.post(ep, data=Q.encode(), headers=HEADERS, timeout=700)
            if r.status_code == 200:
                d = r.json()
                json.dump(d, open(OUT, "w"))
                print(f"OK {ep}: {len(d['elements'])} elements -> {OUT}")
                raise SystemExit
            print(f"{ep} HTTP {r.status_code} (attempt {attempt+1})", flush=True)
        except (requests.RequestException, ValueError) as ex:
            print(f"{ep} {ex} (attempt {attempt+1})", flush=True)
        time.sleep(10 * (attempt + 1))
raise SystemExit("FAILED")
