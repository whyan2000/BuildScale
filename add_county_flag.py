#!/usr/bin/env python3
"""
09-22 county fix: osm_buildings_orange_county_20260918.json was downloaded by
bbox rectangle, so 25% of its buildings are outside Orange County. Street view
was only harvested inside the county and the Orange EPT LiDAR only covers the
county, so the study set is buildings INSIDE the Orange County boundary (OSM
relation 389011, orange_county_boundary_20260922.json).

This writes a copy of the feature table with one extra column meta_in_orange
(all rows kept; neighbourhood features still count out-of-county neighbours,
since those buildings physically exist). Training/eval should filter on it.
Also prints the corrected coverage numbers and the in-county pseudolabel count.
"""
import json

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

IN = "features/building_features_20260922.csv.gz"
OUT = "features/building_features_county_20260922.csv.gz"
TO_UTM = Transformer.from_crs(4326, 32617, always_xy=True)

el = json.load(open("orange_county_boundary_20260922.json"))["elements"][0]
lines = []
for m in el["members"]:
    if m.get("role") == "outer" and "geometry" in m:
        x, y = TO_UTM.transform([p["lon"] for p in m["geometry"]], [p["lat"] for p in m["geometry"]])
        lines.append(shapely.LineString(list(zip(x, y))))
county = shapely.union_all(shapely.get_parts(shapely.polygonize(lines)))
shapely.prepare(county)

d = pd.read_csv(IN, low_memory=False)
d["meta_in_orange"] = shapely.contains_xy(county, d.f_x_utm.to_numpy(), d.f_y_utm.to_numpy()).astype(np.int8)
d.to_csv(OUT, index=False, compression="gzip")

c = d[d.meta_in_orange == 1]
print(f"wrote {OUT}: {len(d)} rows, {len(c)} inside Orange County ({len(d) - len(c)} outside)")
print(f"street-view match inside county: {c.meta_has_streetview_match.sum()} "
      f"({100 * c.meta_has_streetview_match.mean():.2f}%), unseen {100 * (1 - c.meta_has_streetview_match.mean()):.2f}%")
print(f"numeric levels GT inside county: {c.meta_levels_gt.notna().sum()}")

pl = pd.read_json("pseudolabels.jsonl", lines=True)
inside_ids = set(c.building_id)
k = pl.building_id.isin(inside_ids)
print(f"pseudolabels: {len(pl)} total, {k.sum()} inside county, {(~k).sum()} outside")
