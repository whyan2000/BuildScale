"""BuildScale S1 (10-01): OSM-only labels for every in-county building, in the Hazus scheme.

Height: OSM building:levels -> Hazus bins L = 1-3 / M = 4-7 / H = 8+ stories.
  Levels < 1 (e.g. 0.1, data errors) are treated as missing. The OSM `height` tag (metres) is kept
  in its own column (osm_height_m). v2 (user decision 10-01 "换算一下"): buildings WITHOUT levels get
  stories = height / STOREY_M (4.0 m = median height/levels of the 77 buildings carrying both tags;
  4.25 / 4.0 / 3.75 for the L / M / H groups), height_src = "height_tag". Not converted: building=house
  (1,255 bulk-imported values of 11-16 m with 2 decimals in one southern cluster -- most likely feet,
  not checkable against LiDAR; in metres ~300 houses would become 4-7 stories) and heights < 2 m.
  v1 output (levels only) = osm_labels_20261001.csv, v2 = osm_labels_20261001_v2.csv.
Use: two levels, user decision 10-01 "不光是大类，到小类，没有反正就空着":
  use_major  RES / COM / IND / AGR / REL / GOV / EDU / ACC (accessory, not a Hazus class)
  use_fine   Hazus occupancy class (RES1..RES6, COM1..COM10, IND1..IND6, AGR1, REL1, GOV1-2, EDU1-2);
             empty when OSM only tells the major class (e.g. building=commercial / industrial /
             residential) or when the evidence disagrees.
  Sources, in order:
    1. the building's own non-`building` tags (amenity/shop/office/...): most specific, e.g.
       building=commercial + amenity=restaurant -> COM8
    2. the building's own `building` tag
    3. POIs (osm_pois_orange_county_20260922.json) inside the footprint, only when 1-2 give no fine
       class: used only if all POIs agree on the fine class (and on the major class when 1-2 gave one).
       Shopping plazas with mixed POIs therefore keep only the major class (09-23 finding: POIs in
       plazas / mixed-use buildings are unreliable).
  Own tags (1-2) give a fine class -> POIs are ignored (e.g. building=apartments with a shop inside
  stays RES3).
Output: buildscale/osm_labels_20261001.csv
Run from the project root (~3 GB RAM).
"""
import json
import os
from collections import Counter

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely import STRtree

from extract_geometry_features import load_buildings

POIS = "osm_pois_orange_county_20260922.json"
FEATURES = "features/building_features_county_20260922.csv.gz"
OUT = "buildscale/osm_labels_20261001_v2.csv"
STOREY_M, MIN_HEIGHT_M = 4.0, 2.0
NO_HEIGHT_CONVERT = {"house"}
TO_UTM = Transformer.from_crs(4326, 32617, always_xy=True)

# (key, value) -> Hazus fine class ("" = major class only). "*" = any value.
RULES_OTHER = {
    # residential
    ("social_facility", "nursing_home"): "RES6", ("social_facility", "assisted_living"): "RES6",
    ("amenity", "nursing_home"): "RES6",
    ("tourism", "hotel"): "RES4", ("tourism", "motel"): "RES4", ("tourism", "hostel"): "RES4",
    ("tourism", "guest_house"): "RES4", ("tourism", "apartment"): "RES4",
    # commercial
    ("amenity", "hospital"): "COM6", ("healthcare", "hospital"): "COM6",
    ("amenity", "clinic"): "COM7", ("amenity", "doctors"): "COM7", ("amenity", "dentist"): "COM7",
    ("healthcare", "*"): "COM7",
    ("amenity", "bank"): "COM5", ("amenity", "bureau_de_change"): "COM5",
    ("amenity", "restaurant"): "COM8", ("amenity", "fast_food"): "COM8", ("amenity", "cafe"): "COM8",
    ("amenity", "bar"): "COM8", ("amenity", "pub"): "COM8", ("amenity", "ice_cream"): "COM8",
    ("amenity", "nightclub"): "COM8", ("leisure", "fitness_centre"): "COM8",
    ("leisure", "sports_centre"): "COM8", ("leisure", "sports_hall"): "COM8", ("leisure", "stadium"): "COM8",
    ("leisure", "bowling_alley"): "COM8", ("tourism", "attraction"): "COM8",
    ("tourism", "theme_park"): "COM8", ("tourism", "museum"): "COM8",
    ("amenity", "cinema"): "COM9", ("amenity", "theatre"): "COM9", ("amenity", "arts_centre"): "COM9",
    ("amenity", "parking"): "COM10", ("parking", "multi-storey"): "COM10",
    ("amenity", "car_wash"): "COM3", ("amenity", "veterinary"): "COM3",
    ("shop", "hairdresser"): "COM3", ("shop", "beauty"): "COM3", ("shop", "laundry"): "COM3",
    ("shop", "dry_cleaning"): "COM3", ("shop", "car_repair"): "COM3", ("shop", "tyres"): "COM3",
    ("shop", "massage"): "COM3", ("shop", "tattoo"): "COM3",
    ("amenity", "fuel"): "COM1", ("amenity", "pharmacy"): "COM1",
    ("shop", "wholesale"): "COM2", ("shop", "storage_rental"): "COM2",
    ("shop", "*"): "COM1",
    ("office", "government"): "GOV1", ("office", "*"): "COM4",
    # civic / education / religion
    ("amenity", "townhall"): "GOV1", ("amenity", "post_office"): "GOV1", ("amenity", "courthouse"): "GOV1",
    ("amenity", "fire_station"): "GOV2", ("amenity", "police"): "GOV2",
    ("amenity", "school"): "EDU1", ("amenity", "library"): "EDU1",
    ("amenity", "university"): "EDU2", ("amenity", "college"): "EDU2",
    ("amenity", "place_of_worship"): "REL1", ("amenity", "community_centre"): "REL1",
    ("amenity", "social_centre"): "REL1",
    # industrial (Hazus IND1-6 cannot be told apart from OSM tags -> major only)
    ("craft", "*"): "",
}
# children's daycare: Hazus subclass unclear -> major only
RULES_OTHER[("amenity", "childcare")] = ""
RULES_OTHER[("amenity", "kindergarten")] = ""

RULES_BUILDING = {
    "house": "RES1", "detached": "RES1", "bungalow": "RES1", "cabin": "RES1", "farm": "RES1",
    "static_caravan": "RES2", "mobile_home": "RES2", "trailer": "RES2",
    "apartments": "RES3", "terrace": "RES3", "semidetached_house": "RES3", "duplex": "RES3",
    "residential": "",
    "hotel": "RES4", "dormitory": "RES5",
    "retail": "COM1", "supermarket": "COM1", "kiosk": "COM1", "warehouse": "COM2",
    "office": "COM4", "hospital": "COM6", "parking": "COM10", "commercial": "",
    "industrial": "", "manufacture": "",
    "barn": "AGR1", "greenhouse": "AGR1", "farm_auxiliary": "AGR1", "stable": "AGR1",
    "church": "REL1", "chapel": "REL1", "mosque": "REL1", "synagogue": "REL1", "temple": "REL1",
    "religious": "REL1",
    "government": "GOV1", "public": "GOV1", "civic": "GOV1", "fire_station": "GOV2",
    "school": "EDU1", "university": "EDU2", "college": "EDU2", "kindergarten": "",
    "garage": "ACC", "garages": "ACC", "shed": "ACC", "carport": "ACC", "roof": "ACC",
    "boathouse": "ACC", "outbuilding": "ACC",
}
# major class of a "major only" rule
MAJOR_ONLY = {("building", "residential"): "RES", ("building", "commercial"): "COM",
              ("building", "industrial"): "IND", ("building", "manufacture"): "IND",
              ("building", "kindergarten"): "EDU", ("craft", "*"): "IND",
              ("amenity", "childcare"): "EDU", ("amenity", "kindergarten"): "EDU"}


def major_of(fine):
    return "ACC" if fine == "ACC" else fine.rstrip("0123456789")


def match_other(tags):
    """Fine/major from non-building keys. Returns (major, fine) or (None, None). Exact value
    rules beat wildcard rules; among several keys the first specific (fine) hit wins."""
    best = (None, None)
    for (k, v), fine in RULES_OTHER.items():
        if v == "*" or k not in tags or tags[k] != v:
            continue
        if fine:
            return major_of(fine), fine
        best = best if best[0] else (MAJOR_ONLY[(k, v)], "")
    for (k, v), fine in RULES_OTHER.items():
        if v != "*" or k not in tags:
            continue
        if fine:
            return major_of(fine), fine
        best = best if best[0] else (MAJOR_ONLY[(k, v)], "")
    return best


def match_building(tags):
    b = tags.get("building")
    if b not in RULES_BUILDING:
        return None, None
    fine = RULES_BUILDING[b]
    return (major_of(fine), fine) if fine else (MAJOR_ONLY[("building", b)], "")


def label_use(tags, poi_tags):
    om, of = match_other(tags)
    bm, bf = match_building(tags)
    if of:
        return om, of, "own_tag"
    if bf:
        return bm, bf, "building_tag"
    major = om or bm                                  # own tags give at most a major class
    src = "own_tag" if om else "building_tag" if bm else None
    hits = [h for h in (match_other(t) for t in poi_tags) if h[0]]
    fines = {f for _, f in hits if f}
    majors = {m for m, _ in hits}
    if hits and len(fines) == 1 and all(f for _, f in hits):
        f = fines.pop()
        if major is None or major_of(f) == major:
            return major_of(f), f, "poi"
    if major is None and len(majors) == 1:
        return majors.pop(), "", "poi_major"
    return major, "", src


def height_bin(levels):
    if not np.isfinite(levels) or levels < 1:
        return ""
    s = int(round(levels))
    return "L" if s <= 3 else "M" if s <= 7 else "H"


def parse_height(v):
    try:
        return float(str(v).lower().replace("m", "").strip())
    except ValueError:
        return np.nan


def main():
    ids, polys, tags_list = load_buildings()
    feat = pd.read_csv(FEATURES, usecols=["building_id", "meta_in_orange", "meta_levels_gt"])
    in_c = set(feat.building_id[feat.meta_in_orange == 1])
    levels = dict(zip(feat.building_id, feat.meta_levels_gt))

    els = json.load(open(POIS))["elements"]
    lon = [e["lon"] if "lon" in e else e["center"]["lon"] for e in els]
    lat = [e["lat"] if "lat" in e else e["center"]["lat"] for e in els]
    px, py = TO_UTM.transform(lon, lat)
    pi, bi = STRtree(polys).query(shapely.points(px, py), predicate="intersects")
    pois_in = {}
    for p, b in zip(pi, bi):
        pois_in.setdefault(b, []).append(els[p].get("tags", {}))

    rows = []
    for i, b in enumerate(ids):
        if b not in in_c:
            continue
        t = tags_list[i]
        lv = levels.get(b, np.nan)
        major, fine, src = label_use(t, pois_in.get(i, []))
        h = parse_height(t["height"]) if "height" in t else np.nan
        hb, hsrc = height_bin(lv), "levels"
        if not hb:
            hsrc = ""
            if np.isfinite(h) and h >= MIN_HEIGHT_M and t.get("building") not in NO_HEIGHT_CONVERT:
                hb, hsrc = height_bin(max(1.0, h / STOREY_M)), "height_tag"
        rows.append({"building_id": b, "osm_building": t.get("building"),
                     "osm_levels": lv if np.isfinite(lv) and lv >= 1 else np.nan,
                     "osm_height_m": h, "height_bin": hb, "height_src": hsrc, "use_major": major or "", "use_fine": fine or "",
                     "use_src": src or "", "n_pois": len(pois_in.get(i, []))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    n = len(df)
    print(f"in-county buildings {n}")
    print(f"height_bin labelled {int((df.height_bin != '').sum())}: {Counter(df.height_bin[df.height_bin != ''])}")
    for hs in ["levels", "height_tag"]:
        print(f"  from {hs}: {Counter(df.height_bin[df.height_src == hs])}")
    print(f"  height tag present but not used: {int((df.osm_height_m.notna() & (df.height_bin == '')).sum())}")
    print(f"use_major labelled {int((df.use_major != '').sum())}: {Counter(df.use_major[df.use_major != ''])}")
    print(f"use_fine labelled {int((df.use_fine != '').sum())}: "
          f"{sorted(Counter(df.use_fine[df.use_fine != '']).items())}")
    print(f"use_src: {Counter(df.use_src[df.use_src != ''])}")
    print(f"major without fine, by osm_building: "
          f"{Counter(df.osm_building[(df.use_major != '') & (df.use_fine == '')]).most_common(10)}")


if __name__ == "__main__":
    main()
