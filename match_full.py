#!/usr/bin/env python3
"""
Full-county azimuth ray-casting match between Mapillary panorama points and
OSM building footprints (validated on a
downtown test box in match_prototype.py).

Threshold locked at 50m (user decision, 09-19). Buildings and panorama
points transformed to UTM 17N (EPSG:32617) via a single vectorized pyproj
call each (487,084 buildings / ~3.9M vertices, 103,456 cameras) rather than
per-point transforms, since that was the slow part at this scale.

Output:
  matches_50m.jsonl        -- one row per VISIBLE (unoccluded) match:
                               image_id, building_id, distance_m, bearing_deg,
                               compass_angle (camera's own, for later
                               equirect->perspective cropping)
  matched_image_ids.txt    -- unique image ids that appear in >=1 match
                               (this is the actual download list for the
                               next step)
  match_full_stats.json    -- summary counts
"""
import json
import math
import time

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union, nearest_points
from shapely.strtree import STRtree

BUILDINGS_FILE = "osm_buildings_orange_county_20260918.json"
IMAGES_FILE = "mapillary_metadata/images_20260918_dedup.jsonl"
PANORAMA_TYPES = {"spherical", "equirectangular"}
MATCH_RADIUS_M = 50.0
CAMERA_EXCLUSION_RADIUS_M = 2.0

MATCHES_OUT = "mapillary_metadata/matches_50m.jsonl"
IMAGE_IDS_OUT = "mapillary_metadata/matched_image_ids.txt"
STATS_OUT = "match_full_stats.json"

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)


def load_buildings():
    with open(BUILDINGS_FILE) as f:
        d = json.load(f)
    elements = d["elements"]

    # Pass 1: collect every (lon, lat) vertex we'll need, across ways'
    # `geometry` and relations' outer-member `geometry`, with offsets so we
    # can slice the vectorized-transform output back into per-ring lists.
    lons, lats = [], []
    way_specs = []       # (element_index, start, end)
    rel_member_specs = []  # (element_index, member_list_index, start, end)

    for ei, el in enumerate(elements):
        if el["type"] == "way":
            geom = el.get("geometry")
            if not geom or len(geom) < 4:
                continue
            start = len(lons)
            for pt in geom:
                lons.append(pt["lon"])
                lats.append(pt["lat"])
            way_specs.append((ei, start, len(lons)))
        elif el["type"] == "relation":
            outer_idx = 0
            for m in el.get("members", []):
                if m.get("role") != "outer" or not m.get("geometry"):
                    continue
                mgeom = m["geometry"]
                if len(mgeom) < 3:
                    continue
                start = len(lons)
                for pt in mgeom:
                    lons.append(pt["lon"])
                    lats.append(pt["lat"])
                rel_member_specs.append((ei, outer_idx, start, len(lons)))
                outer_idx += 1

    lons = np.array(lons)
    lats = np.array(lats)
    xs, ys = to_utm.transform(lons, lats)

    ways_by_element = {}
    for ei, start, end in way_specs:
        ways_by_element[ei] = list(zip(xs[start:end], ys[start:end]))

    rel_members_by_element = {}
    for ei, _, start, end in rel_member_specs:
        rel_members_by_element.setdefault(ei, []).append(list(zip(xs[start:end], ys[start:end])))

    def make_polygon(coords):
        if coords[0] != coords[-1]:
            coords = coords + [coords[0]]
        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if poly.area > 0 else None
        except Exception:
            return None

    buildings = []
    for ei, el in enumerate(elements):
        if el["type"] == "way":
            coords = ways_by_element.get(ei)
            if not coords:
                continue
            poly = make_polygon(coords)
        else:
            members = rel_members_by_element.get(ei)
            if not members:
                continue
            polys = [p for p in (make_polygon(c) for c in members) if p is not None]
            if not polys:
                continue
            poly = unary_union(polys)
        if poly is None:
            continue
        tags = el.get("tags", {}) or {}
        has_levels = "building:levels" in tags
        has_commercial = any(k in tags for k in ("shop", "amenity", "office", "craft", "tourism"))
        buildings.append({"id": el["id"], "poly": poly, "labeled": has_levels or has_commercial})
    return buildings


def load_panoramas():
    ids, lons, lats, compass = [], [], [], []
    with open(IMAGES_FILE) as f:
        for line in f:
            d = json.loads(line)
            if d.get("camera_type") not in PANORAMA_TYPES:
                continue
            geom = d.get("computed_geometry")
            if not geom:
                continue
            lon, lat = geom["coordinates"]
            ids.append(d["id"])
            lons.append(lon)
            lats.append(lat)
            compass.append(d.get("compass_angle"))
    xs, ys = to_utm.transform(np.array(lons), np.array(lats))
    return [{"id": ids[i], "x": xs[i], "y": ys[i], "compass_angle": compass[i]} for i in range(len(ids))]


def bearing_deg(x0, y0, x1, y1):
    return (math.degrees(math.atan2(x1 - x0, y1 - y0)) + 360) % 360


def main():
    t0 = time.time()
    buildings = load_buildings()
    print(f"[{time.time()-t0:.0f}s] buildings with usable polygons: {len(buildings)}", flush=True)
    panoramas = load_panoramas()
    print(f"[{time.time()-t0:.0f}s] panorama cameras: {len(panoramas)}", flush=True)

    polys = [b["poly"] for b in buildings]
    tree = STRtree(polys)
    print(f"[{time.time()-t0:.0f}s] spatial index built", flush=True)

    n_cameras_excluded = 0
    n_candidates_seen = 0
    n_matched = 0
    n_occluded = 0
    matched_building_ids = set()
    matched_image_ids = set()
    matched_labeled_building_ids = set()
    match_count_per_building = {}

    with open(MATCHES_OUT, "w") as fout:
        for ci, cam in enumerate(panoramas):
            if ci % 10000 == 0 and ci > 0:
                print(f"[{time.time()-t0:.0f}s] processed {ci}/{len(panoramas)} cameras, "
                      f"{n_matched} matches so far", flush=True)

            cx, cy = cam["x"], cam["y"]
            cpt = Point(cx, cy)
            cand_idx = tree.query(cpt.buffer(MATCH_RADIUS_M))
            if len(cand_idx) == 0:
                continue

            cand_info = []
            touches_building = False
            for i in cand_idx:
                poly = polys[i]
                _, np_ = nearest_points(cpt, poly)
                dist = cpt.distance(np_)
                if dist <= CAMERA_EXCLUSION_RADIUS_M:
                    touches_building = True
                    break
                if dist > MATCH_RADIUS_M:
                    continue
                brg = bearing_deg(cx, cy, np_.x, np_.y)
                cand_info.append((i, dist, np_, brg))

            if touches_building:
                n_cameras_excluded += 1
                continue

            n_candidates_seen += len(cand_info)

            for i, dist, np_, brg in cand_info:
                ray_len = dist + 2.0
                rx = cx + ray_len * math.sin(math.radians(brg))
                ry = cy + ray_len * math.cos(math.radians(brg))
                ray = LineString([(cx, cy), (rx, ry)])

                best_j, best_d = None, None
                for j, dist_j, np_j, brg_j in cand_info:
                    poly_j = polys[j]
                    if not ray.intersects(poly_j):
                        continue
                    inter = ray.intersection(poly_j)
                    _, pt_on_inter = nearest_points(cpt, inter)
                    d_hit = cpt.distance(pt_on_inter)
                    if best_d is None or d_hit < best_d:
                        best_d, best_j = d_hit, j

                if best_j == i:
                    n_matched += 1
                    b = buildings[i]
                    fout.write(json.dumps({
                        "image_id": cam["id"],
                        "building_id": b["id"],
                        "distance_m": round(dist, 2),
                        "bearing_deg": round(brg, 2),
                        "compass_angle": cam["compass_angle"],
                    }) + "\n")
                    matched_building_ids.add(b["id"])
                    matched_image_ids.add(cam["id"])
                    if b["labeled"]:
                        matched_labeled_building_ids.add(b["id"])
                    match_count_per_building[b["id"]] = match_count_per_building.get(b["id"], 0) + 1
                else:
                    n_occluded += 1

    dist_counts = {}
    for c in match_count_per_building.values():
        dist_counts[c] = dist_counts.get(c, 0) + 1

    stats = {
        "total_buildings": len(buildings),
        "total_panorama_cameras": len(panoramas),
        "cameras_excluded_degenerate": n_cameras_excluded,
        "candidate_pairs_within_50m": n_candidates_seen,
        "visible_matches": n_matched,
        "occluded_pairs": n_occluded,
        "unique_buildings_matched": len(matched_building_ids),
        "unique_images_used": len(matched_image_ids),
        "unique_labeled_buildings_matched": len(matched_labeled_building_ids),
        "matches_per_building_histogram": dist_counts,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)
    with open(IMAGE_IDS_OUT, "w") as f:
        for iid in sorted(matched_image_ids):
            f.write(iid + "\n")

    print(f"\n[{time.time()-t0:.0f}s] DONE")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
