"""Select one representative street-level match per building, for ALL
buildings with a match (not just the ones with an OSM building:levels
ground-truth tag) -- this is the pseudo-label generation candidate set,
used to train the geometry->attribute propagation model later.

Unlike select_baseline_sample.py (which required a ground-truth label and
was for validation), this has no ground truth -- the VLM's own output IS
the (noisy) label we're generating.
"""

import json

MATCHES_FILE = "mapillary_metadata/matches_50m.jsonl"
OUT_FILE = "all_matches_sample.jsonl"
MIN_DIST = 4.0  # below this, GPS/OSM positional noise dominates (see 09-21 notes)


def main():
    best_per_building = {}
    with open(MATCHES_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["distance_m"] < MIN_DIST:
                continue
            bid = r["building_id"]
            cur = best_per_building.get(bid)
            if cur is None or r["distance_m"] < cur["distance_m"]:
                best_per_building[bid] = r

    print(f"buildings selected: {len(best_per_building)}")
    with open(OUT_FILE, "w") as f:
        for bid, r in best_per_building.items():
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
