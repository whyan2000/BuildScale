# BuildScale Orange County dataset

Building **use** and **number of floors** for all 364,437 OpenStreetMap buildings in Orange County, Florida, USA,
plus the evaluation benchmark and intermediate VLM outputs used in the paper.

License: **ODbL 1.0** (see `LICENSE` in this folder). Contains information from © OpenStreetMap contributors.

## Files

| File | Rows | Description |
|---|---|---|
| `orange_county_buildings_use_floors.csv.gz` | 364,437 | Main dataset, one row per building |
| `benchmark_human_labels_7areas.csv` | 5,987 | Manual Hazus labels for every building in the 7 evaluation areas |
| `benchmark_lidar_heights_7areas.csv` | 5,987 | LiDAR (USGS 3DEP, class 6) roof height per building in the 7 areas |
| `benchmark_areas.csv` | 7 | Extent of each evaluation area |
| `targeted_labels_for_R.csv` | 599 | Targeted manual labels used to fit the reliability model R (outside the evaluation areas) |
| `vlm_use_county_street_satellite.jsonl.gz` | 70,713 | VLM use answers + option probabilities, street view and satellite, every imaged building |
| `vlm_use_eval_areas_all_buildings.jsonl.gz` | 9,258 | Same, for all buildings in the 8 surveyed areas (Full-VLM baseline) |
| `vlm_floors_county_street.jsonl.gz` | 35,355 | VLM floor-count answers + probabilities, street view |
| `vlm_floors_eval_areas_street.jsonl.gz` | 6,997 | Same, evaluation areas (street 1,010 + satellite 5,987) |
| `streetview_matches.csv.gz` | 36,488 | Building → Mapillary image used, camera distance, bearing, heading |

`building_id` is the OSM element id; `osm_type` (way / relation) disambiguates it.

### Main dataset columns

| Column | Meaning |
|---|---|
| `lon`, `lat` | Footprint centroid (WGS84) |
| `use_major` | Final use: RES, COM, IND, AGR, REL, GOV, EDU, ACC (Hazus major classes; ACC = accessory building) |
| `use_src` | `osm` (kept from OSM), `street_vlm` (street-view VLM answer), `gbm` (propagation model) |
| `floors` | Final number of floors (expected value, continuous) |
| `floors_round` | `floors` rounded |
| `floors_src` | `osm`, `street_vlm`, `gbm` |
| `osm_use_major`, `osm_levels` | Original OSM attributes (empty if absent) |
| `use_model` | Model prediction ignoring OSM (VLM if imaged and answered, else two-stage GBM) |
| `floors_model_E` | Model floor estimate ignoring OSM |
| `R` | Reliability score of the street-view pseudo-label (P(correct)); empty if no street view |
| `P_RES` … `P_ACC` | Class probabilities behind `use_model` |

Priority rule: OSM attribute if present, else street-view VLM, else propagation model (the dataset only fills gaps; it never overrides OSM).

### Benchmark labels

- `hazus_code`: Hazus occupancy class (e.g. RES1, COM2) or major class when the subclass is unclear.
  Special codes: `NB` (not a building), `BADFP` (bad footprint), `UNK` (cannot tell) — excluded from evaluation.
- `original_code = IND`: 141 buildings first labelled IND and relabelled COM (warehouse/industrial-park definition aligned to Hazus COM2).
- LiDAR: `height_p95_m` = 95th percentile of class-6 returns above ground inside the footprint; `edge_clipped` = footprint crosses the area border.
  Class 6 misses some high-rise roofs, so tall buildings are under-estimated; use rank metrics (Spearman).

### VLM outputs

Model: Qwen3.6-27B, bf16, greedy, thinking off. Each question is single-letter multiple choice; `p_use`, `p_height`, `p_sub`
are the renormalised probabilities over the options at the answer position; `pmass_*` is the total probability on the options.
`unseen` / `X` = the model says the target is not visible / cannot tell. `raw1`, `raw2` = raw generations.

### Street-view imagery

Images are not redistributed. `streetview_matches.csv.gz` lists the Mapillary `image_id` per building; crops can be
regenerated with `download_all_images.py` and `crop_orient_v3.py` (a Mapillary API token is required).
Mapillary imagery is licensed CC BY-SA 4.0 by its contributors.

## Sources and attribution

- Building footprints and tags: © OpenStreetMap contributors, ODbL 1.0
- Street-level imagery: Mapillary, CC BY-SA 4.0
- Aerial imagery: USDA NAIP 2023 (public domain)
- LiDAR: USGS 3DEP, FL_Peninsular_FDEM_Orange_2018 (public domain)
