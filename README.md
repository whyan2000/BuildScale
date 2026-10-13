# BuildScale

Reliability-aware VLM pseudo-labelling for city-scale building attributes (use and number of floors).

Only about 10% of buildings in a county are visible in street-level imagery. BuildScale asks a vision-language model (VLM)
about the imaged buildings, estimates how reliable each answer is, and propagates the trusted labels to every other building
with a lightweight model. Released here:

- **Code** for the full pipeline (this repository)
- **Dataset**: use + floors for all 364,437 OSM buildings in Orange County, FL, with a manually labelled benchmark
  (7 areas, 5,987 buildings) and LiDAR heights — see [`data/README.md`](data/README.md)

> Anonymous release for double-blind review. Code was cleaned from research scripts; paths assume the repository root as working directory.

## Pipeline

```
OSM footprints + Mapillary panoramas
  └─ ray-cast matching, heading-correct crops            (≈10% of buildings get a street view)
      └─ Qwen3.6-27B multiple-choice questions            street view + satellite, option probabilities stored
          └─ cheap reliability signals → reliability model R
              (token margin, cross-modal agreement, agreement with an OSM-only prior;
               logistic regression fitted on 599 targeted labels with 1/π weights)
              └─ two-stage LightGBM propagation
                  stage 1: residential vs non-residential, all labels
                  stage 2: non-residential subclasses, only pseudo-labels with R ≥ 0.5
                  └─ inference: VLM answer if imaged, otherwise propagation model
```

Floors use the same structure with a continuation-ratio model and *masked neighbour heights*
(labelled neighbours closer than a distance drawn from the unlabelled-building distribution are hidden during training,
removing the train/test shift in neighbour distance).

## Code map

| Stage | Scripts |
|---|---|
| S0 data | `download_osm_buildings.py`, `add_county_flag.py`, `download_osm_pois.py`, `download_osm_landuse.py`, `download_osm_context_20261005.py`, `harvest_mapillary_metadata.py`, `match_full.py`, `select_all_matches.py`, `download_all_images.py`, `fetch_image_orientation.py`, `crop_orient_v3.py` (+ `equirect_to_perspective.py`), `compute_occlusion_full_20260923.py`, `make_satellite_chips_full_20261001.py`, `extract_geometry_features.py`, `buildscale/nbr_features_20261005.py` |
| LiDAR reference | `select_lidar_areas.py`, `lidar/ept_area_height_v3.py` (+ `lidar/ept_area_height.py`) |
| S1 OSM seed labels | `buildscale/osm_labels_20261001.py` |
| S2 VLM | `vlm_hazus_20261001.py` (use), `vlm_floors_20261002.py` (floors), `vlm_height_m_20261011.py` (metres, appendix); helper `vlm_baseline_infer.py` |
| S3 reliability signals | `buildscale/cheap_signals_20261006.py` |
| S4 reliability model R | `annotation_hard_20261006/sample_hard_20261006.py` (targeted sampling), `buildscale/hard_labels_R_20261006.py` |
| S5 propagation + evaluation (use) | `buildscale/train_route_v3_20261006.py`, `aggregate_route_v3_20261006.py`, `table2_ci_20261007.py`, `plot_budget_pareto_v3_20261008.py`, `train_route_v3nm_20261011.py` (ablation) |
| S5 propagation + evaluation (floors) | `buildscale/train_floors_20261007.py`, `aggregate_floors_20261007.py`, `use_oof_county_20261011.py`, `train_floors_v2_20261011.py`, `height_m_vs_floors_20261011.py` |
| County-wide release | `buildscale/county_use_release_20261011.py`, `buildscale/county_floors_nm_20261011.py` |
| Tables / figures | `buildscale/make_result_20261008.py`, `buildscale/make_result_20261011.py` |

Run every script from the repository root (`PYTHONPATH=. python buildscale/<script>.py`).
VLM weights are expected in `models/Qwen3.6-27B` (`--model_dir`); Mapillary scripts read an API token from `token_maplliary.txt` (not included).

## Evaluation protocol

Leave-one-area-out over 7 areas of 2 × 2 km (downtown, industrial, residential, tourism, campus, 2 random), 5 seeds.
Use: macro-F1 against manual labels (5,479 buildings after excluding non-buildings / bad footprints).
Floors: Spearman against LiDAR class-6 roof height (5,414 buildings).

## Licenses

Code: MIT (`LICENSE`). Data: ODbL 1.0 (`data/LICENSE`), derived from © OpenStreetMap contributors.
