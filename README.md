# DCRM
Project: Decolonialisation of Critical Raw Materials

Quantifies the spatial overlap between global mine polygons and indigenous peoples' lands, and attributes that overlap to individual commodities.

---

## Conventions

Variable naming: no units in name, no spaces, first letter capitalised (e.g. `Production_mass`).

---

## Repository structure

```
DCRM/
├── scripts/
│   ├── util.py                  # shared helpers (logging, joblib caching)
│   ├── overlap_calc.py          # model: mine–indig geometric intersection
│   ├── commodity_alloc.py       # model: per-commodity area allocation
│   ├── conflict_to_overlap.py   # model: conflict–mine proximity (Hausdorff)
│   ├── dataset_specs.py         # exploration: dataset summary statistics
│   ├── com_compare.py           # exploration: commodity cross-dataset comparison
│   ├── spatial_plot.py          # analysis: spatial & distributional figures → fig/spatial/
│   ├── share_calc_plot.py       # analysis: overlap share figures → fig/share/
│   ├── overlap_dist.py          # analysis: overlap size distribution figures
│   └── overlap_map.py           # analysis: interactive Folium map → fig/overlap_map.html
├── data/                        # input datasets and intermediate outputs
│   └── interm/over_calc/        # intermediate: over_buffer_*.gpkg, over_com_buffer_*.gpkg
├── fig/                         # all output figures
│   ├── spatial/
│   └── share/
├── legacy/                      # earlier scripts, kept for reference
└── orga/                        # project organisation docs, literature, presentations
```

All scripts are run from the project root, e.g. `python scripts/overlap_calc.py`.

---

## Data

| Dataset | Path | Description |
|---|---|---|
| Mine polygons | `data/dcrm_cluster_data/.../mine_polygons.gpkg` | Global mine footprint polygons with `id` and `materials_list` (comma-separated commodity string; NaN or "Not relevant" treated as Unknown) |
| Indigenous Peoples' Lands | `data/IPL_IndigenousPeoplesLands_2017/.../IPL_2017.shp` | IUCN/WWF IPL 2017 dataset of indigenous territory polygons (`Name_` as ID) |
| World boundaries | `data/world_bound/world-administrative-boundaries.shp` | Country polygons used for spatial attribution and choropleth maps |

---

## Model

### `overlap_calc.py`

Computes the geometric intersection (territorial overlap) between every mine polygon and every indigenous land polygon it touches.

**Key steps:**
1. Both datasets are projected to EPSG:6933 (equal-area metres).
2. A reversed spatial join builds the candidate mine–indig pairs: the STR-tree index is built on mines (many), queried with indig polygons (few) — this is much faster than the naive direction.
3. Each mine polygon is optionally buffered by a configurable distance (currently `buffer = 0` m; the list can include 1 000–50 000 m for sensitivity runs).
4. Intersections are computed via Shapely 2.x vectorised array operations; parallelised across 8 threads using `joblib` (threading backend exploits Shapely's GIL release, avoiding the memory cost of multiprocessing).
5. Output: one GeoDataFrame row per mine–indig overlap with columns `mine_id`, `indig_id`, `overlay_area` (km²), `geometry`, `buffer`. Saved as `data/interm/over_calc/over_buffer_{b}.gpkg`.

### `commodity_alloc.py`

Expands each mine–indig overlap into one row per commodity and allocates area equally across commodities.

**Key steps & assumptions:**
1. `materials_list` is parsed per mine: empty strings are dropped; "Not relevant" / NaN → `"Unknown"`. The number of valid entries `N` is counted per mine.
2. Each overlap row is exploded into `N` rows, one per commodity.
3. **Equal-area allocation assumption:** `alloc_area = overlay_area / N`. Each commodity is assumed to occupy an equal share of the mine footprint. Overlaps for mines with no parseable commodities are dropped.
4. A stable `ov_id` integer is assigned to each raw overlap before exploding, so downstream code can reconstruct the original overlap polygon by grouping on `ov_id`.
5. Output: `data/interm/over_calc/over_com_buffer_{b}.gpkg` with columns `ov_id`, `mine_id`, `indig_id`, `commodity_id`, `alloc_area`, `geometry`, `buffer`.

### `conflict_to_overlap.py`

Maps conflict events (EJAtlas) to mining sites and computes the Hausdorff distance between conflict points and mine polygons.

---

## Analysis

### `spatial_plot.py`

Produces spatial and distributional overview figures saved to `fig/spatial/`.

- **A** — Global KDE heatmap of overlap centroids.
- **B** — Country choropleth of total overlap area (km²), quantile classification, flare palette.
- **C / D** — 50 × 50 km dominant-commodity grid maps (C: unique overlaps; D: all commodity rows).
- **E** — Bar chart of total allocated overlap area by commodity.
- **F** — Top 25 indigenous territories by overlap area.
- **G** — Boxplot of overlap area distribution per commodity (log scale).
- **H** — Stacked bar of overlap area by continent and commodity.
- **I** — Side-by-side maps showing mine commodity assignment status (assigned vs. unassigned) for all mines and overlapping mines.
- **J** — World map of indigenous territories coloured by whether they have any mine overlap.
- **K** — 50 × 50 km grid of total overlap area, Robinson projection with three regional subplots.

### `share_calc_plot.py`

Computes overlap *share* (overlap area / mine area) per country and commodity, saved to `fig/share/`. Country attribution uses polygon clipping (not centroid assignment) so mines straddling borders are split correctly.

Each bar chart shows two reference lines: **per-entity average** (mean of individual country/region/commodity shares, equal weight) and **global share** (∑ overlap / ∑ mine area, area-weighted).

- **A** — Top 10 countries by absolute overlap, share displayed.
- **B** — Top 10 countries by relative share.
- **C** — All world regions by relative share.
- **D** — All commodities by relative share (commodity colours from shared palette).
- **E** — World choropleth of country-level share, Robinson projection, colorscale capped at 95th percentile.
- **F** — Commodity exposure quadrant: total overlap area vs. share.
- **K** — Top-10 countries: stacked overlap area by commodity with share overlay line.

### `overlap_dist.py`

Distribution plots of overlap sizes, saved to `fig/share/`.

### `overlap_map.py`

Generates an interactive Folium map of mine–indig overlaps, saved to `fig/overlap_map.html`.

---

## Exploration

### `dataset_specs.py`

Prints summary statistics for the mine polygon and indigenous lands datasets, and the computed overlap layer (counts, area distributions, commodity coverage).

### `com_compare.py`

Cross-checks commodity labels between the SNL weights table and the mine polygon dataset.

