# DCRM
Project: Decolonialisation of Critical Raw Materials

Quantifies the spatial overlap between global mine polygons and indigenous peoples' lands, and attributes that overlap to individual commodities.

---

## Conventions

Script level prefix:
- `D` — data gathering (one subfolder per source)
- `E` — data exploration
- `M` — modelling
- `R` — results / analysis

Variable naming: no units in name, no spaces, first letter capitalised (e.g. `Production_mass`).

---

## Data

| Dataset | Path | Description |
|---|---|---|
| Mine polygons | `data/dcrm_cluster_data/.../mine_polygons.gpkg` | Global mine footprint polygons with `id` and `materials_list` (comma-separated commodity string; NaN or "Not relevant" treated as Unknown) |
| Indigenous Peoples' Lands | `data/IPL_IndigenousPeoplesLands_2017/.../IPL_2017.shp` | IUCN/WWF IPL 2017 dataset of indigenous territory polygons (`Name_` as ID) |
| World boundaries | `data/world_bound/world-administrative-boundaries.shp` | Country polygons used for spatial attribution and choropleth maps |

Intermediate outputs are written to `data/interm/over_calc/`.

---

## Model

### `scripts/model/M_to_calc_per_buffer.py`

Computes the geometric intersection (territorial overlap) between every mine polygon and every indigenous land polygon it touches.

**Key steps:**
1. Both datasets are projected to EPSG:6933 (equal-area metres).
2. A reversed spatial join builds the candidate mine–indig pairs: the STR-tree index is built on mines (many), queried with indig polygons (few) — this is much faster than the naive direction.
3. Each mine polygon is optionally buffered by a configurable distance (currently `buffer = 0` m; the list can include 1 000–50 000 m for sensitivity runs).
4. Intersections are computed via Shapely 2.x vectorised array operations; parallelised across 8 threads using `joblib` (threading backend exploits Shapely's GIL release, avoiding the memory cost of multiprocessing).
5. Output: one GeoDataFrame row per mine–indig overlap with columns `mine_id`, `indig_id`, `overlay_area` (km²), `geometry`, `buffer`. Saved as `over_buffer_{b}.gpkg`.

### `scripts/model/M_to_to_commodity.py`

Expands each mine–indig overlap into one row per commodity and allocates area equally across commodities.

**Key steps & assumptions:**
1. `materials_list` is parsed per mine: empty strings are dropped; "Not relevant" / NaN → `"Unknown"`. The number of valid entries `N` is counted per mine.
2. Each overlap row is exploded into `N` rows, one per commodity.
3. **Equal-area allocation assumption:** `alloc_area = overlay_area / N`. Each commodity is assumed to occupy an equal share of the mine footprint. Overlaps for mines with no parseable commodities are dropped.
4. A stable `ov_id` integer is assigned to each raw overlap before exploding, so downstream code can reconstruct the original overlap polygon by grouping on `ov_id`.
5. Output: `over_com_buffer_{b}.gpkg` with columns `ov_id`, `mine_id`, `indig_id`, `commodity_id`, `alloc_area`, `geometry`, `buffer`.

---

## Analysis

Both scripts are run from the project root: `python scripts/analysis/<script>.py`.

### `scripts/analysis/spatial.py`

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

### `scripts/analysis/share.py`

Computes overlap *share* (overlap area / mine area) per country and commodity, saved to `fig/share/`. Country attribution uses polygon clipping (not centroid assignment) so mines straddling borders are split correctly.

Each plot shows two reference lines: **per-entity average** (mean of individual country/region/commodity shares, equal weight) and **global share** (∑ overlap / ∑ mine area, area-weighted).

- **A** — Top 10 countries by absolute overlap, share displayed.
- **B** — Top 10 countries by relative share.
- **C** — All world regions by relative share.
- **D** — All commodities by relative share (commodity colours from shared palette).
- **E** — World choropleth of country-level share, same flare palette as spatial B, Robinson projection, colorscale capped at 95th percentile.

---

## Script naming

| Prefix | Level |
|---|---|
| `D` | Data gathering |
| `E` | Exploration |
| `M` | Modelling |
| `R` | Results / Analysis |

