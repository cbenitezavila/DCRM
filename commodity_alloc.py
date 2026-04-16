import os
import geopandas as gpd
import pandas as pd
import numpy as np

'''
Distributes territorial overlap area equally across commodities per mine.

Input:
  - over_buffer_*.gpkg   : mine–indig overlap polygons (mine_id, indig_id, overlay_area, geometry, buffer)
  - mine_polygons.gpkg   : mine attributes incl. materials_list (comma-separated string, may contain NaN / "Not relevant")

Output columns:
  ov_id        – unique overlap ID (one per mine–indig pair)
  mine_id
  indig_id
  commodity_id – one row per valid commodity
  alloc_area   – overlay_area / N  (equal share, km²)
  geometry     – inherited from the mine–indig overlap (same for all commodities of the same overlap)
  buffer
'''

p_mine   = 'data/dcrm_cluster_data/dcrm_cluster_data/mine_polygons.gpkg'
p_ov     = 'data/interm/over_calc/over_buffer_0.gpkg'
out_path = 'data/interm/over_calc/over_com_buffer_0.gpkg'
crs_epsg = 6933

DROP        = {""}              # stripped parts that are removed entirely
NON_COMMODITY = {"not relevant"} # values replaced with NaN but kept in computation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_materials(s):
    """
    Parse a materials_list string into a list where:
      - empty parts ("")          → dropped entirely
      - "Not relevant" / NaN field → replaced with "Unknown" (kept, counted in N)
      - valid commodity           → kept as-is
    Returns ["Unknown"] when the whole field is NaN (mine with unknown commodity).
    """
    if pd.isna(s):
        return ["Unknown"]
    result = []
    for c in [p.strip() for p in str(s).split(",")]:
        if c in DROP:
            continue
        elif c.lower() in NON_COMMODITY:
            result.append("Unknown")
        else:
            result.append(c)
    return result  # may be [] only if every part was an empty string


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def build_commodity_overlap(ov_gdf, mine_gdf):
    """
    Expand each mine–indig overlap into one row per valid commodity.
    Area is split equally: alloc_area = overlay_area / N.
    Overlaps for mines with no valid commodities are dropped.
    """
    # --- 1. assign ov_id (stable integer index on the raw overlap rows) ---
    ov = ov_gdf.copy()
    ov["ov_id"] = np.arange(len(ov))

    # --- 2. parse commodities per mine ---
    com = mine_gdf[["id", "materials_list"]].copy()
    com["commodities"]    = com["materials_list"].apply(parse_materials)
    com["n_commodities"]  = com["commodities"].str.len()
    com = com.rename(columns={"id": "mine_id"})

    # --- 3. merge overlap ← mine commodity info ---
    # keep geometry as plain column so pandas explode works correctly
    merged = pd.DataFrame(ov).merge(
        com[["mine_id", "commodities", "n_commodities"]],
        on="mine_id",
        how="left",
    )

    # drop overlaps where materials_list was entirely empty strings (no parts at all)
    n_dropped = (merged["n_commodities"].fillna(0) == 0).sum()
    if n_dropped:
        print(f"  dropped {n_dropped:,} overlap rows where all material parts were empty strings")
    merged = merged[merged["n_commodities"] > 0].copy()

    # --- 4. explode: one row per (overlap × commodity) ---
    merged = merged.explode("commodities").reset_index(drop=True)
    merged = merged.rename(columns={"commodities": "commodity_id"})

    # --- 5. equal area allocation ---
    merged["alloc_area"] = merged["overlay_area"] / merged["n_commodities"]

    # --- 6. return as GeoDataFrame ---
    cols = ["ov_id", "mine_id", "indig_id", "commodity_id", "alloc_area", "geometry", "buffer"]
    return gpd.GeoDataFrame(merged[cols], geometry="geometry", crs=crs_epsg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    mine_gdf = gpd.read_file(p_mine)
    ov_gdf   = gpd.read_file(p_ov)
   

    print(f"Overlaps loaded : {len(ov_gdf):,}")
    print(f"Mines loaded    : {len(mine_gdf):,}")

    final_gdf = build_commodity_overlap(ov_gdf, mine_gdf)

    print(f"Output rows     : {len(final_gdf):,}")
    print(f"Unique commodities: {final_gdf['commodity_id'].nunique()}")
    print(final_gdf.head())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_gdf.to_file(out_path, driver="GPKG")
    print(f"Saved → {out_path}")
