import os
import sys

# PROJ fix
if not os.environ.get("PROJ_DATA") and not os.environ.get("PROJ_LIB"):
    _proj_dir = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "share", "proj")
    if os.path.isdir(_proj_dir):
        os.environ["PROJ_DATA"] = _proj_dir

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import geopandas as gpd

CRS_PROJ = "EPSG:6933"   # equal-area metres for area calculations

P_MINE  = "data/dcrm_cluster_data/dcrm_cluster_data/mine_polygons.gpkg"
P_INDIG = "data/IPL_IndigenousPeoplesLands_2017/01_Data/IPL_IndigenousPeoplesLands_2017/IPL_2017.shp"
P_OV    = "data/interm/over_calc/over_com_buffer_0.gpkg"


def fmt(x, decimals=0):
    return f"{x:,.{decimals}f}"


def mine_specs():
    print("\n=== Mine polygons ===")
    gdf = gpd.read_file(P_MINE)
    gdf_proj = gdf.to_crs(CRS_PROJ)

    n        = len(gdf)
    crs_src  = gdf.crs.to_string()
    area_km2 = gdf_proj.geometry.area * 1e-6

    # commodity coverage
    has_mat  = gdf["materials_list"].notna()
    n_assign = int(has_mat.sum())

    # all commodity strings
    all_coms = (gdf["materials_list"].dropna()
                  .str.split(",")
                  .explode()
                  .str.strip()
                  .str.lower()
                  .replace("not relevant", pd.NA)
                  .dropna()
                  .str.title())
    unique_coms = sorted(all_coms.unique())

    # bounding box
    bbox = gdf.total_bounds   # minx, miny, maxx, maxy (source CRS)

    print(f"  N polygons          : {fmt(n)}")
    print(f"  Source CRS          : {crs_src}")
    print(f"  Bounding box (deg)  : lon [{bbox[0]:.1f}, {bbox[2]:.1f}]  lat [{bbox[1]:.1f}, {bbox[3]:.1f}]")
    print(f"  Total area          : {fmt(area_km2.sum(), 1)} km²")
    print(f"  Area — mean         : {fmt(area_km2.mean(), 3)} km²")
    print(f"  Area — median       : {fmt(area_km2.median(), 3)} km²")
    print(f"  Area — min / max    : {fmt(area_km2.min(), 4)} / {fmt(area_km2.max(), 1)} km²")
    print(f"  With commodity tag  : {fmt(n_assign)} ({n_assign/n*100:.1f} %)")
    print(f"  Unique commodities  : {len(unique_coms)}")
    print(f"  Commodity list      : {', '.join(unique_coms)}")

    return dict(
        n=n, crs_src=crs_src,
        bbox_lon_min=bbox[0], bbox_lon_max=bbox[2],
        bbox_lat_min=bbox[1], bbox_lat_max=bbox[3],
        total_area=area_km2.sum(),
        mean_area=area_km2.mean(),
        median_area=area_km2.median(),
        min_area=area_km2.min(),
        max_area=area_km2.max(),
        n_assign=n_assign,
        pct_assign=n_assign/n*100,
        unique_coms=unique_coms,
        n_unique_coms=len(unique_coms),
    )


def indig_specs():
    print("\n=== Indigenous Peoples' Lands ===")
    gdf = gpd.read_file(P_INDIG)
    gdf_proj = gdf.to_crs(CRS_PROJ)

    n        = len(gdf)
    crs_src  = gdf.crs.to_string()
    area_km2 = gdf_proj.geometry.area * 1e-6
    bbox     = gdf.to_crs("EPSG:4326").total_bounds

    print(f"  N polygons          : {fmt(n)}")
    print(f"  Source CRS          : {crs_src}")
    print(f"  Bounding box (deg)  : lon [{bbox[0]:.1f}, {bbox[2]:.1f}]  lat [{bbox[1]:.1f}, {bbox[3]:.1f}]")
    print(f"  Total area          : {fmt(area_km2.sum(), 1)} km²")
    print(f"  Area — mean         : {fmt(area_km2.mean(), 1)} km²")
    print(f"  Area — median       : {fmt(area_km2.median(), 1)} km²")
    print(f"  Area — min / max    : {fmt(area_km2.min(), 4)} / {fmt(area_km2.max(), 1)} km²")
    print(f"  ID field            : Name_")

    return dict(
        n=n, crs_src=crs_src,
        bbox_lon_min=bbox[0], bbox_lon_max=bbox[2],
        bbox_lat_min=bbox[1], bbox_lat_max=bbox[3],
        total_area=area_km2.sum(),
        mean_area=area_km2.mean(),
        median_area=area_km2.median(),
        min_area=area_km2.min(),
        max_area=area_km2.max(),
    )


def overlap_specs():
    print("\n=== Commodity-attributed overlaps (buffer = 0 m) ===")
    gdf = gpd.read_file(P_OV)

    n_rows    = len(gdf)
    n_ov      = gdf["ov_id"].nunique()
    n_mines   = gdf["mine_id"].nunique()
    n_indig   = gdf["indig_id"].nunique()
    n_coms    = gdf["commodity_id"].nunique()
    total_alloc = gdf.groupby("ov_id")["alloc_area"].sum()   # sum per overlap = overlay_area
    total_km2 = total_alloc.sum()

    print(f"  Rows (overlap×commodity): {fmt(n_rows)}")
    print(f"  Unique overlaps (ov_id) : {fmt(n_ov)}")
    print(f"  Mines with overlap      : {fmt(n_mines)}")
    print(f"  Indig territories hit   : {fmt(n_indig)}")
    print(f"  Commodities present     : {n_coms}")
    print(f"  Total overlap area      : {fmt(total_km2, 1)} km²")

    return dict(
        n_rows=n_rows, n_ov=n_ov,
        n_mines=n_mines, n_indig=n_indig,
        n_coms=n_coms, total_km2=total_km2,
    )


if __name__ == "__main__":
    m = mine_specs()
    i = indig_specs()
    o = overlap_specs()
    print("\nDone.")
