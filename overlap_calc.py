
# --- PROJ fix: set PROJ_DATA before any spatial import when conda env is not activated ---
import os, sys
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely                         # shapely 2.x vectorised API

from joblib import Parallel, delayed

import time
import os
import logging
import warnings


###############################################Purpose###########################################################
# This code calculates the territorial overlap (TO) per buffer for mines.
# Mine polys are numerous; indig polys are few.

###############################################Global Parameters################################################

buffer_list = [0]  # 1000, 5000, 10000, 30000, 50000]
crs_epsg   = 6933
N_JOBS     = 8      # machine has 8 cores

# configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run.log", mode='w'),
    ]
)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="GeoSeries.notna", category=UserWarning)
warnings.filterwarnings("ignore", message="GeoDataFrame.swapaxes is deprecated", category=FutureWarning)


# ---------------------------------------------------------------------------
# Worker — receives plain numpy arrays, no GeoDataFrame pickling overhead
# ---------------------------------------------------------------------------

def _process_chunk_vec(mine_geoms, indig_geoms, mine_ids, indig_ids, buffer):
    """
    Vectorised chunk worker (shapely 2.x array ops — no Python-level loops).

    Parameters
    ----------
    mine_geoms, indig_geoms : np.ndarray of shapely geometries
    mine_ids, indig_ids     : np.ndarray of ids
    buffer                  : numeric buffer distance (metres, projected CRS)

    Returns
    -------
    dict of arrays ready for pd.DataFrame construction
    """
    # buffer all mine geometries at once — single C-level loop
    buffered = shapely.buffer(mine_geoms, buffer, quad_segs=4)

    # intersect vectorised — single C-level loop
    inter = shapely.intersection(buffered, indig_geoms)

    # shapely.is_empty handles None/null geometries as True
    mask = ~shapely.is_empty(inter)

    return {
        "mine_id":      mine_ids[mask],
        "indig_id":     indig_ids[mask],
        "geometry":     inter[mask],
        "overlay_area": shapely.area(inter[mask]) * 1e-6,  # m² → km²
    }


# ---------------------------------------------------------------------------
# Per-buffer orchestrator
# ---------------------------------------------------------------------------

def process_buffer(mland_join, buffer, n_jobs=N_JOBS):
    """
    Buffer all mine geometries by `buffer` metres and compute intersection
    areas with matched indig polygons, using all available cores.
    """
    mine_geoms  = mland_join["mine_geometry"].values
    indig_geoms = mland_join["indig_geometry"].values
    mine_ids    = mland_join["mine_id"].values
    indig_ids   = mland_join["indig_id"].values

    n = len(mland_join)
    chunk_indices = np.array_split(np.arange(n), n_jobs)

    # threading backend: shapely 2.x releases the GIL for all vectorised ops,
    # so threads give true parallelism with zero memory duplication
    # (loky/multiprocessing pickles geometry arrays into each worker → OOM)
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_process_chunk_vec)(
            mine_geoms[idx],
            indig_geoms[idx],
            mine_ids[idx],
            indig_ids[idx],
            buffer,
        )
        for idx in chunk_indices
        if len(idx) > 0
    )

    # concatenate dict-of-arrays results
    result_df = pd.DataFrame({
        "mine_id":      np.concatenate([r["mine_id"]      for r in results]),
        "indig_id":     np.concatenate([r["indig_id"]     for r in results]),
        "geometry":     np.concatenate([r["geometry"]     for r in results]),
        "overlay_area": np.concatenate([r["overlay_area"] for r in results]),
        "buffer":       buffer,
    })

    return gpd.GeoDataFrame(result_df, geometry="geometry", crs=crs_epsg)


# ---------------------------------------------------------------------------
# Main entry — spatial join + buffer loop
# ---------------------------------------------------------------------------

def calc_inter(mland, iland):
    # project once
    mland = mland.to_crs(epsg=crs_epsg)
    iland = iland.to_crs(epsg=crs_epsg)
    assert mland.crs == iland.crs, "CRS mismatch after projection"

    # Reversed sjoin: build STRtree on mines (many), query with indig polys (few).
    # This runs N_indig queries on a large tree instead of N_mines queries on a tiny tree
    # → orders of magnitude faster when indig polys are few.
    mjoin = gpd.sjoin(
        iland[["Name_", "geometry"]],
        mland,
        how="inner",
        predicate="intersects",
    )
    mjoin = mjoin.rename(columns={"Name_": "indig_id", "id": "mine_id",
                                  "geometry": "indig_geometry"})

  
    mine_geom_map = mland.set_index("id").geometry.to_dict()
    mjoin["mine_geometry"] = mjoin["mine_id"].map(mine_geom_map)

    log.info(f"Spatial join complete: {len(mjoin):,} mine–indig pairs, {N_JOBS} workers")

    output_path = 'data/interm/over_calc'
    os.makedirs(output_path, exist_ok=True)

    for b in buffer_list:
        t0 = time.time()
        log.info(f"Buffer {b} m — start")

        over = process_buffer(mjoin, b)

        out_file = os.path.join(output_path, f"over_buffer_{b}.gpkg")
        over.to_file(out_file, driver="GPKG")

        elapsed = (time.time() - t0) / 60
        log.info(f"Buffer {b} m — done  rows={over.shape[0]:,}  time={elapsed:.2f} min")

    log.info("All buffers processed.")


if __name__ == '__main__':
    p_mine  = 'data/dcrm_cluster_data/dcrm_cluster_data/mine_polygons.gpkg'
    p_indig = 'data/IPL_IndigenousPeoplesLands_2017/01_Data/IPL_IndigenousPeoplesLands_2017/IPL_2017.shp'

    mland = gpd.read_file(p_mine)
    iland = gpd.read_file(p_indig)

    calc_inter(mland, iland)
