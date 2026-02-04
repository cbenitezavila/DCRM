
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely


import multiprocessing as mp
from joblib import Parallel, delayed

import time
import os
import logging
import warnings


###############################################Purpose###########################################################
# This code calculates the to per buffer for mines
# Plus we need and area correction factor for the overlaps between mines

###############################################Global Parameters################################################

buffer_list = [0]#1000, 5000, 10000, 30000, 50000]
crs_epsg = 6933

# configure logger
logging.basicConfig(
    level=logging.INFO,               # show INFO and above
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),      # print to console
        logging.FileHandler("run.log", mode='w')   # also write to file
    ]
)

log = logging.getLogger(__name__)


warnings.filterwarnings("ignore", message="GeoSeries.notna", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="GeoDataFrame.swapaxes is deprecated",
    category=FutureWarning
)



def process_buffer_fast(mland_join, buffer):
    df = mland_join.copy()

    df["buffer_geom"] = df.mine_geometry.buffer(buffer, resolution=4)

    # intersect only with assigned indig geometry
    inter = df.apply(
        lambda row: row["buffer_geom"].intersection(row["indig_geometry"]),
        axis=1
    )

    df["geometry"] = inter
    df = df[~df.geometry.is_empty & df.geometry.notna()].copy()

    df["overlay_area"] = df.geometry.area * 1e-6
    df["buffer"] = buffer

    return df[["mine_id", "indig_id", "geometry", "overlay_area", "buffer"]]



def _process_chunk(chunk, buffer):
    """Internal worker for multiprocessing."""
    df = chunk.copy()

    # buffer each mine geometry
    df["buffer_geom"] = df.mine_geometry.buffer(buffer, resolution=4)

    # intersect only with its matched indig geom
    inter = df.apply(
        lambda row: row["buffer_geom"].intersection(row["indig_geometry"]),
        axis=1
    )

    df["geometry"] = inter
    df = df[~df.geometry.is_empty & df.geometry.notna()].copy()

    df["overlay_area"] = df.geometry.area * 1e-6
    df["buffer"] = buffer

    return df[["mine_id", "indig_id", "geometry", "overlay_area", "buffer"]]


def process_buffer_fast_parallel(mland_join, buffer, n_cores=mp.cpu_count()):

    # split into n chunks
    chunks = np.array_split(mland_join, n_cores)

    # multiprocessing
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(_process_chunk, [(chunk, buffer) for chunk in chunks])

    # combine results
    result = pd.concat(results, ignore_index=True)

    result["geometry"] = gpd.GeoSeries(result["geometry"])

    result_gdf = gpd.GeoDataFrame(result, geometry="geometry", crs=crs_epsg)

    return result_gdf


def calc_inter(mland, iland):

    # project once
    mland = mland.to_crs(epsg=crs_epsg)[:100]
    iland = iland.to_crs(epsg=crs_epsg)[:1]

    # simplify iland
    iland["geometry"] = iland.geometry.apply(lambda g: shapely.force_2d(g))
    iland["geometry"] = iland.geometry.simplify(tolerance=100, preserve_topology=True)


    mland['geometry'] = mland.geometry.buffer(0)
    iland['geometry'] = iland.geometry.buffer(0)

    assert mland.crs == iland.crs, "CRS mismatch after projection"

   
    mjoin = gpd.sjoin(mland, iland[["Name_", "geometry"]], predicate="intersects" )

    #map indig geom based on Name_
    mjoin = mjoin.rename(columns={"id": "mine_id", "Name_": "indig_id"})
    indig_geom_map = iland.set_index("Name_").geometry.to_dict()
    mjoin["indig_geometry"] = mjoin["indig_id"].map(indig_geom_map)

    mjoin.rename(columns={"geometry": "mine_geometry"}, inplace=True)


    output_path = r'data\interm\over_calc'
    os.makedirs(output_path, exist_ok=True)

    for b in buffer_list:
        start_time = time.time()
        log.info(f"Start processing buffer: {b}")

        over = process_buffer_fast_parallel(mjoin, b)

        name = f"over_buffer_{b}.gpkg"
        over.to_file(os.path.join(output_path, name), driver="GPKG")

        elapsed = (time.time() - start_time)/60
        log.info(f"Buffer: {b} done, mines: {over.shape[0]}, time: {elapsed:.2f} min")
    
    log.info("All buffers processed.")


   


if __name__ == '__main__':
   p_mine = r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg'
   mland = gpd.read_file(p_mine)
   p_indig = r'data\IPL_IndigenousPeoplesLands_2017\01_Data\IPL_IndigenousPeoplesLands_2017\IPL_2017.shp'
   iland = gpd.read_file(p_indig)
   calc_inter(mland, iland)