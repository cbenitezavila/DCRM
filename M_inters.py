import multiprocessing as mp
import geopandas as gpd
import pandas as pd
import numpy as np






def calc_inter(mland, iland):

    # project once
    mland = mland.to_crs(epsg=crs_epsg)
    iland = iland.to_crs(epsg=crs_epsg)

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

        over = process_buffer_fast(mjoin, b)

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