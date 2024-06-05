import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import hausdorff_distance
from joblib import Parallel, delayed, Memory
import fiona
import logging
import os

#####
#This script maps conflicts to mining sites and calculates the Hausdorff distance between the two.
#####

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up joblib memory caching
cache_dir = 'cache_dir'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

memory = Memory(location=cache_dir, verbose=0)

@memory.cache
def load_conflict_data(filepath):
    logger.info("Loading conflict data...")
    cf = pd.read_excel(filepath, sheet_name='EJAtlas Data', skiprows=[0])
    return cf

@memory.cache
def load_overlay_data(filepath):
    logger.info("Loading overlay data...")
    with fiona.open(filepath) as src:
        crs = src.crs
        gdf = gpd.GeoDataFrame.from_features(src, crs=crs)
    return gdf

def conflict_processing(cf, columns_of_interest):
    try:
        logger.info("Starting conflict_processing...")
        # Clean and filter the data
        cf['End Date'] = pd.to_datetime(cf['End Date'], errors='coerce', dayfirst=True)
        cf['Start Date'] = pd.to_datetime(cf['Start Date'], errors='coerce', dayfirst=True)
        cf['duration'] = cf['End Date'] - cf['Start Date']

        mask = cf[columns_of_interest].eq(1).any(axis=1)
        cf_filtered = cf.loc[mask]

        cfm = cf_filtered[['Conflict Id', 'Case', 'Country', 'Lat', 'Lon', 'Start Date', 'End Date', 'duration',
                           'MobilizingGroup: Indigenous groups or traditional communities',
                           'MobilizingForm: Lawsuits, court cases, judicial activism']]
        
        # Clean Lat and Lon columns
        cfm['Lon'] = cfm['Lon'].astype(str).str.rstrip('.')
        cfm['Lat'] = cfm['Lat'].astype(str).str.rstrip('.')
        cfm[['Lon', 'Lat']] = cfm[['Lon', 'Lat']].astype(float)

        cg = gpd.GeoDataFrame(cfm, geometry=gpd.points_from_xy(cfm.Lon, cfm.Lat), crs='EPSG:4326')
        cg_end = cg[cg['End Date'].isnull()]
        cg_end = cg_end.drop(columns=['Lat', 'Lon']).set_index('Conflict Id')
        
        logger.info("Completed conflict_processing.")
        return cg_end
    except Exception as e:
        logger.error(f"Error in conflict_processing: {e}")
        return None

def row_hausdorff_distance(row, conflict_pre):
    try:
        polygon = row['geometry']
        point = conflict_pre.loc[row['Conflict_ID']]['geometry']
        return hausdorff_distance(polygon, point)
    except Exception as e:
        logger.error(f"Error in row_hausdorff_distance: {e}")
        return np.nan

def haus_distance(conflict, columns_of_interest, overlay):
    try:
        logger.info("Starting haus_distance calculation...")
        join = join_conflict_overlay(conflict, columns_of_interest, overlay)
        conflict_pre = conflict_processing(conflict, columns_of_interest)
        conflict_pre.to_crs(join.crs, inplace=True)
        assert conflict_pre.crs == join.crs, f"CRS mismatch between conflict and join data."
        results = Parallel(n_jobs=-1)(
            delayed(row_hausdorff_distance)(row, conflict_pre) for _, row in join.iterrows()
        )

        join['hausdorff_distance'] = results

        join['inv_hausdorff_distance'] = 1/join['hausdorff_distance']
        
        logger.info("Completed haus_distance calculation.")
        return join
    except Exception as e:
        logger.error(f"Error in haus_distance: {e}")
        return None

def join_conflict_overlay(conflict, columns_of_interest, overlays):
    try:

        logger.info("Starting join_conflict_overlay...")
        c_pro = conflict_processing(conflict, columns_of_interest)
        c_pro.to_crs(overlays.crs, inplace=True)

        assert c_pro.crs == overlays.crs, f"CRS mismatch between conflict and overlay data."
        cjoin = gpd.sjoin(overlays, c_pro, how='inner', op='intersects')
        cjoin.rename(columns={'index_right': 'Conflict_ID', 'ID': 'Mine_ID'}, inplace=True)
        
        logger.info("Completed join_conflict_overlay.")
        return cjoin
    except Exception as e:
        logger.error(f"Error in join_conflict_overlay: {e}")
        return None

if __name__ == '__main__':
    try:
        logger.info("Starting data loading...")

        conflict_filepath = 'DCRM/data/EJAtlas_dataset_V1_2024-01.xlsx'
        overlay_filepath = 'DCRM/data/interm/mine_indig_footprint_corrected.gpkg'
        
        # Load the conflict data
        cf = load_conflict_data(conflict_filepath)
        logger.info('Conflict data loaded.')

        col_int = ['Type: Uranium extraction', 
                   'Type: Mineral ore exploration',
                   'Type: Mineral processing',
                   'Type: Tailings from mines',
                   'Type: Building materials extraction (quarries, sand, gravel)',
                   'Type: Coal extraction and processing',
                   'Type: Mineral ore exploration']

        # Load the overlay data
        mines = load_overlay_data(overlay_filepath)
        logger.info('Overlay data loaded.')

        # Calculate the Hausdorff distance
        join_haus = haus_distance(cf, col_int, mines)
        
        if join_haus is not None:
            logger.info('Hausdorff distance calculation completed.')
            print(join_haus.head())

            output_filepath = 'DCRM/data/interm/conflict_mine_overlay_dist.gpkg'

            
            logging.info(f"Writing output to {output_filepath}")

            with fiona.open(output_filepath, 'w', driver='GPKG', crs=join_haus.crs) as dst:
                for feature in join_haus.iterfeatures():
                    dst.write(feature)
            logging.info(f"Output written to {output_filepath}")
        else:
            logger.error('Hausdorff distance calculation failed.')

    except Exception as e:
        logger.error(f"Error in main block: {e}")