import logging
import os
from joblib import Parallel, delayed, Memory
import util
import pandas as pd
import geopandas as gpd


# Set up joblib memory caching
cache_dir = 'cache_dir'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

memory = Memory(location=cache_dir, verbose=0)

@memory.cache
def load_csv(filepath, logger):
    logger.info(f"Start: Loading: {filepath}...")
    data = pd.read_csv(filepath)
    logger.info(f"Finished: Loading {filepath}")
    return data

@memory.cache
def load_gpd(filepath, logger):
    logger.info(f"Start: Loading: {filepath}...")
    with fiona.open(filepath) as src:
        crs = src.crs
        gdf = gpd.GeoDataFrame.from_features(src, crs=crs)
    logger.info(f"Finished: Loading {filepath}")
    return gdf


def logger_init():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger