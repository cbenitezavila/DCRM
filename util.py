import logging
import os
from joblib import Parallel, delayed, Memory
import util
import pandas as pd
import geopandas as gpd

import inspect


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

# @memory.cache
# def load_gpd(filepath, logger):
#     logger.info(f"Start: Loading: {filepath}...")
#     with fiona.open(filepath) as src:
#         crs = src.crs
#         gdf = gpd.GeoDataFrame.from_features(src, crs=crs)
#     logger.info(f"Finished: Loading {filepath}")
#     return gdf


def logger_init():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def save_fig_plotnine(plot, name, w=8, h=6):
    base_folder = 'fig'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name)

    plot.save(file_path, width=w, height=h, dpi=300 )

    return None

def df_to_latex(df, filename):
    root_folder = 'tab'
    # Convert DataFrame to LaTeX table format
    latex_table = df.to_latex()
    # Write LaTeX table to a .tex file
    with open(f'{root_folder}/{filename}.tex', 'w') as f:
        f.write(latex_table)
    return None