import geopandas as gpd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd

from joblib import Parallel, delayed

import time



def main_indig():
    p = r'data\IPL_IndigenousPeoplesLands_2017\01_Data\IPL_IndigenousPeoplesLands_2017\IPL_2017.shp'
    indig = gpd.read_file(p)

    # to crs 4326
    indig = indig.to_crs(epsg=3857)

    # to a metric crs to calculate area

    pass

def main_mine():
    p = r'data\dcrm_cluster_data\dcrm_cluster_data\mine_polygons.gpkg'
    mines = gpd.read_file(p)

    pass

if __name__ == '__main__':
    main_indig()