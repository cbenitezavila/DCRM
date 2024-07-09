import util
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union, polygonize
from tqdm import tqdm
import matplotlib.pyplot as plt

from shapely.geometry import box

logger = util.logger_init()

def grid_shape_constructor(resolution, crs_out, min_lon=-180, min_lat=-90, max_lon=180, max_lat=90, crs_init='EPSG:4326')->gpd.GeoDataFrame:
    '''
    Construct a grid with the given resolution and bounds.


    resolution: int, the resolution of the grid in degrees
    min_lon: int, the minimum longitude of the grid
    min_lat: int, the minimum latitude of the grid
    max_lon: int, the maximum longitude of the grid
    max_lat: int, the maximum latitude of the grid
    crs: str, the coordinate reference system of the grid
    '''
    
    # Generate grid cells using list comprehensions
    grid_cells = [box(lon, lat, lon + resolution, lat + resolution)
                for lon in range(min_lon, max_lon, resolution)
                for lat in range(min_lat, max_lat, resolution)]

    # Create a GeoDataFrame from the grid cells

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=crs_init)
    grid.to_crs(crs_out, inplace=True)
    grid['id'] = np.arange(0,grid.shape[0])

    return grid
  

def area_and_centroid_coords(geometry)->str:
    """
    Returns a string of polygon area and centroid coordinates

    geometry: shapely.geometry.Polygon, the geometry to process
    
    """

    area = round(geometry.area, 1)
    centroid = geometry.centroid
    x = round(centroid.x, 1)
    y = round(centroid.y, 1)
    value = f"{area} {x} {y}"
    return value

def count_overlapping_features(gdf) -> dict:
    '''
    Generates two outputs: 
    1. A GeoDataFrame of polygonized boundaries of the unary union.
    2. A GeoDataFrame of overlapping polygons with the count of participants.
    
    gdf: gpd.GeoDataFrame, the GeoDataFrame to process
    '''
    logger.info("Starting to count overlapping features")

    # Simplify geometries to reduce complexity if necessary
    logger.info("Simplifying geometries...")
    gdf['simplified'] = gdf.geometry.simplify(100, preserve_topology=True)
    
    # Generate a single multiline of all polygon boundaries
    logger.info("Generating unary union of polygons...")
    union = unary_union(gdf.simplified.boundary)

    # Polygonize the multiline to create new polygons
    logger.info("Polygonizing the multiline to create new polygons...")
    polygonized = list(polygonize(union))
    df2 = gpd.GeoDataFrame(geometry=polygonized, crs=gdf.crs)
    logger.info("Generated %d new polygon pieces", len(df2))

    # Use spatial index to speed up intersection
    logger.info("Creating spatial index for intersection...")
    gdf_sindex = gdf.sindex
    intersected = []

    logger.info("Intersecting polygons...")
    for poly in tqdm(df2.geometry):
        possible_matches_index = list(gdf_sindex.intersection(poly.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(poly)]
        for match in precise_matches.geometry:
            intersection = poly.intersection(match)
            if not intersection.is_empty:
                intersected.append(intersection)
    
    intersected_gdf = gpd.GeoDataFrame(geometry=intersected, crs=gdf.crs)
    logger.info("Created %d intersected polygon pieces", len(intersected_gdf))

    # Create a unique string to group by based on area and centroid coordinates
    logger.info("Creating unique strings for grouping...")
    intersected_gdf["geometry_distinction"] = intersected_gdf.apply(
        lambda x: area_and_centroid_coords(x.geometry), axis=1)
    
    # Count the number of overlaps
    logger.info("Counting the number of overlaps...")
    intersected_gdf["n_overlaps"] = intersected_gdf.groupby("geometry_distinction")["geometry_distinction"].transform("count")
    
    logger.info("Finished counting overlapping features")

    return {'unary_union_area': df2, 'area_feature_count': intersected_gdf}


def buffer_iteration(gdf)->dict:
    '''
    Iterates over the buffer and initializes the overlapping features count.

    gdf: gpd.GeoDataFrame, the GeoDataFrame to process

    return: dict, the results of the buffer iteration with buffer as key
    
    '''
    res = {}

    gdf_group = gdf.groupby('buffer')

    for buffer, group in tqdm(gdf_group, desc=f"Buffer Iteration"):
        res[buffer] = count_overlapping_features(group)
        break

    return res
    
def overlay_and_area_calc(grid, df):
    '''
    Overlays the grid with the data and calculates the area of the resulting polygons. Per initial grid cell

    grid: gpd.GeoDataFrame, the grid to overlay
    df: gpd.GeoDataFrame, the data to overlay

    return: gpd.GeoDataFrame, the grid with the area of the resulting polygons
    '''
    logger.info("Starting overlay and area calculation")

    # Overlay the grid with the data
    logger.info("Overlaying the grid with the data...")

    assert grid.crs == df.crs, "CRS of the grid and data do not match"
    overlay = gpd.overlay(grid, df, how='intersection')

    # Calculate the area of the resulting polygons
    logger.info("Calculating the area of the resulting polygons...")
    overlay['area'] = overlay.geometry.area / 10**6  # Convert to km²

    overlay_g = overlay.groupby('id')['area'].sum()

    grid['area'] = grid['id'].map(overlay_g)

    grid['area'] = grid['area'].fillna(0) # Fill NaN values with 0 = no overlap in grid

    logger.info("Finished overlay and area calculation")

    return grid


def allocate_poly_to_grid(gdf, grid_resolution, plotting = True)->dict:
    '''
    Allocates the polygons to the grid and calculates the area of the resulting polygons.
    
    
    gdf: gpd.GeoDataFrame, the GeoDataFrame to process
    grid_resolution: int, the resolution of the grid in degrees
    '''

    res = buffer_iteration(gdf)
    grid = grid_shape_constructor(resolution = grid_resolution, crs_out=gdf.crs)

    grid_collect = {}

    for buffer, data in res.items():
        for key, value in data.items():
                grid_collect[(buffer, key)] = overlay_and_area_calc(grid, value)
                
                if plotting: plot_grid(grid_collect[(buffer, key)], buffer, key)

        break
            
    return grid_collect


def plot_grid(grid, buffer, key)->None:

    '''
    Plots the grid with the area of the resulting polygons.
        
    grid: gpd.GeoDataFrame, the grid to plot
    buffer: int, the buffer value
    key: int, the key value

    return: None
    '''

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.preprocessing import StandardScaler
    import seaborn as sns
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    logger.info("Plotting the grid with the area of the resulting polygons...")

    #cpal = sns.color_palette('rocket_r', as_cmap=True)

    # Create a figure and axis with a specified projection
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.Mollweide()})
    
    # Plot the GeoDataFrame
    grid[grid.area != 0.0 ].plot(column='area', scheme='fisher_jenks', k=4, legend=True, cmap='magma', ax=ax, transform=ccrs.Mollweide(), legend_kwds={'fontsize': 8})

    # Add coastlines and land features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.3, edgecolor='black')

    # Customize gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize':8 }  # Rotate latitude labels

    gl.ylabel_style = {'fontsize':8 } 

    legend = ax.get_legend()
    legend.set_bbox_to_anchor((1, 0.09))  # Move legend outside the plot
    legend.set_title(f'Overlay Area (km²), buffer={buffer}, key = {key} km', prop={'size': 8})

    # Set extent to the entire globe
    ax.set_global()

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(f"fig/Area_analysis_buffer={buffer}km,key={key}.png", bbox_inches='tight')  # Use bbox_inches='tight' to ensure the legend is not cut off
    
    logger.info("Finished plotting the grid with the area of the resulting polygons")



def main():
    data_path = 'data/interm/mine_indig_footprint_corrected.gpkg'
    resolution_of_grid = 1 # in degrees

    overlaps = util.load_gpd(data_path, logger)
    
    res = allocate_poly_to_grid(overlaps, grid_resolution=resolution_of_grid, plotting = True)




if __name__ == '__main__':
    
    main()  

