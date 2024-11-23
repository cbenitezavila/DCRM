import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from statsmodels.genmod.families import Gamma
import geopandas as gpd
from util import save_fig_plotnine

from plotnine import *
import pandas as pd
from tqdm import tqdm 
from joblib import Parallel, delayed


def weighted_sum(alloc, mines):
    sum_per_conflict = alloc.groupby(['Conflict Id', 'buffer'])['inv_centroid_distance'].sum().reset_index()

    # Merge the sum back to the original dataframe
    alloc = alloc.merge(sum_per_conflict[['Conflict Id', 'buffer', 'inv_centroid_distance']], on=['Conflict Id', 'buffer'], suffixes=('', '_sum'))
    alloc['weighted_conflict_sum'] = alloc['inv_centroid_distance'] / alloc['inv_centroid_distance_sum']
    
    mines.rename(columns={'ID': 'Mine_ID'}, inplace=True)
    mines = mines.merge(alloc[['Mine_ID', 'buffer', 'weighted_conflict_sum']], on=['Mine_ID', 'buffer'],how = 'left')
    mines['weighted_conflict_sum'] = mines['weighted_conflict_sum'].fillna(0)

    return mines


def process_combination(df, b, c, epsg):
    """
    Process a single buffer and country combination to calculate corrected overlap areas.
    """
    try:
        df.to_crs(epsg=epsg, inplace=True)
        df['Territorial_overlap'] = df['geometry'].area * 10**-6  # Convert to km2

        # Subset data for the current buffer and country
        df_b = df[(df['buffer'] == b) & (df['admin'] == c)]

        if df_b.empty or len(df_b) < 2:
            return None  # Skip if there are no overlaps or insufficient data

        # Compute pairwise intersections
        overlaps = gpd.overlay(df_b, df_b, how='intersection')
        
        # Filter out self-overlaps
        overlaps = overlaps[overlaps['Mine_ID_1'] != overlaps['Mine_ID_2']]

        if overlaps.empty:
            return None # There are no overlaps in this buffer, country

        
        overlaps['Overlap_area'] = overlaps.geometry.area * 10**-6  # Convert to km2

        # Calculate single share
        single_share = overlaps.groupby('Mine_ID_1').apply(
            lambda x: 1 - (x['Overlap_area'].sum() / (2 * x['Territorial_overlap_1'].sum()))
        ).reset_index()

        single_share.rename(columns={'Mine_ID_1': 'Mine_ID', 0: 'Single_share'}, inplace=True)

        if any(single_share['Single_share'] < 0):
            raise ValueError('Negative share detected')

        # Merge back to original dataframe and calculate corrected overlap
        df_b = df_b.merge(single_share, on='Mine_ID', how='left')
        df_b['Territorial_overlap_corrected'] = df_b['Territorial_overlap'] * df_b['Single_share']

        # Select the required columns
        return df_b[['Mine_ID', 'admin', 'buffer', 'mine_area', 'max_buffered_area', 'Territorial_overlap', 'Territorial_overlap_corrected']]
    
    except Exception as e:
        print(f"Error in buffer {b}, admin {c}: {e}")
        return None

def reallocate_area(df, epsg='8857'):
    """
    Efficiently calculate the overlap between the mine polygons in each buffer and distribute the overlapping area
    equally among the mines that share the overlap, using parallel processing.
    """
    # Initialize the area correction column
    df.to_crs(epsg=epsg, inplace=True)
    df['Territorial_overlap'] = df['geometry'].area * 10**-6  # Convert to km2

    # Process each buffer and country combination in parallel
    unique_combinations = df[df['buffer'] > 0][['buffer', 'admin']].drop_duplicates()

    # Parallelize the processing of each unique combination
    results = Parallel(n_jobs=-1, backend='threading')(
        delayed(process_combination)(df, b, c, epsg) for _, (b, c) in tqdm(unique_combinations.iterrows(), total=len(unique_combinations), desc='Processing combinations')
    )

    #buffer_zero = df[df['buffer'] == 0][['Mine_ID', 'admin', 'buffer', 'mine_area', 'max_buffered_area', 'Territorial_overlap']]
    #buffer_zero = buffer_zero.assign(Territorial_overlap_corrected = buffer_zero['Territorial_overlap'])
    # Combine results
    res_df = pd.concat([r for r in results if r is not None], axis=0)
    #res_df = pd.concat([res_df, buffer_zero], axis=0)
    res_df.to_csv('data/interm/corrected_overlap_area.csv', index=False)

    return res_df


def logistic_model(df):
    '''
    This model predicts the probability that a conflict exists (conflict_indicator = 1):

    The logistic regression model will estimate the probability of conflict based on overlap_area.

    '''
    df['conflict_indicator'] = (df['weighted_conflict_sum'] > 0).astype(int)

    X_binary = sm.add_constant(df['overlap_area'])  # Adding a constant for the intercept
    y_binary = df['conflict_indicator']

    # Logistic regression model
    binary_model = sm.Logit(y_binary, X_binary)
    binary_result = binary_model.fit()

    # Display the results of the binary part
    return print(binary_result.summary())


def gamma(df):
    df_nonzero = df[df['conflict_indicator'] == 1]

    # Prepare the predictor (X) and response (y) for the non-zero part
    X_nonzero = sm.add_constant(df_nonzero['overlap_area'])
    y_nonzero = df_nonzero['weighted_conflict_sum']
    # Gamma regression model
    count_model_gamma = sm.GLM(y_nonzero, X_nonzero, family=Gamma()).fit()

    # Display the results of the count part
    print(count_model_gamma.summary())



alloc_path = 'data\interm\conflict_to_alloc.gpkg'
mine_path = 'data\interm\mine_indig_footprint_corrected.gpkg'
alloc = gpd.read_file(alloc_path)
mines = gpd.read_file(mine_path)

if __name__ == "__main__":
    mines_with_con = weighted_sum(alloc, mines)

    reallocate_area(mines_with_con)