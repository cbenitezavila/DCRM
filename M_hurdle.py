import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
from statsmodels.genmod.families import Gamma
import geopandas as gpd
from util import save_fig_plotnine

from plotnine import *

from tqdm import tqdm 

def weighted_sum(alloc, mines):
    sum_per_conflict = alloc.groupby(['Conflict Id', 'buffer'])['inv_centroid_distance'].sum().reset_index()

    # Merge the sum back to the original dataframe
    alloc = alloc.merge(sum_per_conflict[['Conflict Id', 'buffer', 'inv_centroid_distance']], on=['Conflict Id', 'buffer'], suffixes=('', '_sum'))
    alloc['weighted_conflict_sum'] = alloc['inv_centroid_distance'] / alloc['inv_centroid_distance_sum']
    
    mines.rename(columns={'ID': 'Mine_ID'}, inplace=True)
    mines = mines.merge(alloc[['Mine_ID', 'buffer', 'weighted_conflict_sum']], on=['Mine_ID', 'buffer'],how = 'left')
    mines['weighted_conflict_sum'] = mines['weighted_conflict_sum'].fillna(0)

    return mines


def reallocate_area(df):
    '''
    Efficiently calculate the overlap between the mine polygons in each buffer and distribute the overlapping area 
    equally among the mines that share the overlap.
    '''
    df['area_correction'] = 0  # Initialize the area correction column
    
    # Process each buffer separately
    for b in tqdm(df['buffer'].unique()[1:], desc='Buffers'):

        try: 
            df_b = df[df['buffer'] == b][100:]
            
            # Compute all pairwise intersections within the buffer at once
            # Step 1: Calculate all pairwise intersections for the entire dataset at once
            overlaps = gpd.overlay(df_b, df_b, how='intersection')

            # Step 2: Remove self-overlaps by filtering out pairs with identical Mine_IDs
            overlaps = overlaps[overlaps['Mine_ID_1'] != overlaps['Mine_ID_2']]

            # Step 3: Assign a unique ID to each overlap geometry
            overlaps['overlap_id'] = overlaps['geometry'].apply(lambda x: hash(x.wkb))

            # Step 4: Calculate overlap area and group by overlap_id
            overlaps['overlap_area'] = overlaps.geometry.area
            overlap_summary = (overlaps.groupby('overlap_id')
                            .agg(total_overlap_area=('overlap_area', 'sum'),
                                    num_overlaps=('Mine_ID_1', 'nunique'))
                            .reset_index())

            # Step 5: Merge back with original overlaps to distribute overlap areas
            area_correction = overlaps.merge(overlap_summary, on='overlap_id')
            area_correction['distributed_area'] = area_correction['total_overlap_area'] / area_correction['num_overlaps']

            # Step 6: Sum the distributed area for each mine
            area_correction_sum = (area_correction.groupby('Mine_ID_1')
                                .agg(area_correction=('distributed_area', 'sum'))
                                .reset_index())

             # Update the area_correction column in the original dataframe for the current buffer
            df.loc[df['buffer'] == b, 'area_correction'] = (
                df[df['buffer'] == b].merge(area_correction_sum, 
                                            left_on='Mine_ID', 
                                            right_on='Mine_ID_1', 
                                            how='left')['area_correction_y']
                .fillna(0)
            )
            
        except:
            raise ValueError('Error in buffer {}'.format(b))
    
    return df


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