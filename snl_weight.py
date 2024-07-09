import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import uniform
from scipy.stats import bootstrap
from copy import deepcopy
import logging
import os
from joblib import Parallel, delayed, Memory
import fiona
import util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up joblib memory caching
cache_dir = 'cache_dir'

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def snl_data_processing(snl_prod, mine_prop):
    logger.info(f"Start: Preprocessing")
    snl_filterextend = snl_prod[~snl_prod.value.isna() & ~snl_prod.value.isnull()]

    # Ensure 'snl_id' is of the same type in both DataFrames
    mine_prop['snl_id'] = mine_prop['snl_id'].astype(str)
    snl_filterextend['snl_id'] = snl_filterextend['snl_id'].astype(str)
    # Merge the DataFrames
    vsnl_join = mine_prop.merge(snl_filterextend, on='snl_id')

    vsnl_sub = vsnl_join[['snl_id', 'mine', 'value', 'unit', 'year', 'commodity', 'list_of_commodities', 'development_stage']]

    vsnl_harmo = vsnl_sub.copy()
    # tonne conversion 
    vsnl_harmo['value'] = vsnl_harmo.apply(convert_to_tonnes, axis=1)

    vsnl_harmo['unit'] = 'tonne'
    # filter out value = 0
    vsnl_filter = vsnl_harmo[~(vsnl_harmo.value ==0)]

    #Methodological choice: Concentration on Expansion and Operating development stages. Reasons:
    #1. Validity that production is really production and not only exploration
    #2. other development stages not uniquely defined
    #3. >90\% of the operations observations are coverd.

    vsnl_fin = vsnl_filter[vsnl_filter.development_stage.isin(['Operating', 'Expansion'])]
    logger.info(f"Finish: Preprocessing")
    return vsnl_fin

def development_type_table(snl_prod, mine_prop):
    data = snl_data_processing(snl_prod, mine_prop)

    dev_status_year = data.groupby('year')['development_stage'].value_counts().reset_index()

    # Pivot the DataFrame
    pivot_df = dev_status_year.pivot(index='development_stage', columns='year', values='count')

    # Fill missing values with 0 (if needed)
    pivot_df = pivot_df.fillna(0).astype(int) / pivot_df.sum(axis=0) * 100

    pivot_df.round(2)

    return df_to_latex(pivot_df)


def df_to_latex(df, file_name):
    # Export to LaTeX
    latex_table = df.to_latex(index=True)

    # Write the LaTeX table to a file
    with open('tabs/'+file_name+'.tex', 'w') as f:
        f.write(latex_table)


def commodity_bootstrap(data, stat=np.mean, con_levels=[0.9, 0.95, 0.99], **kwargs):
    
    data = np.array(data)
    result = []

    sample_size = len(data)

    if len(data) < 2:
        result = [(sample_size, _, (0, 1), np.nan) for _ in con_levels]

    elif len(np.unique(data)) == 1:
        share = float(np.unique(data))
        result = [(sample_size, _, (share, share), 0 ) for _ in con_levels]

    else:
        for con_level in con_levels:
            boot = bootstrap((data,), statistic=stat, confidence_level=con_level)
            conf_interval = boot.confidence_interval
            conf_interval_round = tuple(round(bound, 3) for bound in conf_interval)

            std_error = round(boot.standard_error, 3)

            result.append((sample_size, con_level, conf_interval_round, std_error))

    return result


def share_calc(snl_prod, mine_prop):
    logger.info(f"Start: Share calculation")
    data = snl_data_processing(snl_prod, mine_prop)

    share = (data.groupby(['snl_id', 'year', 'commodity' ])
              .agg({'value': 'sum'}).round(1)
              .reset_index())

    share['value_total'] = share.groupby(['snl_id', 'year'])['value'].transform('sum').round(2)
    share['share'] = share['value'] / share['value_total'].round(2)

    logger.info(f"Finish: Share calculation")
    return share



def df_bootrstrap(snl_prod, mine_prop):
    logger.info(f"Start: Bootsrap")

    data = share_calc(snl_prod, mine_prop)

    share_boot = (data.groupby(['commodity', 'year'])['share']
              .apply(lambda x: pd.DataFrame(commodity_bootstrap(x)))
              .reset_index(level=-1, drop=True)
              .reset_index()
              .rename(columns={0:'sample_size', 1: 'conf_level', 2: 'conf_interval', 3: 'std_error'})
)
    # Expand the tuples into separate columns and rename columns for clarity
    share_boot[['conf_lower_bound', 'conf_upper_bound']] = pd.DataFrame(share_boot['conf_interval'].tolist())


    share_boot_total = total_value_addition(share= data, boot=share_boot)

    # Calculate the confidence interval length and prettify the DataFrame
    share_boot_total['conf_length'] = share_boot_total['conf_upper_bound'] - share_boot_total['conf_lower_bound']
    share_boot_pretty = share_boot_total.drop(columns=['conf_interval'])
    

    logger.info(f"Finish: Bootsrap")
    return share_boot_pretty

def total_value_addition(share, boot):
    com_group = share.groupby(['commodity', 'year'])['value_total'].sum().reset_index()

    boot_val_tot = boot.merge(com_group, on=['commodity', 'year'])

    return boot_val_tot


    

def convert_to_tonnes(row):
    value_in_tonnes = row['value'] * conversion_factors[row['unit']]
    return value_in_tonnes

if __name__ == '__main__':

    conversion_factors = {
    'tonne': 1,
    'oz': 2.83495e-5, # 1 troy ounce = 2.83495e-5 tonnes
    'lb': 0.000453592, # 1 pound = 0.000453592 tonnes
    'ct': 2e-7 # 1 carat = 2e-7 tonnes
    }

    mine_path = 'data\mine-comm\snl_mining_properties.gpkg'
    snl_path = 'data\mine-comm\snl_production_values.csv'

    mines = util.load_gpd(mine_path, logger=logger)
    snl = util.load_csv(snl_path, logger=logger)

    boot = df_bootrstrap(mine_prop=mines, snl_prod=snl)

    if boot is not None:
            
            boot.to_csv('data\interm\snl_boot.csv', index=False)
            print(boot.head())
            logger.info('Boot calc finished and saved.')

    else:
       logger.error('Hausdorff distance calculation failed.')