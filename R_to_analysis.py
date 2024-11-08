import pandas as pd
from util import load_csv, load_gpd, logger_init
import seaborn as sns
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from pathlib import Path
from adjustText import adjust_text

from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

def abs_com_footprint(data):
    # Group the data by 'Commodity' and 'Allocation_Method', summing 'Value' and 'Error'
    data_g = data.groupby(['Commodity', 'Allocation_Method'])[['Value', 'Error']].sum().reset_index()

    # Create the plot
    f, ax = plt.subplots(figsize=(10, 7))
    
    # Create the barplot without confidence intervals (ci=None)
    sns.barplot(x='Commodity', y='Value', data=data_g, hue='Allocation_Method', palette='rocket', ci=None, ax=ax)

    # Customize the plot
    ax.set(ylabel="TO (km2)", xlabel="Commodity")
    plt.xticks(rotation=45, fontsize=6)
    plt.title('TO (km2) of Commodity by Allocation Method')

    # Save the figure and show the plot
    plt.savefig(f'fig/to_abs.png')
    plt.show()

    plt.close()

    f, ax = plt.subplots(figsize=(10, 7))
    data_g['Coef_Var'] = data_g['Error'] / data_g['Value']

    sns.barplot(x='Commodity', y='Coef_Var', data=data_g, hue='Allocation_Method', palette='rocket', ci=None, ax=ax)

    plt.xticks(rotation=45, fontsize=6)
    plt.ylabel('Coefficient of Variance (-)')
    plt.title('Coefficient of Variance by Allocation Method')

    # Save the figure and show the plot
    plt.savefig(f'fig/coeff_var_to_abs.png')
    plt.show()

    plt.close()
    


def plot_world_com_facet(data, world_bounds, com_facet,  method='score'):
    data_g = data[(data.Allocation_Method ==method) & (data.Commodity.isin(com_facet))].groupby(['Commodity', 'Country'])[['Value', 'Error']].sum().reset_index()

    world_bounds = world_bounds.to_crs('EPSG:4326')

    join = world_bounds.merge(data_g, left_on='iso3', right_on='Country', how='right')


    g = sns.FacetGrid(join, col='Commodity', col_wrap=3, height=5, aspect=2.5, sharex=True, sharey=True, gridspec_kws={"wspace": 0.5, "hspace": -0.5})

    def plot_facet(data, **kwargs):
        ax = plt.gca()

        data.plot(ax=ax, column='Value', legend=True, cmap='viridis', legend_kwds={'label': "Absolute TO (km2)",'shrink': 0.8})
        world_bounds.boundary.plot(ax=ax, color='grey', linewidth=0.5)
        ax.set_title(data['Commodity'].iloc[0], fontsize=16)
    
    
    g.map_dataframe(plot_facet)
    g.set_axis_labels("Longitude", "Latitude")

    
    plt.tight_layout()
    
    plt.savefig('fig/world_com_facet.png')
    plt.show()


    
def to_normalized_by_production(data, snl, year=2017):
    # Group data and sum 'Value' and 'Error' by 'Commodity' and 'Allocation_Method'

    data_s = data[(data['Allocation_Method'] == 'score')]
    data_g = data_s.groupby(['Commodity', 'Allocation_Method'])[['Value', 'Area', 'Error']].sum().reset_index()
    
    # Get the production values for the given year
    snl_sub = snl[snl['year'] == year][['commodity', 'value_total']].drop_duplicates()
    
    # Merge data with the production values
    join = data_g.merge(snl_sub, left_on='Commodity', right_on='commodity', how='left')

    # Normalize the 'Value' and 'Error' by production
    join['Value_norm'] = join['Value'] / join['value_total'] * 1e6
    join['Error_norm'] = join['Error'] / join['value_total'] * 1e6

    # Prepare for plotting

    highest_com = ['Antimony', 'Niobium', 'Tin', 'Titanium', 'Tungsten', 'Vanadium']


    # f, ax = plt.subplots(1, 2, figsize=(10, 7), width_ratios=[2, 1])

    # # Create barplot without internal error bars
    # sns.barplot(x='Commodity', y='Value_norm', data=join[~join['Commodity'].isin(highest_com)], hue='Allocation_Method',
    #                     palette='rocket', ci=None, ax=ax[0])

    # sns.barplot(x='Commodity', y='Value_norm', data=join[join['Commodity'].isin(highest_com)], hue='Allocation_Method',
    #                     palette='rocket', ci=None, ax=ax[1])


    # ax[0].set(ylabel="TOnorm (m2/t)", xlabel="Commodity")
    # ax[1].set(ylabel="TOnorm (m2/t)", xlabel="Commodity")

    # ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, fontsize=6)
    # ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, fontsize=5)
    
    # plt.subplots_adjust(wspace=0.4)

    # # Save and display plot
    # plt.savefig(f'fig/to_norm_prod_{year}.png')
    # plt.show()
    # plt.close()


    # Create figure and axis
    f, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Get the unique commodities
    unique_commodities = join['Commodity'].unique()

    # Create a custom palette using the 'hls' palette for the unique commodities
    palette = sns.color_palette("hls", len(unique_commodities))

    # Map the custom palette to each commodity
    commodity_palette = dict(zip(unique_commodities, palette))

    join['Production'] = join['value_total'] / 1e6

    # Create scatter plot
    sns.scatterplot(x='Area', y='Value', hue='Commodity', size='Production', palette=commodity_palette, 
                    ax=ax, alpha=0.8, legend='brief', data=join, sizes=(50,300))
    
    ax.legend(fontsize=10)
    # Set axis labels
    plt.ylabel('TO (km2)')
    plt.xlabel('Mine Area (km2)')

    # Create a list to store the annotations for `adjustText`
    texts = []

    # Annotate each commodity exactly once
    for commodity in unique_commodities:
        # Get the subset of data for this commodity
        subset = join[join['Commodity'] == commodity]

        # Calculate the position to place the annotation (mean of x and y coordinates)
        mean_x = subset['Area'].mean()
        mean_y = subset['Value'].mean()

        # Add the annotation and append to the texts list
        texts.append(ax.text(mean_x, mean_y, commodity, fontsize=10,color=commodity_palette[commodity]))

    # Use adjustText to adjust the annotations to avoid overlap
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=.5))

    plt.tight_layout()

    # Save and show the plot
    plt.savefig('fig/scatter_relation_global.png')
    plt.show()


def scatter_relation(data):
    # Filter data
    data_selection = data[(data['Allocation_Method'] == 'score')]

    # Standardize the data
    scaler = StandardScaler()
    data_selection['Mine_area_norm'] = data_selection[['Area']]
    data_selection['TO_norm'] = data_selection[['Value']]

    # Get the unique commodities
    unique_commodities = data_selection['Commodity'].unique()

    # Create a custom palette using the 'hls' palette for the unique commodities
    palette = sns.color_palette("hls", len(unique_commodities))

    # Map the custom palette to each commodity
    commodity_palette = dict(zip(unique_commodities, palette))

    # Create scatter plot
    f, ax = plt.subplots(figsize=(16,10))
    sns.scatterplot(x='Mine_area_norm', y='TO_norm', data=data_selection, hue='Commodity', palette=commodity_palette, ax=ax, alpha=0.4, legend=True)

    ax.legend(fontsize=10)


    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()

    # Set axis labels
    plt.xlabel('Mine Area (log(km2))')
    plt.ylabel('TO (log(km2))')

    # Save the plot
    plt.savefig('fig/scatter_relation.png')
    plt.show()

def scatter_relation_power(data):
    # Filter data
    data_selection = data[(data['Allocation_Method'] == 'score')]

    # Log-transform the data
    data_selection['Mine_area_norm'] = np.log(data_selection['Area'])
    data_selection['TO_norm'] = np.log(data_selection['Value'])

    # Get the unique commodities
    unique_commodities = data_selection['Commodity'].unique()

    # Create a custom palette for commodities
    palette = sns.color_palette("hls", len(unique_commodities))
    commodity_palette = dict(zip(unique_commodities, palette))

    # Create the log-log scatter plot
    f, ax = plt.subplots(figsize=(16,10))
    sns.scatterplot(x='Mine_area_norm', y='TO_norm', data=data_selection, hue='Commodity', 
                    palette=commodity_palette, ax=ax, alpha=0.4, legend=False)

    # Fit and plot power-law regression for each commodity
    for commodity in unique_commodities:
        subset = data_selection[data_selection['Commodity'] == commodity]
        slope, intercept, r_value, p_value, std_err = linregress(subset['Mine_area_norm'], subset['TO_norm'])
        
        # Plot the power law fit line
        x_vals = np.linspace(subset['Mine_area_norm'].min(), subset['Mine_area_norm'].max(), 100)
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, color=commodity_palette[commodity], label=f'{commodity} (slope={slope:.2f}, R^2={r_value**2:.2f})')

    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.xlabel('Mine Area (log(km2))')
    plt.ylabel('TO (log(km2))')
    plt.savefig('fig/scatter_relation_powerlaw.png')
    plt.show()

def main():

    footprint_path = 'data\interm\commodity_footprint_per_mine.csv'
    world_bounds_path = Path('data\world_bound\world-administrative-boundaries.shp')
    snl_path = Path('data\interm\snl_boot.csv')
    log = logger_init()
    # Load data
    
    

    # Load world boundaries

    data = pd.read_csv(footprint_path)
    world_bounds = gpd.read_file(world_bounds_path, engine='pyogrio')
    snl = pd.read_csv(snl_path)

    com_facet = ['Copper', 'Gold', 'Iron Ore', 'Nickel', 'Platinum', 'Silver', 'Zinc', 'Bauxite', 'Potash']

    scatter_relation_power(data)



    return None


if __name__ == '__main__': 
    main()