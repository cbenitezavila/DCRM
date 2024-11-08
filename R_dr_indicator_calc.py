
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
import mpltern


def read_supply_risk_data(file_path):
    """
    Reads supply risk data from a .txt file into a pandas DataFrame, where the first column is the metal,
    followed by numeric columns, and the last column is the full string describing supply use.
    
    Parameters:
    file_path (str): Path to the input .txt file containing supply risk data.

    Returns:
    pd.DataFrame: DataFrame containing the supply risk data.
    """
    try:
        # Read the .txt file line by line and split appropriately
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # List to store rows
        data = []

        # Iterate through each line to parse data
        for line in lines:
            # Split the line using whitespace
            parts = line.split()

            # Handling cases where material name is two words (like "Coking coal")
            if parts[1] not in  ['Processing', 'Extraction']:
                metal = parts[0] + ' ' + parts[1]
                stage = parts[2]
                numeric_data = parts[3:7]
                supply_use = ' '.join(parts[7:])
            else:
                metal = parts[0]
                stage = parts[1]
                numeric_data = parts[2:6]
                supply_use = ' '.join(parts[6:])

            # Append the parsed row to data list
            data.append([metal] + [stage] + numeric_data + [supply_use])

        # Create DataFrame from the parsed data
        columns = ['Commodity', 'Stage', 'SR', 'EI', 'IR', 'EoL-RIR',  'Supply used in SR calc.']
        df = pd.DataFrame(data, columns=columns)

        # Convert numeric columns to appropriate data types
        numeric_columns = [ 'SR', 'EI', 'IR', 'EoL-RIR']
        string_columns = ['Commodity', 'Stage', 'Supply used in SR calc.']

        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df[string_columns] = df[string_columns].astype(str)

        return df
    
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")
    

def plot_crm(df, ei_threshold, si_threshold, m = 'score', vars = ['EI', 'SR']):
    
    data = df[df['Allocation_Method'] == m]

    data['CR_norm'] = StandardScaler().fit_transform(data[['CR']].values)

    cr_thres = data['CR_norm'].mean()

    thres = {'CR': cr_thres, 'EI': ei_threshold, 'SR': si_threshold}

    coms_critical_before = data[(data['EI'] > ei_threshold) & (data['SR'] > si_threshold)]['Commodity'].unique()
    coms_critical = data[(data['CR_norm'] > cr_thres) & (data['EI'] > ei_threshold) & (data['SR'] > si_threshold)]['Commodity'].unique()
    
    for v in vars:
        
        f, ax, = plt.subplots(1, 2, figsize=(16, 8))


       
        
        sns.scatterplot(data=data, x='EI', y='SR', ax = ax[0], color = '#1f78b4')

        sns.scatterplot(data=data, x=v, y='CR_norm', ax = ax[1], color = '#1f78b4')

        
        
        ax[1].axvline(thres[v], color='#e31a1c', linestyle='--', label=f'{v} threshold')
        ax[1].axhline(cr_thres, color='#e31a1c', linestyle='--', label='CR threshold')

        ax[0].axvline(ei_threshold, color='#e31a1c', linestyle='--', label='EI threshold')
        ax[0].axhline(si_threshold, color='#e31a1c', linestyle='--', label='SR threshold')

        # Collect annotations
        texts = {0:[], 1:[]}
        for i, row in data.iterrows():
            texts[0].append(ax[0].text(row['EI'], row['SR'], row['Commodity'], fontsize=7))
            texts[1].append(ax[1].text(row[v], row['CR_norm'], row['Commodity'], fontsize=7))
        
        # Automatically adjust text positions to avoid overlap
        adjust_text(texts[0], arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax[0])
        adjust_text(texts[1], arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax[1])



        ax[0].set_xlabel('EI')
        ax[0].set_ylabel('SR')
        ax[1].set_xlabel(v)
        ax[1].set_ylabel('CR_norm')

        
        plt.tight_layout()
        plt.savefig(f'fig/crm_cr_vs_{v}.png')
        plt.show()





def determine_com_match(crm, snl):
    mapping_table = crm.merge(snl['commodity'].drop_duplicates(), left_on='Commodity', right_on='commodity', how='left')
    mapping_table.rename(columns={'commodity': 'SNL_commodity'}, inplace=True)

    mannual_map = {'Iron ore':'Iron Ore', 'Magnesite': 'Magnesium', 'Phosphate rock': 'Phosphate' }

    mapping_table['SNL_commodity'] = mapping_table['SNL_commodity'].fillna(mapping_table['Commodity'].map(mannual_map))
    
    # all commodities in SNL_Commodity are snl commodities assert
    assert mapping_table['SNL_commodity'].dropna().isin(snl['commodity'].drop_duplicates().dropna()).all(), \
    "Some commodities in mapping_table are not found in the initial classification (snl['commodity'])."

    mapping_table.to_csv('data/interm/mapping_table.csv', index=False)


    return mapping_table

def cr(to, crm, snl):
    mapp = determine_com_match(crm, snl)
    
    to_g = to.groupby(['Commodity', 'Allocation_Method'])[['Value', 'Error']].sum().reset_index() 
    mapp.drop('Commodity', axis=1, inplace=True)
    
    merge = mapp.merge(to_g, left_on='SNL_commodity', right_on='Commodity', how='left')

    merge_score = merge[(merge['SNL_commodity'].notna())]

    merge_score.rename(columns={'Value': 'CR'}, inplace=True)

    plot_crm(merge_score, ei_threshold, si_threshold,  m='score')

    return None

if __name__ == '__main__':
    si_threshold = 1
    ei_threshold = 2.8

    crm_path= r'data\crm_eu_2017.txt'
    snl_path = 'data/interm/snl_boot.csv'
    to_path = r'data\interm\commodity_footprint_per_mine.csv'


    crm = read_supply_risk_data(crm_path)
    snl = pd.read_csv(snl_path)
    to = pd.read_csv(to_path)

    cr = cr(to, crm, snl)



    


    


    