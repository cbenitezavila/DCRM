
import util
import pandas as pd

logger = util.logger_init()
from collections import defaultdict




def mine_to_list(mine_indig):
    mine_all_commodities = [commodity.strip() for sublist in mine_indig.dropna().str.split(',') for commodity in sublist if commodity.strip() != 'None']
    return mine_all_commodities


def check_commodity(snl_weights, mine_indig):
        logger.info('Checking commodities between SNL and mine_indig...')
        snl_com = snl_weights['commodity'].unique()
        
        mine_com_buffer = {}
        mine_indig_group = mine_indig.groupby('buffer')
        mine_com_buffer = {buffer: mine_to_list(group['list_of_commodities']) for buffer, group in mine_indig_group}


        mine_all_commodities = mine_to_list(mine_indig['list_of_commodities']) #all commodities in mine_indig
        mine_unique = list(set(mine_all_commodities)) # unique commodities in mine_indig
        snl_unique = list(set(snl_com)) #unique commodities in snl_weights
        
        result = commodity_compare_table(snl_unique, mine_unique, mine_com_buffer)
        
        logger.info('Finished commodity check')

        return result


def commodity_compare_table(snl_com, mine_com, mine_com_buffer):
    # Initialize dictionaries to store buffer counts
    commodity_buffer_counts = defaultdict(lambda: defaultdict(int))

    # Helper function to determine presence in buffers and count occurrences
    def get_buffers_and_counts(commodity):
        buffer_counts = defaultdict(int)
        for buffer_name, commodities in mine_com_buffer.items():
            buffer_counts[buffer_name] = commodities.count(commodity)
        return buffer_counts

    # Calculate common, SNL-only, and mine-only commodities
    common_com = set(snl_com).intersection(mine_com)
    snl_not_mine = set(snl_com).difference(mine_com)
    mine_not_snl = set(mine_com).difference(snl_com)

    all_commodities = sorted(common_com | snl_not_mine | mine_not_snl)

    # Prepare data for DataFrame
    data = []
    for commodity in all_commodities:
        in_snl = commodity in snl_com
        in_mine = commodity in mine_com
        buffer_counts = get_buffers_and_counts(commodity)
        
        # Append separate rows for each buffer
        for buffer_name, count in buffer_counts.items():
            row = {
                'Commodity': commodity,
                'In_SNL': 'TRUE' if in_snl else 'FALSE',
                'In_Mine': 'TRUE' if in_mine else 'FALSE',
                'Buffer': buffer_name,
                'Mine_Count': count
            }
            data.append(row)
        
        # Update commodity_buffer_counts with buffer counts
        commodity_buffer_counts[commodity].update(buffer_counts)

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


def mine_list_to_tidy(mine_indig):
    if 'geometry' in mine_indig.columns:
        df = mine_indig.copy().drop('geometry')
    else:
        df = mine_indig.copy()
        
    df['list_of_commodities'] = df['list_of_commodities'].str.split(',')
    tidy_df = df.explode('list_of_commodities').reset_index(drop=True)
    return tidy_df



if __name__ == '__main__':

    snl_weights_path = 'data/interm/snl_boot.csv'

    mine_indig_path = 'data/interm/mine_indig_footprint_corrected.gpkg'

    snl_weights = util.load_csv(snl_weights_path, logger=logger)
    mine_indig = util.load_gpd(mine_indig_path, logger=logger)

    mine_crop = mine_list_to_tidy(mine_indig)

    



