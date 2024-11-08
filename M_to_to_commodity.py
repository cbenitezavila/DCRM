import util
import pandas as pd
import numpy as np
import D_com_compare
from tqdm import tqdm

'''
This script runs so far under the assumption that the buffer value is 0 for all mines.
'''




logger = util.logger_init()

class Commodity:

    _instances = []
    
    def __init__(self, name, mine_id, buffer, area, country):
        self.name = name
        self.mine_id = mine_id
        self.buffer_mine = buffer
        self.area_mine = area
        self.country = country
        self.simple_weight = None
        self.conf_interval_weight = None
        self.conf_interval_error = None
        self.method_eval = {}
        Commodity._instances.append(self)

        

    @classmethod
    def get_instances(cls):
        return cls._instances

    def add_mine_to_com(self, area):
        try:
            self.area_mine = area
        except Exception as e:
            logger.error(f"Error adding mine to Commodity {self.name}: {e}")
            raise

    def add_conf_interval_weight(self, weight, error):
        try:
            self.conf_interval_weight = weight
            self.conf_interval_error = error
           
        except Exception as e:
            logger.error(f"Error adding confidence interval weight to Commodity {self.name}: {e}")
            raise
    def calculate_allocation(self, method):
                    
            if method == 'simple':
                try:
                    
                    self.get_simple_weight()
                    
                    assert self.simple_weight is not None, "simple_weight is not set"

                    value = self.area_mine * self.simple_weight
                    error = value * area_error_par*self.simple_weight
                
                except AssertionError as ae:
                    logger.error(f"AssertionError in simple_allocation for Commodity {self.name}: {ae}")
                    raise
                except Exception as e:
                    logger.error(f"Error in simple_allocation for Commodity {self.name}: {e}")
                    raise

            elif method == 'weight':
                try:
                    assert self.conf_interval_weight is not None, "conf_interval_weight is not set"
                    assert self.area_mine is not None, "area_mine is empty"

                    value = self.area_mine * self.conf_interval_weight
                    area_error = area_error_par * self.area_mine  # Assuming area_error is std deviation of area_mine
                    
                    # error calculation -> propagation
                    error = value * np.sqrt(
                        (area_error/self.area_mine)**2 + 
                        (self.conf_interval_error/self.conf_interval_weight)**2)
        
                except AssertionError as ae:
                    logger.error(f"AssertionError in weight_allocation for Commodity {self.name}: {ae}")
                    raise
                except Exception as e:
                    logger.error(f"Error in weight_allocation for Commodity {self.name}: {e}")
                    raise
            
            elif method == 'score':
                try:
                    assert self.conf_interval_weight is not None, "conf_interval_weight is not set"
                    assert self.area_mine is not None, "area_mine is empty"
                    
                    list_com = self.get_mine().list_commodities # commodities of the mine
                
                    sum_weights = sum([self.get_commodity_same_mine(c).conf_interval_weight for c in list_com])
                    b = self.conf_interval_weight / sum_weights
                    value = self.area_mine * b

                    # error calculation -> propagation

                    error_sum = np.sqrt(sum([self.get_commodity_same_mine(c).conf_interval_error**2 for c in list_com]))
                    error_b = b *np.sqrt((self.conf_interval_error/sum_weights)**2 + (error_sum/sum_weights)**2)
                    error_area = area_error_par * self.area_mine
                    error= value * np.sqrt((error_b/b)**2 + (error_area/self.area_mine)**2)

        
                except AssertionError as ae:
                    logger.error(f"AssertionError in weight_allocation for Commodity {self.name}: {ae}")
                    raise
                except Exception as e:
                    logger.error(f"Error in weight_allocation for Commodity {self.name}: {e}")
                    raise
            
            else:
                raise ValueError(f"Method {method} is not valid")
            return (value, error)
            


    def get_simple_weight(self):
        try:
            assert self.mine_id != None, "mine_ids is empty"
            assert self.buffer_mine != None, "buffer_mine is empty"

            mine_instance = self.get_mine()
            self.simple_weight = 1/len(mine_instance.list_commodities)
        
        except AssertionError as ae:
            logger.error(f"AssertionError in get_simple_weight for Commodity {self.name}: {ae}")
            raise
        except Exception as e:
            logger.error(f"Error in get_simple_weight for Commodity {self.name}: {e}")
            raise

        
    def get_mine(self):
        return next((mine for mine in Mine.get_instances() if mine.id == self.mine_id and mine.buffer == self.buffer_mine), None)
    
    def get_commodity_same_mine(self, name):
        return next((commodity for commodity in Commodity.get_instances() if commodity.name == name and commodity.mine_id == self.mine_id and commodity.buffer_mine == self.buffer_mine), None)   



class Mine:
    _instances = []
    def __init__(self, id, buffer, country):  
        self.id = id
        self.buffer = buffer 
        self.area = None
        self.country = country
        self.list_commodities = []
        Mine._instances.append(self)

    @classmethod
    def get_instances(cls):
        return cls._instances

    def init_area(self, area, max_area):
        try:
            self.area = area
            self.max_buffered_area = max_area
            
        except Exception as e:
            logger.error(f"Error initializing area for Mine {self.id}: {e}")
            raise

    def init_commodities(self, list_commodities):
        try:
            self.list_commodities = list_commodities

        except Exception as e:
            logger.error(f"Error initializing commodities for Mine {self.id}: {e}")
            raise

    

def initialize_mines(df, compare):
    df.drop('geometry', axis=1, inplace=True)
    mine_counter = 0  # Initialize the counter
   

    df = df[df['buffer'] == buffer]  # Filter out mines with buffer > 0

    try:
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Initializing Mines"):
            if row['list_of_commodities'] is None:
                continue  # Skip mines with no commodities allocated

            mine = Mine(id=row['ID'], buffer=row['buffer'] , country=row['adm0_a3'])
            mine.init_area(row['overlay_area'], row['max_buffered_area'])
            mine.init_commodities(str_to_list_com_com(row['list_of_commodities'], compare))
            mine_counter += 1  # Increment the counter for each initialized mine
            
        logger.info(f"Initialized {mine_counter} Mines.")
    except Exception as e:
        logger.error(f"Error initializing Mines: {e}")
        raise
    
def str_to_list_com_com(commodities, compare):
    if commodities is None:
        return None
    list_commodities = [c.strip() for c in commodities.split(',')]
    list_commodities_allowed = [c for c in list_commodities if c in get_common_commodities(compare).values]
    return list_commodities_allowed

def get_common_commodities(compare):
    common_commodities = compare[(compare['In_SNL'] == True) & (compare['In_Mine'] == True) & (compare['Buffer'] == 0)]['Commodity']
    return common_commodities

def initialize_commodities(overlay, snl, compare):
    logger.info('Initializing Commodities')

    snl['conf_interval_weight'] = (snl['conf_upper_bound'] + snl['conf_lower_bound']) / 2 # center of the confidence interval
    snl['conf_interval_error'] = snl['conf_length'] / 2 # half of the confidence interval length as error



    common_commodities = get_common_commodities(compare)
    logger.debug(f"Common commodities count: {len(common_commodities)}")
    logger.debug(f"Common commodities: {common_commodities.tolist()}")

    initialize_mines(overlay, compare)

    snl_filtered = snl[snl['conf_level'] == 0.99].groupby('commodity')

    try:
        mines = Mine.get_instances()
        logger.debug(f"Mines count: {len(mines)}")       

        for c in tqdm(common_commodities, desc="Processing Commodities"):  # for all common commodities
            if c not in snl_filtered.groups:
                 continue  # Skip if the commodity isn't in the filtered data
    
            snl_group = snl_filtered.get_group(c)
            
            for m in mines:  # for all mines
                if c not in m.list_commodities:
                    continue  # Skip if the commodity isn't in the mine's list of commodities

                com = Commodity(name=c, mine_id=m.id, buffer=m.buffer, area=m.area, country=m.country)
                # Use the pre-filtered data for commodity 'c' instead of filtering every time
                com.conf_interval_weight = snl_group['conf_interval_weight'].mean()
                com.conf_interval_error = snl_group['conf_interval_error'].mean()
        
        for com in tqdm(Commodity.get_instances(), desc="Calculating Allocation Scenarios"): 
            if com.name in prio_commodities:
                for method in methods_to_evaluate:
                    com.method_eval[method] = com.calculate_allocation(method = method)
                   

    except Exception as e:
        logger.error(f"Error initializing Commodities: {e}")
        raise



def commodity_instances_to_df(overlay, snl, compare):
    initialize_commodities(overlay, snl, compare)

    df = None  # Initialize df outside try block

    try:
        data = []
        for c in Commodity.get_instances():
            if c.name in prio_commodities:
                for m in methods_to_evaluate:
                        row = {
                            'Commodity': c.name,
                            'Mine_ID': c.mine_id,
                            'Buffer': c.buffer_mine,
                            'Area': c.area_mine,
                            'Country': c.country,
                            'Simple_Weight': c.simple_weight,
                            'Conf_Interval_Weight': c.conf_interval_weight,
                            'Conf_Interval_Error': c.conf_interval_error,
                            'Allocation_Method': m,
                            'Value': c.method_eval[m][0],
                            'Error': c.method_eval[m][1]
                        }
                        data.append(row)

        df = pd.DataFrame(data)  # Move DataFrame creation outside the loop

        

        df.to_csv('data/interm/commodity_footprint_per_mine.csv', index=False)

    except Exception as e:
        logger.error(f"Error converting commodity instances to DataFrame: {e}")
        raise

    return df


if __name__ == '__main__':
    area_error_par = .01
    buffer = 0
    
    methods_to_evaluate = ['simple', 'weight', 'score']

    snl_weights_path = 'data/interm/snl_boot.csv'
    compare_path = 'data/interm/commodity_consistency_check.csv'
    mine_indig_path = 'data/interm/mine_indig_footprint_corrected.gpkg'

    map_table_path = 'data/interm/mapping_table.csv'
    

    snl_weights = pd.read_csv(snl_weights_path)
    mine_indig = util.load_gpd(mine_indig_path, logger=logger)
    compare = util.load_csv(compare_path, logger=logger)
    map_table = pd.read_csv(map_table_path)

    prio_commodities = map_table['SNL_commodity'].dropna().unique().tolist()

    

    res = commodity_instances_to_df( overlay=mine_indig, snl=snl_weights, compare=compare)

    

