class mine:
    def __init__(self, mine_id=None, com=None):
        self.mine_id = mine_id
        self.commodities = com # list of commodities
        self.com_weight = None


    def calc_weights(self):
        commodities = [c for c in self.commodities] # dict of weights ad commodities



    def optimize_weight(self, list_com):
        params = [c.self.opt_weigth for c in self.commodities]
        bounds = [c.int for c in self.commodities]


    def loss_func(self, params, list_com):
        # params is a list of weights
        # list_com is a list of commodities
        # returns the loss function
        return abs(sum([i for i in params]) - 1)


        

class commodity:
    def __init__(self, name = None, conf_int = None):
        self.name = name
        self.int = conf_int # tuple of (upper, lower)
        self.opt_weigth = None

    
        