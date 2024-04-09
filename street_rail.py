
import osmnx as ox
import networkx as nx
import shapely.geometry as sg
import numpy as np




class Street_network:
    def __init__(self, mine_id= None, p=None, buffer=None ):
        self.mine_location = p
        self.type = 'drive'
        self.mine_id = mine_id
        self.buffer = buffer
        self.graph_osmnx = self.get_graph()  # Fix: Added 'self.' to method call
        self.graph_sdp = self.transform_to_line_geometry()  # Fix: Added 'self.' to method call
        self.net_properties = self.get_net_properties()  # Fix: Added 'self.' to method call
        self.average_degree = self.net_properties['average_degree']
        self.stdv_degree = self.net_properties['stdv_degree']
        self.num_nodes = self.net_properties['num_nodes']
        self.num_edges = self.net_properties['num_edges']

    def get_graph(self):
        try:
            G_drive = ox.graph_from_point((self.mine_location.y, self.mine_location.x), dist=self.buffer, network_type=self.type, simplify=True)  # Fix: Replaced 'self.p' with 'self.mine_location'
        except (ox._errors.InsufficientResponseError, ValueError, nx.NetworkXPointlessConcept):
            G_drive = None

        return G_drive
    
    def transform_to_line_geometry(self):

        if self.graph_osmnx is None:
            return None
        else:
            lines = ox.graph_to_gdfs(self.graph_osmnx, nodes=False, edges=True)
            line_geometries = [sg.LineString(line['geometry']) for _, line in lines.iterrows()]
            multi_line_string = sg.MultiLineString(line_geometries)
            return multi_line_string
    
    def get_net_properties(self):
        if self.graph_osmnx is None:
            return None
        else:
            #The degree centrality for a node v is the fraction of nodes it is connected to.
            network = nx.Graph(self.graph_osmnx)
            centrality = nx.degree_centrality(network)
            self.average_degree = np.mean(list(centrality.values()))  # Fix: Added 'list()' to convert dict_values to a list
            self.stdv_degree = np.std(list(centrality.values()))  # Fix: Added 'list()' to convert dict_values to a list
            self.num_nodes = network.number_of_nodes()
            self.num_edges = network.number_of_edges()
            self.density = nx.density(network)

            return {'average_degree': self.average_degree, 'stdv_degree': self.stdv_degree, 'num_nodes': self.num_nodes, 'num_edges': self.num_edges, 'density': self.density}
        

class Rail_network(Street_network):

    #todo : add the railway type to the graph , does not work so far
    
    ox.settings.useful_tags_way += ["railway"]

    def __init__(self, mine_id=None, p=None, buffer=None):
        #super().__init__(mine_id, p, buffer)
        self.mine_location = p
        self.mine_id = mine_id
        self.buffer = buffer
        self.type = '["railway"]'
        self.graph_osmnx = self.get_graph()
        self.graph_sdp = self.transform_to_line_geometry()
        
        def get_graph(self):
            try:
                G_drive = ox.graph_from_point((self.mine_location.y, self.mine_location.x), dist=self.buffer, costume_filter=self.type, simplify=True)  # Fix: Replaced 'self.p' with 'self.mine_location'
            except (ox._errors.InsufficientResponseError, ValueError, nx.NetworkXPointlessConcept):
                G_drive = None

            return G_drive

            
        

    
        