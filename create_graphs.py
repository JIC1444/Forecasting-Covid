import geopandas as gpd
import pandas as pd
from libpysal import weights
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

# Helper functions.
from graph_helper_functions import *

# Read and create a NetworkX graph from the geojson file.
us_counties = gpd.read_file('./data/raw/counties.geojson')
us_counties = us_counties.to_crs(epsg=3857)
centroids = np.column_stack((us_counties.centroid.x, us_counties.centroid.y))
queen = weights.Queen.from_dataframe(us_counties, use_index=False)
graph = queen.to_networkx()


def main(df, setname):
    df = df.rename(columns={"countyFIPS": "countyfips"})
    df = df.dropna(subset="countyfips", axis=0)

    # Remove the nodes/rows from geojson data, where counties are not in the df.
    us = us_counties.copy()

    # Force the countyfips columns to be integer, rather than float.
    df['countyfips'] = df['countyfips'].astype(int)
    us['countyfips'] = us['countyfips'].astype(int)
    counties1 = set(df['countyfips'])
    counties2 = set(us['countyfips'])
    common_counties = counties1.intersection(counties2)
    df = df[df['countyfips'].isin(common_counties)]

    # Merge the dataframes
    cdf = pd.merge(us, df, on=['countyfips'], how='left')
    
    # Assign properties to the graph nodes.
    dates = list(cdf.columns[13:])
    attrs = ["countyfips"] + dates # INCLUDE THE GEOMETRY/POS OF THE COUNTY?

    for att in attrs:
        assign_attributes(cdf, graph, attr=att)
    
    # Test: ommiting every state bar one.
    # Need a dictionary with state: [fips] where the fips are the counties in the states.
    state_to_cfip = {}
    for state in cdf.statefp.unique():
        temp = cdf[cdf['statefp'] == state]
        countyfps = temp.countyfips.to_list()
        state_to_cfip[state] = countyfps

    # Need a dictionary to then convert the countyfp to the node number.
    fp_to_node = {}
    for node, data in graph.nodes(data=True):
        fp_to_node[data['countyfips']] = node

    # Now use this to filter the graph to a one-state graph.
    # state_to_fip gives the entire state's worth of fips.
    # Each individual fip can be used to build a dictionary with state to node.
    state_to_node = {}
    for statefp, countyfpss in state_to_cfip.items():
        li = [fp_to_node[cfp] for cfp in countyfpss]
        state_to_node[statefp] = li

    states = cdf.statefp.to_list()
    save_loc = f"./data/processed/{setname}_graphs"
    create_graphs(save_loc, states, state_to_node, dates)


# Run for all three subsets of the data.
if '__name__' == '__main__':
    traindf = pd.read_csv('./data/processed/TRAIN_SCALED.csv')
    main(traindf, setname='train')

    valdf = pd.read_csv('./data/processed/VAL_SCALED.csv') 
    main(valdf, setname='val')

    testdf = pd.read_csv('./data/processed/TEST_SCALED.csv') 
    main(testdf, setname='test')