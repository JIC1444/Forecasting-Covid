import geopandas as gpd
import pandas as pd
from libpysal import weights
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Torch.
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

# Graph creation helper functions.
def assign_attributes(cdf, graph, attr):
    node_to_attr = dict(enumerate(cdf[attr]))
    nx.set_node_attributes(graph, node_to_attr, attr)

def filter_graph(graph, positions, node_subset):
    pos_filtered = {n: positions[n] for n in node_subset}
    return graph.subgraph(node_subset), pos_filtered

def state_graph(state_fp, state_to_node, graph, positions, us_counties, showplot=False):
    """state_fp: str(int)"""
    node_subset = state_to_node[state_fp]
    graph_filtered, pos_filtered = filter_graph(graph, positions, node_subset)
    # Show the plot on the map backing.
    if showplot:
        ax = us_counties.plot(linewidth=1, edgecolor='grey', facecolor='lightblue', figsize=(10, 5))
        ax.axis('off')
        nx.draw(graph_filtered, pos=pos_filtered, ax=ax, node_size=5, node_color='green')
        plt.show()
    return graph_filtered, pos_filtered

def nx_to_pyg(G, pos, dates): # Custom function required because of the odd names of the nodes (i.e California has nodes 0, 682, 1025 etc.)
    # Rename the nodes 0:num_nodes. Key is old name, value is new name.
    nodes = {node: n for n, node in enumerate(G.nodes)}

    # Transfer graph from nx to pytorch_geometric.
    edge_index = torch.zeros((2, len(G.edges)), dtype=torch.long)
    for i, (n1, n2) in enumerate(G.edges):
        tens = torch.tensor([nodes[n1], nodes[n2]])
        edge_index[:, i] += tens # Index is unimportant here since the node, node pair is all that is needed.

    x = torch.zeros((len(nodes.keys()), len(dates)), dtype=torch.float32) # (num_nodes, num_node_features).
    i = 0 # Count through the 1st axis of the feature matrix.
    for i, date in enumerate(dates):
        if '-' not in date: # Skip if not a date.
            continue
        cases = nx.get_node_attributes(G, date) # Dictionary of all the nodes on a given day.
        cases_nn = {nodes[node]: case for node, case in cases.items()} # New node name: case.
        for node, case in cases_nn.items():
            x[node, i] += case
    
    # Order-preserving since the nodes otherwise come out randomly assigned.
    tpos = torch.zeros((len(nodes.keys()), 2)) #(num_nodes, num_dimensions).
    for i, node in enumerate(nodes.keys()):
        # Ensure pos[node] exists and convert it to a tensor.
        tpos[i, :] = torch.tensor(pos[node], dtype=torch.float32).flatten()

    return Data(x=x, edge_index=edge_index, pos=tpos)

# Create and save the state graphs - in the format statename.pt
def create_graphs(save_loc, states, state_to_node, dates, graph):
    os.makedirs(save_loc, exist_ok=True)

    for state in states:
        graph_filtered, pos_filtered = state_graph(state_fp=state, state_to_node=state_to_node)
        H = nx_to_pyg(graph_filtered, pos_filtered, dates)
        save_g_loc = os.path.join(save_loc, f"{state}.pt")
        if not os.path.exists(save_g_loc):
            torch.save(H, save_g_loc)