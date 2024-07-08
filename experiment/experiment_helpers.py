#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import vishn_old as vn
# Other packages used
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import xarray as xr

#%%

def gen_ER_probs(BA_hyperparameter_list, N_nodes):
    """
    Given a list of hyperparameters for generating a barabasi-albert graph, 
    returns a list of edge creation probabilities for an erdos-renyi graph such 
    that the average degree of a node in the ER graph is the same as the BA graph
    """
    homo_hyperparam = []
    # Thus, they are a function of the ER hyperparameters
    for i in BA_hyperparameter_list:
        i = float(i)
        avg_degrees = []
        for j in range(10):
            BA = nx.barabasi_albert_graph(N_nodes, int(i))
            # calculate average degree
            degrees = dict(BA.degree())
            avg_degrees += [sum(degrees.values()) / len(degrees)]
        homo_hyperparam += [np.mean(avg_degrees)/N_nodes]
    return homo_hyperparam


#%% Functions to make running experiments across many hyperparameter combos easier

def gen_param_file(N_nodes, mean_dict, ntwrk_type, hyperparameter, weight, sd=None, seed=None):
    """
    Generates parameter data for a simulation and writes it to a csv file. The 
    parameter data is generated according to a normal distribution with mean 
    values for each parameter according to mean_dict and standarad deviation as 
    sd percent of the mean value.
    
    Args:
    N_nodes (int): number of nodes
    
    mean_dict (dict): keys are name of parameters in the subhost equation and 
    values are the average value of the parameter in the population
    
    hyperparameter (float): hyperparameter for generating the graph
    
    sd (float): percentage of the mean value used for the standard deviation of 
    the parameter within the population
    
    ntwrk_type (str): either "Homo" or "Hetero", for writing to correct directory
    
    weight (float): the weight on the edges of the network that will run the 
    simulation, for writing to correct directory
    
    seed (int): random seed for generation
    
    """
    if seed==None:
        seed = np.random.randint(10000)
    np.random.seed(seed)
    if sd == None:
        singlerow = pd.DataFrame(mean_dict, index=[0])
        paramdata = pd.concat([singlerow]*N_nodes, ignore_index=True)
        paramdata.insert(0, "id", range(0,N_nodes))
        pop_type = "Homo"
        stand_dev = "0"
    else: 
        paramdata = pd.DataFrame()
        paramdata["id"] = pd.Series(range(0,N_nodes))
        for param in mean_dict.keys():
            paramdata[param] = np.random.normal(mean_dict[param], 
                                                sd*mean_dict[param], 
                                                N_nodes)
        pop_type = "Hetero"
        stand_dev = str(sd)
    naming_convention={"Homo":"ER","Hetero":"BA"}
    paramfile_name = pop_type+"Sd"+stand_dev.replace(".","p")+"N"+str(N_nodes)+naming_convention[ntwrk_type]+"w"+str(weight).replace(".","p") + str(hyperparameter).replace(".","p")+".csv"
    directory_path = Path("./generated_files/"+ntwrk_type+"/"+str(weight))
    directory_path.mkdir(parents=True, exist_ok=True)
    paramdata.to_csv(directory_path / paramfile_name, index=False)
    return paramfile_name, str(directory_path / paramfile_name)

#%%

def gen_graph_wparams(g_type, hyperparameter, N_nodes, start_seed):
    """ 
    Generates either an erdos-renyi or barabasi-albert graph according to the 
    supplied hyperparameters. Will continue to generate erdos-renyi graphs 
    until it is connected.
    ___________________________________________________________________________
    
    Args:
    g_type (string): either "Homo" for erdos-renyi or "Hetero" for 
    Barabasi-Albert
    
    hyperparameters (numeric): if Homo, the hyperparameter for edge creation 
    probabilit. If Hetero, or number of edges to add for new nodes
    
    N_nodes (int): Number of nodes in the generated graph
    
    start_seed (int): seed for hetero, starting seed for homo which will 
    continue to increment by 1 until the graph generated is connected
    
    Returns: (networkx.Graph) G
    """
    g_type = g_type.lower()
    if g_type == "homo":
        is_connected = False
        # Throw away any graphs that aren't connected
        ER = None
        while not is_connected:
            ER =  nx.erdos_renyi_graph(N_nodes, hyperparameter, seed=start_seed)
            largest_component = len(max(nx.connected_components(ER), key=len))
            if not (largest_component > 0.9*N_nodes):
                start_seed += 1
            else:
                is_connected = True
        # pick node from largest component as seed for infection
        large_component = max(nx.connected_components(ER), key=len)
        random_node = np.random.choice(list(large_component))
        return ER, random_node
    elif g_type == "hetero":
        hyperparameter = int(hyperparameter)
        BA = nx.barabasi_albert_graph(N_nodes, hyperparameter, seed=start_seed)
        random_node = np.random.choice(list(BA.nodes()))
        return BA, random_node
#%%
    
def weight_and_write(G, g_type, hyperparameter, weight, sd):
    """ 
    Adds a weight to all adges of the supplied graph G and writes the graph to 
    an edgelist
    
    Args:
        G (networkx.graph): graph to apply weights to
        
        g_type (string): either "Homo" for erdos-renyi or "Hetero" for 
        Barabasi-Albert
        
        hyperparameters (numeric): if Homo, the hyperparameter for edge creation 
        probabilit. If Hetero, or number of edges to add for new nodes
        
        weight (float): a real number in the range [0,1]
    
    Returns: G, graph with weight applied to all edges
    """
    # Apply weights
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = weight
    # Write file
    naming_convention={"homo":"ER","hetero":"BA"}
    networkname = naming_convention[g_type.lower()]+str(hyperparameter).replace(".", "")+"w"+str(weight).replace(".", "")+"sd"+str(sd)+".edgelist"
    directory_path = Path("./generated_files/"+g_type+"/"+str(weight))
    directory_path.mkdir(parents=True, exist_ok=True)
    nx.write_edgelist(G, directory_path / networkname)
    return networkname, str(directory_path / networkname)

#%%

def write_config(networkfile, parameterfile, networkfile_name, parameterfile_name, hyperparameter, weight, sd, initial_node):
    """
    Writes the configuration file for an experiment given a networkfile and parameterfile
    """
    naming_convention={"ER":"Homo","BA":"Hetero"}
    if parameterfile[0:2] == "Ho":
        param_type = "Homo"
    else:
        param_type = "Hetero"
    config_file_name = networkfile[0:2]+str(hyperparameter).replace(".", "")+"w"+str(weight).replace(".","")+param_type+str(sd).replace(".", "")+".txt"
    # Store config file in correct directory 
    destination_path = Path("./generated_files/"+naming_convention[networkfile[0:2]]+"/"+str(weight)+"/"+param_type)
    destination_path.mkdir(parents=True, exist_ok=True)
    # set path for writing config file
    config_file_path = destination_path / config_file_name
    # Use the default file and populate necessary values
    source_file="default.txt"
    with open(source_file, 'r') as source, config_file_path.open('w') as dest:
        # Read in the default file
        content = source.read()
        # Populate with hyperparameters
        modified_content = content.replace('NTWRKFILE', networkfile_name).replace("PARAMFILE", parameterfile_name).replace("NODE", str([initial_node]))
        # write to its own config file
        dest.write(modified_content)
    return str(config_file_path)
