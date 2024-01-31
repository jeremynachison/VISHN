#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##### To run the experiment in serial, this script can be used. However, ######
##### the amount of data generated is large (about 50GB) and the script may ###
##### take very long (over 12 hours) to complete running all 540 hyperparameter
##### combinations in serial on a standard laptop/desktop. It is highly #######
##### recommended the experiment is ran using th generateExperiments.py file ##
##### on a computing cluster.


import sys
sys.path.append('../')
import vishn as vn
import ExperimentHelpers as exh
# Other packages used
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
import xarray as xr
from joblib import Parallel, delayed

#%%

np.random.seed(0)
seed = 0
# Number of nodes
N_nodes = 500
# Number of trials
N_trials = 10

#%%

# Graph types
g_types = ["Hetero","Homo"]
# Graph Hyperparameters
hetero_hyperparam = [1.00,2.00,3.00]
# Erdos-renyi hyperparameters chosen such that both have average node degree
homo_hyperparam = exh.gen_ER_probs(hetero_hyperparam, N_nodes)

# set parameters in a dictionary for access
hyperparam_dict = {"Hetero":hetero_hyperparam, 
                   "Homo":homo_hyperparam}
# edge weights to test
edge_weights = [0.03,0.04,0.05]

# These will be the average parameters for each host
param_dict = {"delta":0.8,"p":4.5e2,"c":5e-2,"beta":2e-5,"d":0.065,
              "dX":1e-4,"r":0.01,"alpha":1.2,"N":6e4}
param_sd = [None,0.05,0.1]

#%%

def run_combination(graph, graphHyperparam, edgeWeight, popSd):
    # Generate the graph based on parameters, add weights and write to edgelist
    G, init_node = exh.gen_graph_wparams(graph, graphHyperparam, N_nodes, seed)
    ntwrk_name, ntwrk_path = exh.weight_and_write(G, graph, graphHyperparam, edgeWeight, popSd)
    # Generate node parameter file and configuration file
    param_name, param_path = exh.gen_param_file(N_nodes, param_dict, graph, graphHyperparam, edgeWeight, sd = popSd, seed = seed)
    filepath = exh.write_config(ntwrk_name,param_name,
                            ntwrk_path, param_path,
                            graphHyperparam, edgeWeight, popSd, init_node)
    # run N_trials of the simulation
    results = Parallel(n_jobs=N_trials)(delayed(vn.simulate)(filepath,output="pandas")for i in range(N_trials))
    # Combine all 10 trials into one dataframe
    full_df = pd.concat([df.assign(trial=index) for index, df in enumerate(results)], ignore_index=True)
    # add columns keeping track of hyperparams for the trials
    full_df['graph'], full_df['hyperparam'],full_df['weight'], full_df["sd"] = graph, graphHyperparam, edgeWeight, popSd
    return full_df

#%%

for g_type in g_types:
    for hyperparameter in hyperparam_dict[g_type]:
        for weight in edge_weights:
            for stand_dev in param_sd:
                result = run_combination(g_type, hyperparameter, weight, stand_dev)
                result.to_csv(g_type+str(hyperparameter).replace(".","p")+"w"+str(weight).replace(".","p")+str(stand_dev).replace(".","p")+"N"+str(N_nodes)+".csv", index=False)




