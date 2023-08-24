#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:34:31 2022

@author: jeremynachison
"""

import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import random
from ast import literal_eval


def read_config(config_path):
    """ Read the configuration file of the experiment.
    
    Configuration files contain one key=value pair per line.
    The following is an example of the contents of a config file:
    _____________________________________________________________   
    
    NetworkFile=ntwrk.txt
    NumInitialInfected=4
    InitInfectedNodes=[0,1,7,4]
    
    InfectionMethod=Individual
    Center=1700000
    Steepness=-3/1700000
    
    Equation=TIVE
    ParameterData=ParameterDF.csv
    TragetInit=[6000,0,0,0]
    InfectedInit=[6000,1,1/6000,0]
    
    Duration=365
    SamplesPerDay=6
    InitialExposure=1e5
    Seed=None
    
    ____________________________________________________________
    
    Args: 
        config_path (str): file path to configuration file
        
    Returns: 
        dict (str : str) configuration key value pairs  
    """
    # read in config file as csv, convert to dictionary
    config_df = pd.read_csv(config_path, delimiter="=", names=["key", "val"])
    configs = dict(zip(config_df.key, config_df.val))
    # If no initial nodes given, set to None for a randomized experiment
    configs.setdefault("InitInfectedNodes", None)
    # If no seed is given,
    configs.setdefault("Seed", random.randint(0,9999))
    return configs

def load_Network(configs):
    G = nx.read_edgelist(configs["NetworkFile"], nodetype=int)
    return G

def load_InitialNodes(configs):
    nodes = literal_eval(configs["InitInfectedNodes"])
    num_nodes = 5 #int(configs["NumInitialInfected"])
    if len(nodes) != num_nodes:
        Warning(f"The number of initially infected nodes, {num_nodes}, does not match the list of nodes to seed with infection, which has a length of {len(nodes)}.") 
    return nodes


def inf_prob(x, center, steepness):
    """ Calculates the probability of infection given viral load and an 
    estimated maximum viral load usig a sigmoid function. The sigmoid function 
    achieves 0.5 probability when the viral load is equivalent to 2 individuals 
    with maximum viral load.
    
    Args:
        x (float): viral load
        viral_max (float): approximate maximum viral load that can be achieved by an individual"""
    # before steepness was -8/viral_max, center was viral_max
    return 1/(1+(np.e)**((steepness)*(x-center)))

def neighborhood(node, 
                 G, viral_load_ind, infected, center, steepness, init_expo, time, user_fct, paramDf, seed):
    """ Mechanism for virus transmission via the pooled neighborhood viral load. 
    The probability of a node becoming infected is calculated by the viral load 
    of all neighboring nodes.
    
    Args:
        node (int): id of node to be infected
        All other variables are defined and passed from host_ntwrk function
    "
    """
    # sum total viral load (at time t) across all infected neighbors
    virus_tot = 0
    # count infected neighbors
    inf_neighbors = 0
    for n in G.nodes[node]["Neighbors"]:
        # check if neighbor infected
        if (float(G.nodes[n]["state"][-1, viral_load_ind]) >= float(infected[-1,viral_load_ind])):
            # if neighbor infected, add viral load to virus_tot and add 1 to inf_neighbors
            virus_tot += G.nodes[n]["state"][-1, viral_load_ind]
            inf_neighbors += 1
    # probability of node getting infected is determined by the pooled viral load of its neighbors
    # transmission rate is average of all edge weights in neighborhood
    trans_rate = np.mean([G[node][neighbor]["weight"] for neighbor in G.nodes[node]["Neighbors"]])
    if inf_neighbors > 0:
        probability = trans_rate * inf_prob(virus_tot, center, steepness)
    else: 
        probability = 0
    if np.random.rand(seed=seed) < probability:
        # node is exposed to constant viral load, regardless of neighborhood vload
        exposure = (init_expo,) 
        inf_update = solve_ivp(user_fct, (time-0.1,time), G.nodes[node]["state"][-1], 
                               t_eval = np.linspace(time-0.1,time,2), args = (node,paramDf,exposure)).y
        G.nodes[node]["state"] = np.vstack((G.nodes[node]["state"],np.transpose(inf_update)[-1]))
        outcome = 1
    # node remains uninfected, add another uninfected row to diff eq array
    else:
        uninf_update = solve_ivp(user_fct,(time-0.1,time), G.nodes[node]["state"][-1], 
                                 t_eval = np.linspace(time-0.1,time,2)).y
        G.nodes[node]["state"] = np.vstack((G.nodes[node]["state"],np.transpose(uninf_update)[-1]))
        outcome = 0
    G.nodes[node]["Infected"] += [outcome]
    return outcome, probability


def individual(node, 
               G, viral_load_ind, infected, center, steepness, init_expo, time, user_fct, paramDf):
    """ Mechanism for virus transmission via viral load from each individual 
    neighbor. The probability of a node becoming infected is calculated for each 
    neighbor of the node based on the neighbor's viral load. 
    
    Args:
        node (int): id of node to be infected
        All other variables are defined and passed from host_ntwrk function
    "
    """
    probability_list = []
    outcome = 0
    for n in G.nodes[node]["Neighbors"]:
        # check if neighbor infected
        if (float(G.nodes[n]["state"][-1, viral_load_ind]) >= float(infected[-1,viral_load_ind])):
            # if neighbor infected, take its viral load to calculate infection probability
            neighbor_virus =  G.nodes[n]["state"][-1, viral_load_ind]
            trans_rate = G[node][n]["weight"]
            probability = trans_rate * inf_prob(neighbor_virus, center, steepness)
            probability_list += [(n,probability)]
        else:
            # otherwise infection probability is 0
            probability = 0
            probability_list += [(n,probability)]
        if np.random.rand() < probability:
            # node is exposed to constant viral load, regardless of neighbor vload
            exposure = init_expo 
            inf_update = solve_ivp(user_fct, (time-0.1,time) ,G.nodes[node]["state"][-1], 
                                   t_eval = np.linspace(time-0.1,time,2), args = (node, paramDf, exposure)).y
            G.nodes[node]["state"] = np.vstack((G.nodes[node]["state"],np.transpose(inf_update)[-1]))
            outcome += 1
            break
    # node remains uninfected, add another uninfected row to diff eq array
    if outcome == 0:
        uninf_update = solve_ivp(user_fct,(time-0.1,time), G.nodes[node]["state"][-1], 
                                 t_eval = np.linspace(time-0.1,time,2), args = (node , paramDf, 0)).y
        G.nodes[node]["state"] = np.vstack((G.nodes[node]["state"],np.transpose(uninf_update)[-1]))
    # outcome will be 1 automatically if one neighbor transmits virus, else 0
    G.nodes[node]["Infected"] += [outcome]
    return outcome, probability_list

def tive(time, x, node_id, paramDf, trigger=0):
    """ This defines the TIVE system used to model the sub-host dynamics of a virus.
    This function is formatted such that it can be evaluated by scipy's 
    solve_ivp function.
    ---------------------------------------------------------------------------
    Needed to be evaluated by scipy (in this order):
        
    time (float): passes the time to solve_ivp
    x (4x1 array): passes the T,I,V,E states to solve_ivp
    
    To define parameters specific to each host:
    
    ParamDF (pandas.DataFrame): 
    node_id (int): the id of a node in the host network. This is used to 
    extract the parameters specific to this host contained in the Parameter 
    Dataframe.
    
    
    """
    # Extract values of parameters from row of parameter dataframe by ID
    paramDict = paramDf.iloc[node_id].to_dict()
    delta, p, c, beta, d, dX, r, alpha, N = (paramDict["delta"], paramDict["p"], 
                                           paramDict["c"], paramDict["beta"], 
                                           paramDict["d"], paramDict["dX"], 
                                           paramDict["r"], paramDict["alpha"],
                                           paramDict["N"])
    # For scipy solver, extract individual compartments and create new array to store solutions
    t,i,v,e = x
    dx = np.zeros(4)
    if (v > 1e-5):
        dx[0] = - beta * v * t + alpha*t*(1-(t + i)/N) 
        dx[1] = (beta * v * t - delta * i - dX * i * e ) 
        dx[2] = p * i - c * v * e
        dx[3] = r*i - d*e
    else:
        dx[0] = alpha*t*(1-(t + i)/N)
        dx[1] = 0 
        dx[2] = 0 + trigger
        dx[3] = r*i - d*e
    return dx

def simulate(configuration_file):
    """ function that runs a simulation of a within host model across a network.
    Args:
        G (Networkx Graph): Network disease spreads over
        
        num_inf_init (int): Initial number of infected individuals
        
        T (int): Length of time (in days) simulation runs for
        
        EdgeWeightData (pandas.DataFrame): edgelist data containing the weight 
        to apply to each edge of the network
        
        user_fct (function): User defined within-host differential equations model
        
        paramDf (pandas.DataFrame): DataFrame containing parameter values for each node.
        Each row of the dataframe must contain all parameters necessary to 
        execute the user_fct, with the node id (an int) as the index
        
        viral_load_ind(int): The index of the output of user_fct corresponding to viral load
        
        uninf_init (list): Initial conditions for uninfected individuals
        
        inf_init (list): Initial conditions for infected individuals
        
        transmission (str): The mechanism for virus trnasmission, either 
        "neighborhood" or "individual".
        
        center (float): For transmission probability, the amount of virus at 
        which the transmission probability reaches 50%
        
        steepness (float): For transmission probability, the steepness of the sigmoid function
        
        spd (int): samples per day. How many times a day are the epidemic dynamics updated.
        
        init_expo (float): Amount of virus uninfected cells are exposed to when they become infected
        
        Init_Nodes (None or List): If a list of ints (correspondning to 
        individual nodes) is given, these nodes in the network will be seeded 
        with virus in the network. If None, random nodes in the network will be 
        selected for initial seeding
        
        Returns: Netorkx graph containing all nodes with updated within-host data
    """
    configs = read_config(configuration_file)
    
    G = load_Network(configs)
    num_inf_init = int(configs["NumInitialInfected"])
    Init_Nodes = load_InitialNodes(configs)
    
    transmission = configs["InfectionMethod"]
    center = eval(configs["Center"])
    steepness = eval(configs["Steepness"])
    
    user_fct = globals()[configs["Equation"]]
    viral_load_ind = int(configs["InfectiveIndex"])
    paramDf = pd.read_csv(configs["ParameterData"])
    uninf_init = eval(configs["TargetInit"]) 
    inf_init = eval(configs["InfectedInit"]) 
    
    T = int(configs["Duration"]) 
    spd = int(configs["SamplesPerDay"])
    init_expo = float(configs["InitialExposure"])
    seed = int(configs["Seed"])
    np.random.seed(seed)
    # Set dicitonary for transmission type, used to determine which function to 
    # use for infection probability
    trans_type = {"Individual":individual, "Neighborhood":neighborhood}
    # set initial conditions for uninfected nodes
    target = np.array([uninf_init])
    # Create uninfected nodes and data stored in each node
    for u in G.nodes():
        # create node attributes to collect data
        # state contains the state array, the solutions to the diff eqs
        G.nodes[u]["state"] = target
        # List of nodes with an edge connected to node u
        G.nodes[u]["Neighbors"] = [n for n in G.neighbors(u)]
        # Data for the probability of a node becoming infected at a time step and
        # The outcome for that probability (1, becomes infected, 0 remains uninfected)
        G.nodes[u]["Probability"] = {"Time":[],"Prob":[], "Outcome":[]}
        # Indicator for infection, 1 infected, 0 otherwise (at each time step)
        G.nodes[u]["Infected"] = [0]    
    # initialize infected nodes, either randomized or as passed by the Init_Nodes argument
    if Init_Nodes == None:
        init = random.sample(list(G.nodes()), num_inf_init)
    else:
        init = Init_Nodes
    # set initial conditions for infected nodes
    infected = np.array([inf_init])
    for u in init:
        # create infected node attributes
        G.nodes[u]["state"] = infected
        G.nodes[u]["Infected"][0] = 1
    for int_time in range(1,spd*T):
        # sample spd times per day
        time = int_time / spd
        for u in G.nodes:
            # update state of infected nodes first (check if node is infected)
            if (float(G.nodes[u]["state"][int_time - 1,viral_load_ind]) > float(infected[0,viral_load_ind])):
                # update age of infection and differential equation model
                G.nodes[u]["Infected"] += [1]
                update = solve_ivp(user_fct, (time-(1/spd),time), G.nodes[u]["state"][-1], 
                                   t_eval = np.linspace(time-(1/spd),time,2), args=(u, paramDf, 0)).y
                # add new row of solutions to differential equation array
                G.nodes[u]["state"] = np.vstack((G.nodes[u]["state"],np.transpose(update)[-1]))
        # check contagion after updating all infected nodes
        for u in G.nodes:
            # This is virus within a node
            node_virus = float(G.nodes[u]["state"][int_time-1,viral_load_ind])
            # This is the minimum amount of virus needed to be considered infected
            min_virus = float(infected[0,viral_load_ind])
            # update state of uninfected nodes second (check if node is uninfected)
            outcome = np.nan
            probability = np.nan
            if (node_virus <= min_virus):
               outcome, probability = trans_type[transmission](u, G, viral_load_ind, 
                                                               infected, center, steepness,
                                                               init_expo, time, user_fct, paramDf, seed)
            G.nodes[u]["Probability"]["Time"] += [time]
            G.nodes[u]["Probability"]["Prob"] += [probability]
            G.nodes[u]["Probability"]["Outcome"] += [outcome]
    return G 




