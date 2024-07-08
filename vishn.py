#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:08:13 2023

@author: jeremynachison
"""

# Required libraries
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
import scipy.sparse as sp
import pandas as pd
import dask.dataframe as dd
import equation_lib as eql
import warnings
import importlib

#%% Related to configuration file

def read_config(config_path):
    """ Read the configuration file of the experiment.
    
    Configuration files contain one key=value pair per line.
    The following is an example of the contents of a config file:
    _____________________________________________________________   
    
    NetworkFile=ntwrk.edgelist
    InitialHostStates=host_states.csv
    ParameterData=ParameterDF.csv
    
    Center=1700000
    Steepness=-3/1700000
    
    InitialExposure=1e5
    MinimalViralLoad=1e-5
    Equation=tive
    
    Duration=365
    SamplesPerDay=6
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
    # If no seed is given, use random seed
    configs.setdefault("Seed", np.random.randint(0,10000))
    return configs

def load_Network(configs):
    G = nx.read_edgelist(configs["NetworkFile"], nodetype=int)
    num_nodes = G.number_of_nodes()
    return G, num_nodes

def load_subhost_model(configs, num_nodes, minVload):
    equation_name = configs["Equation"].lower().replace(" ","").replace("_","")
    # load subhost parameters
    paramDf = pd.read_csv(configs["ParameterData"])
    # imports one of the built-in models
    if equation_name in {"tive", "tivelite", "tiv"}:
        equation_dict = {"tive": eql.Tive, "tivelite":eql.TiveLite, "tiv":eql.Tiv}
        subhost_model = equation_dict[equation_name](num_nodes, minVload, paramDf)
    # for custom models
    else:
        subhost_class_path = configs["Equation"]
        # if in an external file, path will have a '.' in it
        if "." in subhost_class_path:
            # get module and class name
            module_name, subhost_name = configs["Equation"].rsplit('.', 1)
            try:
                module = importlib.import_module(module_name)
                subhost_class = getattr(module, subhost_name)
                subhost_model = subhost_class(num_nodes, minVload, paramDf)
            except ModuleNotFoundError:
                raise ImportError(f"Module {module_name} not found.")
        # if defined in the same file
        else:
            if subhost_class_path in globals():
                subhost_class = globals()[subhost_class_path]
                subhost_model = subhost_class(num_nodes, minVload, paramDf)
            else:
                raise ImportError(f"Class {subhost_class_path} not found in this file. \n If trying to use a custom model defined in a class from another module, make sure configuration file has format module_name.class_name")
    return subhost_model

def load_initial_states(configs, data_brick, equation, num_nodes):
    initial_states = np.loadtxt(configs["InitialHostStates"], delimiter=',')
    try:
        data_brick[0,:,0:equation.compartments] = initial_states
    except:
        num_hosts = initial_states.shape[0]
        num_compartments = initial_states.shape[1]
        true_compartments = equation.compartments
        if num_hosts < num_nodes:
            raise ValueError(f"The InitialHostStates Data contains data for {num_hosts} hosts, but the NetworkFile has {num_nodes} hosts")
        if num_compartments != true_compartments:
            raise ValueError("The InitialHostStates Data contains data for a system of {num_compartments} equations, but the selected equation is a system of {true_compartments} equations")
        if num_hosts > num_nodes:
            warnings.warn(f"The InitialHostStates Data contains data for {num_hosts} hosts, but the NetworkFile has {num_nodes} hosts. \n Only the data for the first {num_nodes} hosts will be used.", category=UserWarning)
            initial_states = initial_states[0:num_nodes,:]
            data_brick[0,:,:] = initial_states
    

def initialize_simulation(configs):
    """
    Reads in all data from the config file to set all parameters and initial 
    conditions of the simulation.
    ___________________________________________________________________________
    Args:
        
    """
    G, num_nodes = load_Network(configs)
    center = eval(configs["Center"])
    steepness = eval(configs["Steepness"])
    init_expo = float(configs["InitialExposure"])
    min_vload = float(configs["MinimalViralLoad"])
    subhost_model = load_subhost_model(configs, num_nodes, min_vload)
    v_index = subhost_model.index
    T = int(configs["Duration"]) 
    spd = int(configs["SamplesPerDay"])
    seed = int(configs["Seed"])
    return G, num_nodes, center, steepness, subhost_model, v_index, init_expo, min_vload, T, spd, seed

#%% Related to infection spreading

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


def initial_infection_pass(adj_mat, num_nodes, data_brick, macro_time, spd, 
                          subhost_model, center, steepness):
    """
    Performs the initial pass of deterining which nodes are to be seeded with 
    virus in the next step of the simulation. However, each infectee may have a 
    tie for multiple infectors, the ties will be decided using the break_ties 
    function.
    ___________________________________________________________________________
    Args:
        adj_mat (scipy.sparse.csr_matrix): the adjacency matrix of the contact 
            network
        num_nodes (int): number of hosts in the simulation
        data_brick (np.array): the 3 dimensional numpy array that stores all 
            simulation data
        macro_time (int): the current (whole number) day of the simulation
        spd (int): the number of steps simulated per day (the # of steps per day)
        subhost_model (SubHostModel object): the object storing all information 
            about the specified subhost model, from the EquationLibrary
        center, steepness (float): the center and steepness of the sigmoid 
            function for calculating infections.
    Returns:
        infected_mat (scipy.sparse.csr_matrix): matrix where the i,jth non-zero 
            entry correspondes to the probability of node j being infected by host
            i and the infection was successful in transmitting.
        uninfected_mat (scipy.sparse.csr_matrix): Like infected_mat, except the 
            infection was not successful in transmitting
    """
    # get the maximum acorss the previous simulated steps
    state_mat = np.max(data_brick[(macro_time-1)*spd:macro_time*spd+1, :, :], axis=0)
    viral_vect = state_mat[:,subhost_model.index]
    # convert viral load to probability
    probs = inf_prob(viral_vect, center, steepness).reshape(num_nodes,1)
    # min viral load to be infectious
    uninf_vect = viral_vect <= subhost_model.min_vload
    inf_vect = (~uninf_vect).reshape(num_nodes,1)
    # matrix where the i, j th entry is probability of node j being infected by node i
    # second .multiply sets all rows for noninfected nodes to 0, third sets all columns for uninfected nodes to 0
    prob_mat = adj_mat.multiply(probs).multiply(inf_vect).multiply(uninf_vect)
    # Generate cutoff values for infection probs (as a matrix with identical columns for column-wise comparison)
    prob_cutoff = np.tile(np.random.rand(num_nodes), (num_nodes,1))
    infected_mat = prob_mat.multiply(sp.csr_matrix(prob_mat > prob_cutoff))
    uninfected_mat = prob_mat - infected_mat
    return infected_mat, uninfected_mat

def break_ties(infected_mat, num_nodes):
    """
    In the case of ties for multiple infectors per infectee, this function 
    breaks these ties such that each infectee only has one infector.
    ___________________________________________________________________________
    Args:
        infected_mat (scipy.sparse.csr_matrix): matrix where i,jth non-zero 
            entry correspondes to the probability of node j being infected by host i
        num_nodes (int): number of hosts in the simulation
    """
    # ties occur when infected_mat has more than one nonzero entry per column
    ties_mask = infected_mat.getnnz(axis=0) > 1
    tied_cols = infected_mat[:,ties_mask]
    # Take the tied columns and normalize their non-zero values to add to 1
    tied_sums = tied_cols.sum(axis=0)
    scaled_probs = tied_cols @ sp.diags(1/tied_sums)
    # Create new array to hold the values of the broken-ties
    tie_breakers = sp.lil_matrix(scaled_probs.shape)
    for col in range(scaled_probs.shape[1]):
        # break ties proprtionally to the probability of each nodes infectiousness
        winner = np.random.choice(range(num_nodes), p=scaled_probs[:,[col]].toarray().flatten())
        tie_breakers[[winner],[col]] = 1
    # convert to csr for faster multiplication adn get rid of implicit zeros
    tie_breakers = tie_breakers.tocsr()
    # convert to lil for bettter slicing
    tie_broken = scaled_probs.multiply(tie_breakers).tolil()
    infected_mat = infected_mat.tolil()
    # replace ties with the tie breakers
    infected_mat[:,ties_mask] = tie_broken
    

def update_and_store_outcomes(infected_mat, uninfected_mat, init_expo, 
                              subhost_model, num_nodes, data_brick, macro_time, spd):
    """
    Calculates subhost data for the next time step based on the infection spread 
    and stores all data (subhost data, infection probability, outcome, and 
    infector data) in the data_brick.
    ___________________________________________________________________________
    Args:
        infected_mat (scipy.sparse.csr_matrix): matrix where i,jth non-zero 
            entry correspondes to the probability of node j being infected by host i
        uninfected_mat (scipy.sparse.csr_matrix): Like infected_mat, except the 
            infection was not successful in transmitting
        init_expo (float): Amount of virus uninfected cells are exposed to 
            when infected
        subhost_model (SubHostModel object): the object storing all information 
            about the specified subhost model, from the EquationLibrary
        num_nodes (int): number of hosts in the simulation
        data_brick (np.array): the 3 dimensional numpy array that stores all 
            simulation data
        macro_time (int): the current (whole number) day of the simulation
        spd (int): the number of steps simulated per day (the # of steps per day)
    """
    # Store data in databrick
    outcome_vect = infected_mat.getnnz(axis=0)
    prob_vect = infected_mat.sum(axis=0)
    prob_vect = (prob_vect == 0) * uninfected_mat.max(axis=0).toarray() + prob_vect
    infector, infectee = infected_mat.nonzero()
    source_vect = np.full(num_nodes, np.nan)
    source_vect[infectee] = infector
    # insert in to data_brick
    data_brick[macro_time*spd,:,[subhost_model.compartments,subhost_model.compartments+1,subhost_model.compartments+2]] =  np.array([prob_vect.flatten(), outcome_vect, source_vect])
    # calculate new states
    trigger = init_expo*outcome_vect
    most_recent_states = data_brick[macro_time*spd,:,0:subhost_model.compartments]
    vector_result = solve_ivp(subhost_model.equation, (macro_time,macro_time+1), most_recent_states.T.flatten(), 
              t_eval = np.linspace(macro_time,macro_time+1,spd+1),
              args=(trigger,)).y
    for i in range(subhost_model.compartments):
        data_brick[spd*macro_time:spd*(macro_time+1)+1,:,i] = vector_result[i*num_nodes:(i+1)*num_nodes,:].T

#%%
# For output of data

# for numpy array
def convert_to_2d_array(data_brick, subhost_model, num_nodes, spd, T):   
    node_ids = np.tile(np.arange(num_nodes),spd*T+1).reshape(-1, 1)
    time_labels = np.repeat(np.linspace(0,T, T*spd+1, endpoint=True), num_nodes).reshape(-1,1)
    output_arr = np.hstack((time_labels,node_ids,data_brick.reshape(-1,subhost_model.compartments+3)))
    return output_arr
 
#%% Simulation   

def simulate(configuration_file, output="pandas"):
    """ function that runs a simulation of a within host model across a network.
    All of these arguments are specified in the configuration file
    Args:
        configuration_file (str): file path to the configuration file of the 
        experiment. 
        
        output (str): "graph" for the simulation to return the networkx graph 
        with all simulated data. "xarray" for the simulation to return an 
        xarray dataset with data arrays containing all node data.
        
        
        What is specified in the configuration file?
            
            G (Networkx Graph): The weighted contact network the disease spreads over
            
            T (int): Length of time (in days) simulation runs for
            
            subhost_model (SubHostModel object): sub-host differential 
                equations model from the equation library
                
            paramDf (pandas.DataFrame): DataFrame containing parameter values 
                for each node. Each row of the dataframe must contain all 
                parameters necessary to execute the specified subhost model 
                from the equation library, with the node id (an int) as the index
                
            center (float): For transmission probability, the amount of virus at 
                which the transmission probability reaches 50%
                
            steepness (float): For transmission probability, the steepness of 
                the sigmoid function
                
            spd (int): steps per day. How many times a day are the subhost 
                dynamics updated.
            
            init_expo (float): Amount of virus uninfected cells are exposed to 
                when infected
            
        Returns: if output="graph", Netorkx graph containing all nodes with updated within-host data.
                 if output="xarray", xarray dataset containing data arrays of all node data
    """
    ###### READ IN CONFIGURATION FILE######
    configs = read_config(configuration_file)
    (G, num_nodes, center, steepness, subhost_model, 
     v_index, init_expo, min_vload, T, spd, seed) = initialize_simulation(configs)
    np.random.seed(seed)
    # create array to store data, dimensions are (Time, node ID, data)
    # data in the format [compartments] + [infection probability, outcome, infector]
    data_brick = np.full(((T*spd)+1, num_nodes,subhost_model.compartments+3), np.nan)
    # load the initial states into data_brick
    load_initial_states(configs, data_brick, subhost_model, num_nodes)
    adj_mat = nx.to_scipy_sparse_array(G, weight='weight')
    
    ##### BEGIN SIMULATION #######
    # first do initial update of states to prepare for first infection spreading step
    vector_result = solve_ivp(subhost_model.equation, (0,1), 
                              data_brick[0,:,0:subhost_model.compartments].T.flatten(), 
                              t_eval = np.linspace(0,1,spd+1),
                              args=(np.zeros(num_nodes),)).y
    for i in range(subhost_model.compartments):
        data_brick[0:spd+1,:,i] = vector_result[i*num_nodes:(i+1)*num_nodes,:].T
    # Iteratively check for new infections and update states
    for macro_time in range(1,T):
        # first determine which nodes are to be infected in the next step of simulation
        infected_mat, uninfected_mat = initial_infection_pass(adj_mat, num_nodes, data_brick, 
                                                              macro_time, spd, subhost_model, 
                                                              center, steepness)
        # break any ties if multiple infected hosts transmit to the same infectee
        break_ties(infected_mat, num_nodes)
        # solve the subhost system for next time step, store all data in data_brick
        update_and_store_outcomes(infected_mat, uninfected_mat, init_expo, subhost_model, 
                                  num_nodes, data_brick, macro_time, spd)
    
    ##### RETURNING DATA BACK TO USER #####
    if output=="pandas":
        output_arr = convert_to_2d_array(data_brick, subhost_model, num_nodes, spd, T)
        col_names = ['time', 'node_id']+ subhost_model.compartment_names + ["probability", "outcome","infector"]
        output = pd.DataFrame(output_arr, columns=col_names)
    elif output=="numpy":
        output = convert_to_2d_array(data_brick, subhost_model, num_nodes, spd, T)
    elif output=="numpy3d":
        output = data_brick
    elif output == "dask":
        output_arr = convert_to_2d_array(data_brick, num_nodes, spd, T)
        col_names = ['time', 'node_id']+ subhost_model.compartment_names + ["probability", "outcome","infector"]
        chunk_size = np.where(num_nodes>=500000,
                              100000,
                              np.where(num_nodes>=50000, 25000, 1000))
        output = dd.from_array(output_arr, columns=col_names, chunksize=chunk_size)
    else:
        print("Invalid Ouput specified")
        return None
    return output

#%%
# Utility functions

def print_config_file(file):
    """
    This function prints the contents of a configuration file
    
    file (str): file path to config file
    """
    with open(file, "r") as config_file:
        contents = config_file.read()
    print(contents)

def write_config_file(filename, NetworkFile, InitialHostStates, ParameterData, 
                      Center, Steepness, InitialExposure, MinimalViralLoad,
                      Equation, Duration, SamplesPerDay, Seed=None):
    """
    Writes a simulation configuration file to the specified filepath configured
    with the supplied arguments.
    """
    with open(filename,"w") as file:
        file.write(f"NetworkFile={NetworkFile}\n")
        file.write(f"InitialHostStates={InitialHostStates}\n")
        file.write(f"ParameterData={ParameterData}\n\n")
        file.write(f"Center={Center}\n")
        file.write(f"Steepness={Steepness}\n\n")
        file.write(f"InitialExposure={InitialExposure}\n")
        file.write(f"MinimalViralLoad={MinimalViralLoad}\n")
        file.write(f"Equation={Equation}\n\n")
        file.write(f"Duration={Duration}\n")
        file.write(f"SamplesPerDay={SamplesPerDay}\n")
        if Seed != None:
            file.write(f"Seed={Seed}\n")
    
    
    
    
    

