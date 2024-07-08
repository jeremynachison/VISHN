#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Required libraries
import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import random
from ast import literal_eval
import xarray as xr

#%% Related to configuration file

def read_config(config_path):
    """ Read the configuration file of the experiment.
    
    Configuration files contain one key=value pair per line.
    The following is an example of the contents of a config file:
    _____________________________________________________________   
    
    NetworkFile=ntwrk.edgelist
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

def load_equation(configs, equation):
    if equation != None:
        return equation
    else: 
        return globals()[configs["Equation"]]

def load_InitialNodes(configs):
    if configs["InitInfectedNodes"] == None:
        return None
    else:
        nodes = literal_eval(configs["InitInfectedNodes"])
        num_nodes = int(configs["NumInitialInfected"])
    if len(nodes) != num_nodes:
        Warning(f"The number of initially infected nodes, {num_nodes}, does not match the list of nodes to seed with infection, which has a length of {len(nodes)}.") 
    return nodes

def print_ConfigFile(file):
    """
    This function prints the contents of a configuration file
    
    file (str): file path to config file
    """
    with open(file, "r") as config_file:
        contents = config_file.read()
    print(contents)
    config_file.close()
    return None
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

def neighborhood(node, 
                 G, viral_load_ind, infected, center, steepness, init_expo, 
                 time, user_fct, paramDf, edgelist_data, seed):
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
               G, viral_load_ind, infected, center, steepness, init_expo, time, 
               user_fct, paramDf, edgelist_data, seed):
    """ Mechanism for virus transmission via viral load from each individual 
    neighbor. The probability of a node becoming infected is calculated for each 
    neighbor of the node based on the neighbor's viral load. 
    
    Args:
        node (int): id of node to be infected
        All other variables are defined and passed from host_ntwrk function
    "
    """
    probability_list = []
    node_list = []
    outcome = 0
    for n in G.nodes[node]["Neighbors"]:
        # check if neighbor infected
        if (float(G.nodes[n]["state"][-1, viral_load_ind]) >= float(infected[-1,viral_load_ind])):
            # if neighbor infected, take its viral load to calculate infection probability
            neighbor_virus =  G.nodes[n]["state"][-1, viral_load_ind]
            trans_rate = G[node][n]["weight"]
            probability = trans_rate * inf_prob(neighbor_virus, center, steepness)
            probability_list += [probability]
            node_list += [n]
        else:
            # otherwise infection probability is 0
            probability = 0
            probability_list += [probability]
            node_list += [n]
        if np.random.rand() < probability:
            # node is exposed to constant viral load, regardless of vload
            exposure = init_expo 
            inf_update = solve_ivp(user_fct, (time-0.1,time) ,G.nodes[node]["state"][-1], 
                                   t_eval = np.linspace(time-0.1,time,2), args = (node, paramDf, exposure)).y
            G.nodes[node]["state"] = np.vstack((G.nodes[node]["state"],np.transpose(inf_update)[-1]))
            edgelist_data = add_to_edgelist(n, node, time, edgelist_data)
            infProb = probability_list[-1]
            infNode = node_list[-1]
            outcome += 1
            break
    # node remains uninfected, add another uninfected row to diff eq array
    if outcome == 0:
        uninf_update = solve_ivp(user_fct,(time-0.1,time), G.nodes[node]["state"][-1], 
                                 t_eval = np.linspace(time-0.1,time,2), args = (node , paramDf, 0)).y
        G.nodes[node]["state"] = np.vstack((G.nodes[node]["state"],np.transpose(uninf_update)[-1]))
        maxIndex = np.argmax(probability)
        infProb = probability_list[maxIndex]
        infNode = node_list[maxIndex]
    # outcome will be 1 automatically if one neighbor transmits virus, else 0
    G.nodes[node]["Infected"] += [outcome]
    return outcome, infProb, infNode, edgelist_data
#%% Built-in subhost models

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

#%% Related to ouput

def add_to_edgelist(source, target, time, edgelist_data):
    """
    Stores data of the infection spreading through the network as an edgelist
    
    Args:
        source (int): id of source node (the infectious node)
        target (int): id of target node (the node the infection has spread to)
        edgelist_data (pd.DataFrame): Running edgelist data of infection
    """
    row_dict = {"source":[source],"target":[target], "time":[time]}
    new_row = pd.DataFrame(row_dict, index=[0])
    edgelist_data = pd.concat([edgelist_data , new_row], ignore_index = True)
    return edgelist_data

#%%
# For xarray

def node2dataArray(node):
    nodeStatus=np.array(node["Infected"]).reshape((-1,1))
    nodeState=node["state"]
    nodeProbOut= pd.DataFrame(node["Probability"]).iloc[:,[1,2]].to_numpy(na_value=np.nan)
    time=node["Probability"]["Time"]
    tempArr = np.append(nodeProbOut,nodeStatus,axis=1)
    nodeArr = np.append(nodeState, tempArr,axis=1)
    datArray = xr.DataArray(
        nodeArr,
        dims=("time","data"),
        coords={
            "time":time,
            "data":["T","I","V","E", "probability", "outcome","status"]
            },
        attrs={}
        )
    return datArray

def graph2dataArray(edgelist, attribute):
    datArray = xr.DataArray(
        edgelist,
        dims=("index","node"),
        coords={
            "index" : [i for i in range(len(edgelist))],
            "data" : ["target","source",attribute]
            }
        )
    return datArray


def makeVarDict(G):
    varDict={}
    for i in G.nodes():
        datArr = node2dataArray(G.nodes[i])
        varDict[str(i)] = datArr
    return varDict

#%%

# for pandas dataframe
def graph2dataframe(G):
    allFrames = []
    for i in G.nodes():
        node = G.nodes[i]
        state = pd.DataFrame(data=node["state"], columns=["T", "I", "V", "E"])
        nodeDF = pd.DataFrame(node["Probability"])
        nodeDF = pd.concat([state,nodeDF],axis=1)
        nodeDF["status"] = node["Infected"]
        nodeDF.insert(0,"id",[i]*len(nodeDF))
        allFrames += [nodeDF]
    full_dat = pd.concat(allFrames, ignore_index=True)
    return full_dat

#%% Simulation

def simulate(configuration_file, custom_equation = None, output="graph"):
    """ function that runs a simulation of a within host model across a network.
    All of these arguments are specified in the configuration file
    Args:
        configuration_file (str): file path to the configuration file of the 
        experiment. 
        
        custom_equation (None or function): When using a differential equation 
        model built into hostsim,this argument is None. If using a custom model,
        this is the name of the function as it is defined in the same python 
        file of the experiment.
        
        output (str): "graph" for the simulation to return the networkx graph 
        with all simulated data. "xarray" for the simulation to return an 
        xarray dataset with data arrays containing all node data.
        
        
        What is specified in the configuration file?
            
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
            
        Returns: if output="graph", Netorkx graph containing all nodes with updated within-host data.
                 if output="xarray", xarray dataset containing data arrays of all node data
    """
    configs = read_config(configuration_file)
    
    G = load_Network(configs)
    num_inf_init = int(configs["NumInitialInfected"])
    Init_Nodes = load_InitialNodes(configs)
    
    transmission = configs["InfectionMethod"]
    center = eval(configs["Center"])
    steepness = eval(configs["Steepness"])
    
    user_fct = load_equation(configs, custom_equation)
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
    edgelist_df = pd.DataFrame({"source":[],"target":[], "time":[]})
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
        G.nodes[u]["Probability"] = {"Time":[0],"Prob":[np.nan], "Node":[np.nan], "Outcome":[np.nan]}
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
               outcome, probability, srcNode, edgelist_df = trans_type[transmission](u, G, viral_load_ind, 
                                                               infected, center, steepness,
                                                               init_expo, time, user_fct, paramDf, edgelist_df, seed)
            G.nodes[u]["Probability"]["Time"] += [time]
            G.nodes[u]["Probability"]["Prob"] += [probability]
            G.nodes[u]["Probability"]["Node"] += [srcNode]
            G.nodes[u]["Probability"]["Outcome"] += [outcome]
    G.graph["transmission_edgelist"] = edgelist_df
    # return output in format specified by user
    if output=="graph":
        output = G
    elif output=="xarray":
        output = xr.Dataset(
            data_vars = makeVarDict(G)
            )
    elif output=="pandas":
        output = graph2dataframe(G)
    return output

#%%

# utility functions

def print_ConfigFile(file):
    """
    This function prints the contents of a configuration file
    
    file (str): file path to config file
    """
    with open(file, "r") as config_file:
        contents = config_file.read()
    print(contents)
    config_file.close()
    return None

def write_config_file(filename, NetworkFile, NumInitialInfected, InitInfectedNodes,
                      InfectionMethod, Center, Steepness, InitialExposure,
                      Equation,InfectiveIndex, ParameterData, TargetInit, InfectedInit,
                      Duration, SamplesPerDay, Seed=None):
    """
    Writes a simulation configuration file to the specified filepath configured
    with the supplied arguments.
    """
    with open(filename,"w") as file:
        file.write(f"NetworkFile={NetworkFile}\n")
        file.write(f"NumInitialInfected={NumInitialInfected}\n")
        file.write(f"InitInfectedNodes={InitInfectedNodes}\n\n")
        
        file.write(f"InfectionMethod={InfectionMethod}\n")
        file.write(f"Center={Center}\n")
        file.write(f"Steepness={Steepness}\n")
        file.write(f"InitialExposure={InitialExposure}\n\n")
        
        file.write(f"Equation={Equation}\n")
        file.write(f"InfectiveIndex={InfectiveIndex}\n")
        file.write(f"ParameterData={ParameterData}\n")
        file.write(f"TargetInit={TargetInit}\n")
        file.write(f"InfectedInit={InfectedInit}\n\n")
        
        file.write(f"Duration={Duration}\n")
        file.write(f"SamplesPerDay={SamplesPerDay}\n")
        if Seed != None:
            file.write(f"Seed={Seed}\n")
    
    




