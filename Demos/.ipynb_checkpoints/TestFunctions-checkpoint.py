#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:45:59 2023

@author: jeremynachison
"""

import sys
sys.path.append('../')
import hostsim as hs
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating path graphs
G0 = nx.path_graph(3)
G1 = nx.path_graph(3)
for edge in G0.edges():
    G0[edge[0]][edge[1]]['weight'] = 0
    G1[edge[0]][edge[1]]['weight'] = 1
nx.write_edgelist(G0, "Path_0Weight.edgelist")
nx.write_edgelist(G1, "Path_1Weight.edgelist")

param_dict = {"delta":0.8,"p":4.5e2,"c":5e-2,"beta":2e-5,"d":0.065,
              "dX":1e-4,"r":0.01,"alpha":1.2,"N":6e4}
singlerow = pd.DataFrame(param_dict, index=[0])
pathDf = pd.concat([singlerow]*3, ignore_index=True)
pathDf.to_csv("path_params.csv", index=False)     


# Creating the barbell graph
B = nx.barbell_graph(3, 1)
for edge in B.edges():
    if (edge[0] or edge[1]) <= 3:
        B[edge[0]][edge[1]]['weight'] = 1
    else:
        B[edge[0]][edge[1]]['weight'] = 0
nx.write_edgelist(B, "barbell.edgelist")

barbellDf = pd.concat([singlerow]*7, ignore_index=True)
barbellDf.to_csv("barbell_params.csv", index=False)     

# Creating star graph
S = nx.star_graph(5)
nx.draw_networkx(S,pos=nx.spring_layout(S), with_labels = True)
for edge in S.edges():
    S[edge[0]][edge[1]]['weight'] = 1
nx.write_edgelist(S, "star.edgelist")
starDf = pd.concat([singlerow]*6, ignore_index=True)
starDf.to_csv("star_params.csv", index=False)     

# Creating barabasi-albert graphs

# low connectivity
BA_low = nx.barabasi_albert_graph(15,1, seed=3)
nx.draw_networkx(BA_low,pos=nx.spring_layout(BA_low), with_labels = True)

# high connectivity
BA_high =  nx.barabasi_albert_graph(15,3, seed=3)
nx.draw_networkx(BA_high,pos=nx.spring_layout(BA_high), with_labels = True)

for edge in BA_low.edges():
    BA_low[edge[0]][edge[1]]['weight'] = 1

for edge in BA_high.edges():
    BA_high[edge[0]][edge[1]]['weight'] = 1

nx.write_edgelist(BA_low, "Barabasi-AlbertLOW.edgelist")
nx.write_edgelist(BA_high, "Barabasi-AlbertHIGH.edgelist")
BA_Df = pd.concat([singlerow]*15, ignore_index=True)
BA_Df.to_csv("barabasi_albert_params.csv", index=False)     

def make_4plots(result, T, spd, colors, alpha=1, xlim=None):
    if xlim==None:
        xlim=T
    plt.figure(figsize=(12,10))
    t1 = np.linspace(0,T,T*spd)
    plt.style.use('fivethirtyeight')
    plt.subplot(4, 1, 1)
    for i in result.nodes:
        result.nodes[i]["color"] = colors[i]
        plt.plot(t1, result.nodes[i]["state"][:,2], color=result.nodes[i]["color"], alpha = alpha)
    plt.ylabel("Viral Load")
    ax = plt.gca()
    ax.set_xlim([0, xlim])
    plt.subplot(4, 1, 2)
    for i in result.nodes:
        plt.plot(t1, result.nodes[i]["state"][:,1], color=result.nodes[i]["color"], alpha = alpha)
    plt.ylabel("# of Infected Cells")
    ax = plt.gca()
    ax.set_xlim([0, xlim])
    plt.subplot(4,1,3)
    for i in result.nodes:
        plt.plot(t1, result.nodes[i]["state"][:,0], color=result.nodes[i]["color"], alpha = alpha)
    plt.ylabel("# of Target Cells")
    ax =  plt.gca()
    ax.set_xlim([0, xlim])
    plt.subplot(4,1,4)
    for i in result.nodes:
        plt.plot(t1, result.nodes[i]["state"][:,3], color=result.nodes[i]["color"], alpha = alpha)
    plt.ylabel("Immune Response")
    ax = plt.gca()
    ax.set_xlim([0, xlim])
    fig = plt.gcf()
    fig.suptitle("Diagnostic Plot of TIVE Model", fontsize=24, y=0.92)
    plt.show()
    return

def getPopulationLevel(result, j, T, spd):
    pop_tot = np.zeros((T*spd,))
    for i in result.nodes:
        node_lvl = result.nodes[i]["state"][:,j]
        pop_tot += node_lvl
    return pop_tot

def population_vload_plot(result, T, spd, color, alpha=1, xlim=None):
    vload_tot = getPopulationLevel(result, 2, T, spd)
    plt.style.use('fivethirtyeight')
    time = np.linspace(0,T,T*spd)
    plt.figure(figsize=(10,6))
    plt.plot(time, vload_tot, color=color, alpha = 0.8)
    plt.ylabel("Viral Load")
    plt.xlabel("Days")
    plt.title("Population Viral Load")
    ax =  plt.gca()
    ax.set_xlim([0,xlim])
    plt.show()
    return 

def simulate_and_plot(config, T, spd, colors, alpha=1, xlim=None):
    result = hs.simulate(config)
    make_4plots(result, T, spd, colors, alpha, xlim)

path_colors = ["orangered", "cornflowerblue", "yellowgreen"] 
simulate_and_plot("Path_0Weight.txt", 60, 10, path_colors, 0.8)

def ProbabilityZero():    
    result = hs.simulate("Path_0Weight.txt")
    make_4plots(result, 60, 10, ["orangered", "cornflowerblue", "yellowgreen"], 0.8)
    population_vload_plot(result, 60, 10, "orangered", alpha=0.8, xlim=10)
    return result

def ProbabilityOne():
     result = hs.simulate("Path_1Weight.txt")
     make_4plots(result, 60, 10, ["orangered", "cornflowerblue", "yellowgreen"], 0.8)
     population_vload_plot(result, 60, 10, "darkorchid", alpha=0.8, xlim=10)
     return result

def barbell():
    result = hs.simulate("barbell.txt")
    colors = ["cornflowerblue"]*4 + ["orangered"]*3
    make_4plots(result, 60, 10, colors, 0.8)
    population_vload_plot(result, 60, 10, "darkorchid", alpha=0.8, xlim=10)
    return result

def star():
    result = hs.simulate("star.txt")
    colors = ["cornflowerblue"] + ["orangered"]*5
    make_4plots(result, 60, 10, colors, 0.8)
    return
    
def low_barabasi_albert():
    result = hs.simulate("BarabasiAlbert_low.txt")
    colors = ["limegreen"]*15
    make_4plots(result, 60, 10, colors, 0.5)
    return

def high_barabasi_albert():
    result = hs.simulate("BarabasiAlbert_high.txt")
    colors = ["darkgreen"]*15
    make_4plots(result, 60, 10, colors, 0.5)
    return

