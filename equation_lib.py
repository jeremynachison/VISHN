#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:23:14 2024

@author: jeremynachison
"""
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd

class SubHostModel(ABC):
    """
    Abstract class outlining the required format for a sub-host model to be 
    used in VISHN. 
    """
    def __init__(self, num_nodes, minVload, paramDf):
        """
        num_nodes (int): the number of hosts (nodes) in the contact network of 
            the simulation.
        
        index (int): the index of the viral load comprtment, starting from 0, 
            to be used to calculate infection probabilities.
        
        compartments (int): the number of equations used to define the system.
            e.g. the TIVE model has 4, the TIV has 3
        
        compartment_names (list): the label of each compartment of the sub-host 
            model, ordered as they are in the equation function, to be used when 
            returning the data as a dataframe.
        
        min_vload (float): the minimum viral load required to be considered 
            infectious
        """
        self.num_nodes = num_nodes
        self.index = 0
        self.compartments = 0
        self.compartment_names = []
        self.min_vload = minVload
    @abstractmethod
    def equation(self, time, x, trigger):
        """
        the function defining the sub-host system of differential 
        equations. The system must be writtien in the format such that it can 
        be evaluated by scipy's solve_ivp. Arguments must be in the order:
            
        time (float): the current time step in the solver
        
        x (np.array): an array with shape (num_nodes * compartments, 1) holding 
            all states of the system for all nodes. This array must be formatted 
            such that the first num_nodes entries correspond to the first 
            compartment for all nodes, the next num_nodes entries after these 
            correspond to the second, etc.
            
        trigger (np.array): an array of shape (num_nodes,1) where the ith entry 
        holds the viral load host i was infected with.
        """
        pass
        
        

class Tive (SubHostModel):
    """
    The TIVE system is capable of capturing waning immunity, multiple infection 
    cycles, and viral load dependent transmission. VISHN was initially designed 
    around using the TIVE sub-host model, so it is highly recommended to use 
    this system for simulations.
    """
    def __init__(self, num_nodes, minVload, paramDf):
        self.num_nodes = num_nodes
        self.index = 2
        self.compartments = 4
        self.compartment_names =["T","I","V","E"]
        self.min_vload = minVload
        (self.delta, self.p, self.c, self.beta, self.d, 
         self.dX, self.r, self.alpha, self.N) = (
            paramDf[col].values for col in ["delta", "p", "c", "beta", "d", 
                                            "dX", "r","alpha","N"])
    def equation(self, time, x, trigger):
        t = x[0:self.num_nodes]
        i = x[self.num_nodes:2*self.num_nodes]
        v = x[2*self.num_nodes:3*self.num_nodes]
        e = x[3*self.num_nodes:]
        dx = np.zeros(self.num_nodes*4)
        dx[0:self.num_nodes] = np.where(v <= self.min_vload,  
                                        self.alpha*t*(1-(t + i)/self.N),  
                                        - self.beta * v * t + self.alpha*t*(1-(t + i)/self.N) )
        dx[self.num_nodes:2*self.num_nodes] = np.where(v <= self.min_vload, 
                                                       0,
                                                       self.beta * v * t - self.delta * i - self.dX * i * e )
        dx[2*self.num_nodes:3*self.num_nodes] = np.where( v <= self.min_vload, 
                                                         trigger,  
                                                         self.p * i - self.c * v * e)
        dx[3*self.num_nodes:] = self.r*i - self.d*e
        return dx
    
class TiveLite(SubHostModel):
    """
    TiveLite is a stripped down version of the Tive system that does not 
    capture multiple infection cycles. TiveLite captures viral load depenedent 
    transmission (like the Tiv system) with the addition of modelling 
    immunity dynamics.
    """
    def __init__(self, num_nodes, minVload, paramDf):
        self.num_nodes = num_nodes
        self.index = 3
        self.compartments = 4
        self.min_vload = minVload
        self.compartment_names =["T","I","V","E"]
        (self.delta, self.p, self.c, self.beta, self.d, 
         self.dX, self.r) = (
            paramDf[col].values for col in ["delta", "p", "c", "beta", "d", 
                                            "dX", "r"])
             
    def equation(self, time, x, trigger):
        t = x[0:self.num_nodes]
        i = x[self.num_nodes:2*self.num_nodes]
        v = x[2*self.num_nodes:3*self.num_nodes]
        e = x[3*self.num_nodes:]
        dx = np.zeros(self.num_nodes*4)
        dx = np.zeros(self.num_nodes*4)
        dx[0:self.num_nodes] = -self.beta*v*t
        dx[self.num_nodes:2*self.num_nodes] = self.beta*v*t - self.delta*i - self.dX*i*e
        dx[2*self.num_nodes:3*self.num_nodes] = self.p*i - self.c*v*e + trigger
        dx[3*self.num_nodes:] = self.r*i-self.d*e
        return dx
    
class Tiv(SubHostModel):
    """
    The Tiv system is a stripped down version of the Tive system that is only 
    capable of modelling a single infection cycle for each host and does not 
    model any immunity. The Tiv system should be used when viral load dependent 
    transmission is the only sub-host factor of interest.
    """
    def __init__(self, num_nodes, minVload, paramDf):
        self.num_nodes = num_nodes
        self.index = 2
        self.compartments = 3
        self.min_vload = minVload
        self.compartment_names =["T","I","V"]
        (self.delta, self.p, self.c, self.beta) = (
            paramDf[col].values for col in ["delta", "p", "c", "beta"])
        
    def equation(self, time, x, trigger):
        t = x[0:self.num_nodes]
        i = x[self.num_nodes:2*self.num_nodes]
        v = x[2*self.num_nodes:]
        dx = np.zeros(self.num_nodes*3)
        dx[0:self.num_nodes] = -self.beta*v*t
        dx[self.num_nodes:2*self.num_nodes] = self.beta*v*t - self.delta*i
        dx[2*self.num_nodes:] = self.p*i - self.c*v + trigger
        return dx
