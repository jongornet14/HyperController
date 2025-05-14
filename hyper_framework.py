import torch
import numpy as np
import hyper_optimizers
import itertools

from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from ray.tune.utils.util import flatten_dict, unflatten_dict

from pb2 import PB2
from gp_ucb import GP_UCB

class Scheduler:
    
    def __init__(self,hyperparameter_bounds,t_ready,method,tEND):
        
        self.window_size           = 1
        self.hyperparameter_bounds = hyperparameter_bounds
        self.hyperparameters       = {}
        self.records               = {}
        self.agent_dict            = {}
        self.config                = {}
        self.last_exploration_time = 0
    
        self.number_of_hyperparameters = 10

        self.t_ready = t_ready
        self.method  = method
        self.tEND    = tEND
        self.current = None
        
        self._hyperparam_bounds_flat = flatten_dict(
            self.hyperparameter_bounds, prevent_delimiter=True
        )
        
        self.reset()
            
    def reset(self):
        
        for key in self.hyperparameter_bounds.keys():
            
            self.hyperparameters[key] = np.linspace(self.hyperparameter_bounds[key][0],self.hyperparameter_bounds[key][1],self.number_of_hyperparameters)
                
            if self.method == 'UCB':
                self.agent_dict[key] = hyper_optimizers.UCB(0,self.number_of_hyperparameters)
                self.records[key]    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
                self.records[key].record(name=self.method)
            elif self.method == 'Sliding_UCB':
                self.agent_dict[key] = hyper_optimizers.UCB(10,self.number_of_hyperparameters)
                self.records[key]    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
                self.records[key].record(name=self.method)
            elif self.method == 'HyperController':
                self.agent_dict[key] = hyper_optimizers.HyperController(self.window_size,self.number_of_hyperparameters,self.number_of_hyperparameters)
                self.records[key]    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
                self.records[key].record(name=self.method)
            elif self.method == 'UBSS':
                self.agent_dict[key] = hyper_optimizers.UBSS(self.window_size,self.number_of_hyperparameters,self.number_of_hyperparameters)
                self.records[key]    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
                self.records[key].record(name=self.method)
            elif self.method == 'Kalman_UCB':
                self.agent_dict[key] = hyper_optimizers.Feedback_Kalman(self.number_of_hyperparameters,self.hyperparameters[key],'Kalman_UCB')
                self.records[key]    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
                self.records[key].record(name=self.method)    
            elif self.method == 'Kalman_Observer':
                self.agent_dict[key] = hyper_optimizers.Feedback_Kalman(self.number_of_hyperparameters,self.hyperparameters[key],'Kalman_Observer')
                self.records[key]    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
                self.records[key].record(name=self.method)    
            elif self.method == 'Kalman_Greedy':
                self.agent_dict[key] = hyper_optimizers.Feedback_Kalman(self.number_of_hyperparameters,self.hyperparameters[key],'Kalman_Greedy')
                self.records[key]    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
                self.records[key].record(name=self.method)    
            
        if self.method == 'PB2':
            self.agent  = PB2(self.hyperparameter_bounds)
            self.config = self.agent.config
        elif self.method == 'GP-UCB':
            self.agent  = GP_UCB(self.hyperparameter_bounds)
            self.config = self.agent.config
        elif self.method == 'HyperBand':
            self.agent_dict = hyper_optimizers.HyperBand(self.tEND,self.hyperparameter_bounds)
            self.records    = hyper_optimizers.Recorder(1,self.number_of_hyperparameters)
            self.records.record(name=self.method)    
        
    def step(self,data,hyperparam_data):

        iteration = hyperparam_data['iteration'].iloc[-1]
        if len(np.array(data["reward"])) > 2:
            R  = data['Reward'][-1]
            R0 = data['Reward'][-2]
        else:
            R  = data['Reward'][-1]
            R0 = data['Reward'][-1]

        if self.method == 'GP-UCB' or self.method == 'PB2':
            self.agent.update_data(R)
            if iteration % self.t_ready == 0:
                self.config = self.agent.make_choice()
        elif self.method in ['LBSS','UBSS','Kalman_UCB','Kalman_Observer','Kalman_Greedy','UCB','HyperController']:
            for key in self.hyperparameter_bounds.keys():
                action = hyper_optimizers.interact(self.agent_dict[key],self.records[key],R-R0,R-R0,self.t_ready,iteration)
                self.config[key] = self.hyperparameters[key][action]
        elif self.method in ['HyperBand']:
            self.config = hyper_optimizers.interact(self.agent_dict,self.records,R-R0,R-R0,self.t_ready,iteration)
        elif self.method in ['Random']:
            if iteration % self.t_ready == 0:
                for key in self.hyperparameter_bounds.keys():
                    self.config[key] = ((self.hyperparameter_bounds[key][1]-self.hyperparameter_bounds[key][0])*np.random.rand()+self.hyperparameter_bounds[key][0])
        elif self.method in ['Random_Start']:
            if iteration == 0:
                for key in self.hyperparameter_bounds.keys():
                    self.config[key] = ((self.hyperparameter_bounds[key][1]-self.hyperparameter_bounds[key][0])*np.random.rand()+self.hyperparameter_bounds[key][0])

        return self.config
    