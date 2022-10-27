#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:24:41 2022

@author: kurup
"""

import numpy as np
from numpy.random import binomial as get_bin
from abc import ABC
from random import choice as select
from abstract import *
'''
##################################
# get_op() returns the output suggested by all 10 arms
no_of_arms=10 # Number of arms
arm_mean=np.linspace(0,1,no_of_arms+2)[1:no_of_arms+1] # have to be an array of length no_of_arms
def get_op(inp=no_of_arms):
    return get_bin(1,arm_mean,no_of_arms)
##################################

class algo(ABC):# Abstract class for the algorithm to be implemented
    def get_pred(self):
        pass
    def update(self):
        pass
'''
class Eps_greedy(algo):
    def __init__(self,eps):
        self.eps=eps
        self.pull_count=np.zeros(no_of_arms)
        self.prob_arm=np.zeros(no_of_arms)
        self.arms=list(range(no_of_arms))
    def get_pred(self,ch=1):
        if get_bin(1,self.eps)==1:
            pull_arm=select(self.arms)
            new_arm=1
        else:
            pull_arm=np.argmax(self.prob_arm)
            new_arm=0
        if ch==1:
            return pull_arm
        pred=get_op()[pull_arm]
        self.prob_arm[pull_arm]=(self.prob_arm[pull_arm]*self.pull_count[pull_arm]+pred)/(self.pull_count[pull_arm]+1)
        self.pull_count[pull_arm]+=1
        return pred,pull_arm,new_arm
        # Prediction <pred> by arm_no <pull arm> ; new_arm ==1 if explore,  0 if exploit
    def update(self,pull_arm,pred):
        self.prob_arm[pull_arm]=(self.prob_arm[pull_arm]*self.pull_count[pull_arm]+pred)/(self.pull_count[pull_arm]+1)
        self.pull_count[pull_arm]+=1