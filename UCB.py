#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:49:21 2022

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

class UCB(algo):
    def __init__(self):
        self.pull_count=np.zeros(no_of_arms)
        self.prob_arm=np.zeros(no_of_arms)
        self.arms=list(range(no_of_arms))
        self.counter=0
    def get_pred(self,ch=1):
        if self.counter<no_of_arms:
            pull_arm=self.counter
        else:
            temp=np.repeat((2*np.log(self.counter)),no_of_arms)/self.pull_count
            pull_arm=np.argmax(self.prob_arm+temp)
        if ch==1:
            return pull_arm
        pred=get_op()[pull_arm]
        self.prob_arm[pull_arm]=(self.prob_arm[pull_arm]*self.pull_count[pull_arm]+pred)/(self.pull_count[pull_arm]+1)
        self.pull_count[pull_arm]+=1
        self.counter+=1
        return pred,pull_arm
    def update(self,pull_arm,pred):
        self.prob_arm[pull_arm]=(self.prob_arm[pull_arm]*self.pull_count[pull_arm]+pred)/(self.pull_count[pull_arm]+1)
        self.pull_count[pull_arm]+=1
        self.counter+=1