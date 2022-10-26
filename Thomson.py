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