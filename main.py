#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 02:09:54 2022

@author: kurup
"""

from Thomson import *
from eps_greedy import *
from UCB import *
from EXP3 import *
import numpy as np
import matplotlib.pyplot as plt

U=UCB()
T=Thompson()
EXP=EXP3(0.3)
Ep_1=Eps_greedy(1)
Ep_2=Eps_greedy(0.1)

objs=[U,T,EXP,Ep_1,Ep_2]


arm_out=[]
Real_out=[]

for i in range(100):
    all_arms=[]  # which all arms are the algos going to pull in order U,T,EXP,Ep_1,Ep_2
    for i in objs:
        all_arms.append(i.get_pred())
    #all_arms=[U.get_pred(),T.get_pred(),EXP.get_pred(),Ep_1.get_pred(),Ep_2.get_pred()]
    Real_pred=get_op()
    Real_out.append(Real_pred)
    all_preds=[] # rewardsarms got in the order U,T,EXP,Ep_1,Ep_2
    for i in all_arms:
        all_preds.append(Real_pred[i])
    arm_out.append(all_preds)
    for i in range(len(objs)):
        objs[i].update(pull_arm=all_arms[i], pred=all_preds[i])


cum_rew=[] #cumulative reward if pulled the arm at each time
for i in range(1,len(Real_out)+1):
    cum_rew.append(np.sum(Real_out[:i],axis=0))
cum_rew=np.array(cum_rew)
cum_arm_rew=[]
for i in range(1,len(arm_out)+1):
    cum_arm_rew.append(np.sum(arm_out[:i],axis=0))
cum_arm_rew=np.array(cum_arm_rew)

regrets=[]
for i in range(len(cum_arm_rew)):
    regrets.append(np.max(cum_rew[i])-cum_arm_rew[i])
regrets=np.array(regrets)

def plot_graph(regrets):
    L_C={
        0:['UCB','r'],
        1:['Thompson','g'],
        2:['EXP3','b'],
        3:['Eps:1','y'],
        4:["Eps:0.1",'c']}
    l=len(regrets)
    regrets=regrets.T
    for i in range(5):
        plt.plot(np.arange(l),regrets[i],label=L_C[i][0],color=L_C[i][1])
    plt.legend(loc='best')
    plt.show()
plot_graph(regrets)


