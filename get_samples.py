import pdb
from numpy import *
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
import scipy.stats
import time
import math
from datetime import datetime
import socket
from configuration import * 
from Environment import BipedalWalkerEnvironment
from sklearn import preprocessing 

def get_samples(W=None,M=None,tau=None,Latent=None,dimensions_per_group=None,Time=None,rendering=0,*args,**kwargs):

    DoF=sum(dimensions_per_group)
    number_of_groups=dimensions_per_group.shape[0]    
    
    means = np.arange(-3,Time+3,3)
    BasisDim = len(means)
    variance = 3
    bafu = scipy.stats.norm.pdf(1,means,variance)
    bafu = bafu.reshape(-1,1)
    
    for i in range(1,Time):
        function_i = scipy.stats.norm.pdf(i+1,means,variance)
        function_i = function_i.reshape(-1,1)
        bafu = np.concatenate((bafu,function_i),1)
    
    
    sum_bafu = np.sum(bafu,0)
    sum_bafu = sum_bafu.reshape(1,-1)
    sum_bafu = np.repeat(sum_bafu,len(means),0)
    Basisfunctions = np.divide(bafu,sum_bafu)
    
    Z=np.random.randn(Latent,BasisDim)
    Actions=np.zeros([DoF,Time])
    reward=np.zeros([1,Time])

    ## simulator for OpenAI gym
    env = BipedalWalkerEnvironment()
    obs = env.reset()
    ####
    for t in arange(0,Time).reshape(-1):
        for m in arange(0,number_of_groups).reshape(-1):
            
            startDim = sum(dimensions_per_group[:m])
            pdb.set_trace()
            xx = scipy.stats.norm.rvs(0, inv([[tau[m][0]]])[0][0] + 2, (dimensions_per_group[m],BasisDim))
            xx = xx/2
            time.sleep(0.05)
            #pdb.set_trace()
            Actions[startDim:(startDim + dimensions_per_group[m]),t]=dot(dot(W[m][0],Z),Basisfunctions[:,t]) + dot(M[m][0],Basisfunctions[:,t]) + dot(xx,Basisfunctions[:,t])
            # ValueError: shapes (2,6) and (19,) not aligned: 6 (dim 1) != 19 (dim 0)

            # Basisfunction: (19,50)

            # !dot(M[m][0],Basisfunctions[:,t])
            # *** ValueError: shapes (2,6) and (19,) not aligned: 6 (dim 1) != 19 (dim 0)
            # when Time=10
            #   array([0., 0.])
            
            # This value M[m][0].shape() = (2,original_feature_dimension)       
        
        CurrentAngle=Actions[:,t]
        
        #print(CurrentAngle)
        #the below code is for interacting with the simulator
        
        
        #normalize values before sending them in as actions
        # no idea if if this works
        action_norm = preprocessing.normalize([CurrentAngle])
        action_norm = action_norm.reshape(4)

        #print(action_norm)
        
        state = 0
        state_t2,reward_t,terminal = env.step(state,action_norm,rendering)
        

        ## REWARD CALCULATION

        reward[0][t] = reward_t
        if terminal == 1:
            print("Done")
            break
            
        
    #reward2=exp(- reward)

    #Open AI directly gives us a reward thereby we do not need to do this.
    #reward = dot(np.ones(reward2.shape),sum(reward2))

    Z=dot(Z,Basisfunctions)
    env.close()
    return Actions,Basisfunctions,reward,Z
    
if __name__ == '__main__':
    pass
    
