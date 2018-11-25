import pdb
from numpy import *
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
import scipy.stats
import time
import math
from datetime import datetime
import traceback
from configuration import * 
from Environment import BipedalWalkerEnvironment
from sklearn import preprocessing 
from basis_model import PeriodicBasisModel

def get_samples(env,W=None,M=None,tau=None,Latent=None,dimensions_per_group=None,Time=None,rendering=0,*args,**kwargs):



    DoF=sum(dimensions_per_group)
    number_of_groups=dimensions_per_group.shape[0]    
    
    #periodic basis functions
    numOfBaFu = 20
    Basisfunctions = PeriodicBasisModel().get_basis_functions(Time,20)
    BasisDim = numOfBaFu


    Z=np.random.randn(Latent,BasisDim)
    Actions=np.zeros([DoF,Time])
    reward=np.zeros([1,Time])

    # ## simulator for OpenAI gym
    # env = BipedalWalkerEnvironment()
    # obs = env.reset()
    # ####
    for t in arange(0,Time).reshape(-1):
        for m in arange(0,number_of_groups).reshape(-1):
            
            try:
                startDim = sum(dimensions_per_group[:m])
                # pdb.set_trace() #due to domain error in arguements
                # inv([[tau[m][0]]])[0][0] + 2 should never be negative 
                # but can get there
                xx = scipy.stats.norm.rvs(0, inv([[tau[m][0]]])[0][0] + 2, (dimensions_per_group[m],BasisDim))
                xx = xx/2
                time.sleep(0.05)
                #debugger for bad dimensions, ensure orig_dim = 
                #pdb.set_trace()
                Actions[startDim:(startDim + dimensions_per_group[m]),t]=dot(dot(W[m][0],Z),Basisfunctions[:,t]) + dot(M[m][0],Basisfunctions[:,t]) + dot(xx,Basisfunctions[:,t])
            except ValueError:
                traceback.print_exc()
                print(tau)
                print( inv([[tau[m][0]]])[0][0] )
                pdb.set_trace()
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
        #rendering = 1
        state_t2,reward_t,terminal = env.step(state,action_norm,rendering)
        

        ## REWARD CALCULATION

        reward[0][t] = reward_t
        if terminal == 1:
            print("Done")
        
            
        
    #reward2=exp(- reward)

    #Open AI directly gives us a reward thereby we do not need to do this.
    #reward = dot(np.ones(reward2.shape),sum(reward2))

    #print(reward)
    Z=dot(Z,Basisfunctions)
    #env.close()
    return Actions,Basisfunctions,reward,Z
    
if __name__ == '__main__':
    pass
    
