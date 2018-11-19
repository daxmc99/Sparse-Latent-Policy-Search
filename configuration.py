import numpy as np
from numpy.linalg import inv

best_size=10
sample_size=20
number_of_groups=2
#two degrees of freedom per group, knee and
#dimensions_per_group=np.array([7,7]) 
dimensions_per_group=np.array([2,2])
latent_dimension_size=6
orginial_feature_dimension_size=19
Time=50
max_iterations=300
max_inner_iterations=20
#don't touch 
sigma2_M=100
anti_convergence_factor=1.5
tauA=1000
tauB=1000
alphaA=1
alphaB=1
initial_Tau= inv(np.array([10]).reshape(1,-1)**1.5)[0][0]
initial_Alpha=1

load_the_latest_state = True ## Loads the checkpoint.npy
ip_address = '127.0.0.1' ## '192.168.125.1' - Yumi IP