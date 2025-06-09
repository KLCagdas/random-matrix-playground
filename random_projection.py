import numpy as np
from math import sqrt
from random_matrices import projector
from random_matrices import wigner

# number of states
N = 1000

# parameter for the variance of X and M
sigma_X = 0.5
sigma_M = 0.3

# define a Wigner matrix
X = wigner(sigma_X, N)
# define another Wigner matrix
_, M = projector(sigma_M, N)
# create a noisy version of M 
E = M + X
# TAKE PROJECTION OF EIGVALS OF E  



""" '''
checks
'''
# check if cols are orthogonal
dot = np.dot(M[:, 0], M[:, 1])
t=0
for i in range(N):
    t += np.dot(V[:, 2], V[:, i])
print(t)

# check if M depends on sigma (it shouldn't)
np.random.seed(1) # put this to the first line
print(M)
 """