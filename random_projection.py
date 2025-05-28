import numpy as np
from math import sqrt

'''
parameters
'''
# number of states
N = 5
""" # number of observations
T = 1000
# ratio of num. of states to num. of observations
q = N/T """
# parameter for the variance of X and M
sigma_X = 0.5
sigma_M = 0.3

# define a non-symmetric Gaussian matrix
H = np.random.normal(loc=0.0, scale=sigma_X**2/N, size=(N, N))
# create a Wigner matrix
X = sigma_X * (H + H.T) / sqrt(2*N)

# define another non-symmetric Gaussian matrix
H = np.random.normal(loc=0.0, scale=sigma_M**2/N, size=(N, N))
# create another Wigner matrix
M = sigma_M * (H + H.T) / sqrt(2*N)
# diagonalize M
eigval, V = np.linalg.eig(M)
# define Lambda to be diagonal matrix of eigenvalues
L = np.sign(np.diag(eigval))
# reconstruct M by a similarity transformation
M = V @ L @ np.linalg.inv(V)

# create a projector
P = 0.5 * (M + np.ones(N)) 

# create a noisy version of M
E = M + X

print(np.dot(E, P))


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