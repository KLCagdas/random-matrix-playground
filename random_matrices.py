import numpy as np
from math import sqrt

''''
THIS WHOLE FILE CAN BE ONE CLASS
SINCE THERE ARE NESTED FUNCTIONS
'''

def wigner(var, N):
    # define a non-symmetric Gaussian matrix
    H = np.random.normal(loc=0.0, scale=var**2/N, size=(N, N))
    # create a Wigner matrix
    X = var * (H + H.T) / sqrt(2*N)
    
    return X

def projector(var, N):
    # create a Wigner matrix
    W = wigner(var, N)
    # diagonalize W
    eigval, V = np.linalg.eig(W)
    # define Lambda to be diagonal matrix of eigenvalues
    L = np.sign(np.diag(eigval))
    # generate a random symmetric real orthogonal matrix by a similarity transformation
    M = V @ L @ np.linalg.inv(V)
    # create a random projector
    P = 0.5 * (M + np.ones(N)) 

    return P, M