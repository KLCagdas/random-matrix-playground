import numpy as np
from math import sqrt

class RandomMatrix():
    def __init__(self, N, T):
        # number of variables
        self.N = N
        # number of observations
        self.T = T
        # ratio of num. of states to num. of observations
        self.q = self.N / self.T
        # non-symmetric random matrix
        self.X = np.random.normal(0, 1, (self.T, self.N))

    def covariance(self):
        # NxN covariance
        C_X = 1/self.T * self.X.T @ self.X
        # compute the eigenvalues
        eig_X = np.linalg.eigvalsh(C_X)
        # nonzero eigenvalues
        self.nonzero_eig = eig_X[eig_X > 1e-10]
        # number of zero eigenvalues
        self.num_zero_eig = np.array(np.where(eig_X < 1e-10)).size

    def wigner(self, std):
        # define a non-symmetric Gaussian matrix
        H = np.random.normal(loc=0.0, scale=std/np.sqrt(2*self.N), size=(self.N, self.N))

        # return a Wigner matrix
        return H + H.T
    
    def half_projector(self, std):
        # create a Wigner matrix
        W = self.wigner(std)
        # diagonalize W with eigh since W is symmetric
        eigval, V = np.linalg.eigh(W)
        # define Lambda to be diagonal matrix of eigenvalues
        L = np.sign(np.diag(eigval))
        # generate a random symmetric real orthogonal matrix 
        M = V @ L @ np.linalg.inv(V)
        # create a random projector
        P = 0.5 * (M + np.ones(self.N)) 

        return P, M
    
    """ def m_projector(self, var, m):
        self.q_p = m / self.T
        # create a Wigner matrix
        W = self.wigner(var)
        # diagonalize W 
        eigval, V = np.linalg.eigh(W)
        # sort eigenvalues in descending order and get indices
        idx = np.argsort(eigval)[::-1]
        # create a diagonal matrix with the first m diagonals as 1, others as 0
        L = np.zeros_like(eigval)
        L[idx[:m]] = 1
        # construct the projector matrix
        P = V @ np.diag(L) @ np.linalg.inv(V)

        return P """
    
    def m_projector(self, std, m):
        self.q_p = m / self.T
        # create a Wigner matrix
        W = self.wigner(std)
        # diagonalize W 
        eigval, V = np.linalg.eigh(W)
        # sort eigenvalues in descending order and get indices
        idx = np.argsort(eigval)[::-1]
        # create a diagonal matrix with the first m diagonals as 1, others as 0
        L = np.zeros_like(eigval)
        L[idx[:m]] = 1
        # construct the projector matrix
        P = V @ np.diag(L) @ np.linalg.inv(V)

        return P

