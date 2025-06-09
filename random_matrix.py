import numpy as np
from math import sqrt
from random_matrices import projector
from marchenko_pastur import mp_distribution, mp_plot

class RandomMatrix():
    def __init__(self, N, T):
        # number of variables
        self.N = N
        # number of observations
        self.T = T
        # ratio of num. of states to num. of observations
        self.q = self.N / self.T
        # Marcenko-Pastur bounds
        self.lambda_p = (1 + sqrt(self.q))**2
        self.lambda_m = (1 - sqrt(self.q))**2
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


""" P, _ = projector(0.3, N)
Y = X @ P
C_Y = Y.T @ Y
eig_Y = np.linalg.eigvalsh(C_Y) """
