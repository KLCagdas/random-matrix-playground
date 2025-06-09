import numpy as np
from math import sqrt
from random_matrices import projector
from marchenko_pastur import mp_distribution, mp_plot

# number of variables
N = 1000
# number of observations
T = 2000
# ratio of num. of states to num. of observations
q = N/T

# non-symmetric random matrix
X = np.random.normal(0, 1, (T, N))
# NxN covariance
C_X = 1/T * X.T @ X
# compute the eigenvalues
eig_X = np.linalg.eigvalsh(C_X)

# Marcenko-Pastur bounds
lambda_p = (1 + sqrt(q))**2
lambda_m = (1 - sqrt(q))**2

# x values for MP distribution
x = np.linspace(lambda_m, lambda_p + 1, 1000)
# expected MP distribution of the eigenvalues
rho = mp_distribution(x, q, lambda_p, lambda_m)


# number of zero eigenvalues
nonzero_eig = eig_X[eig_X > 1e-10]
# check the number of zero eigenvalues
num_zero_eig = np.array(np.where(eig_X < 1e-10)).size

mp_plot(x, rho, lambda_m, lambda_p, nonzero_eig)

    
""" P, _ = projector(0.3, N)
Y = X @ P
C_Y = Y.T @ Y
eig_Y = np.linalg.eigvalsh(C_Y) """
