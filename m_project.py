from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution, mp_plot
import numpy as np

# create a random matrix
X = RandomMatrix(1000, 2000)
# compute the eigvals of its covariance
X.covariance()
# project X onto eigenbasis with m < N directions kept and return gamma and nonzero eigenvalues
gamma, nonzero_eig_C = X.m_projector(0.4, 500)
# calculate the Marcenko-Pastur distribution
x, rho, lambda_m, lambda_p = mp_distribution(X.q, X.q_p, gamma)

# plot the MP and eigenvalue distribution
mp_plot(x, rho, lambda_m, lambda_p, nonzero_eig_C)