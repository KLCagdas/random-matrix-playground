from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution_m, mp_plot
import numpy as np

# create a random matrix
X = RandomMatrix(1000, 2000)
# compute the eigvals of its covariance
X.covariance()

gamma, nonzero_eig_C = X.m_projector(0.4, 500)

x, rho, lambda_m, lambda_p = mp_distribution_m(X.q, X.q_p, gamma)

print(np.trapz(rho, x))

# plot the MP and eigenvalue distribution
mp_plot(x, rho, lambda_m, lambda_p, nonzero_eig_C)

