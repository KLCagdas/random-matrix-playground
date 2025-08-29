from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution_m, mp_plot
import numpy as np

# create a random matrix
X = RandomMatrix(2000, 1000)
# compute the eigvals of its covariance
X.covariance()

# variance of the random projector
sigma_m = 0.4
# number of directions to project onto
m = 100
# create a random projector
P = X.m_projector(sigma_m, m)
# take a random projection of X
Y = X.X @ P
# define the Wishart ensemble 
C = 1/X.T * Y.T @ Y
# compute the eigenvalues of the covariance
eig_C = np.linalg.eigvalsh(C)
# Filter out eigenvalues that are effectively zero
nonzero_eig_C = eig_C[np.abs(eig_C) > 1e-12]

x, rho, lambda_m, lambda_p = mp_distribution_m(X.q, X.q_p, m/X.N)

print(np.trapz(rho, x))

# plot the MP and eigenvalue distribution
mp_plot(x, rho, lambda_m, lambda_p, nonzero_eig_C)

