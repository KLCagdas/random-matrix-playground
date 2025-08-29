from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution_half, mp_plot
import numpy as np

# create a random matrix
X = RandomMatrix(2000, 1000)
# compute the eigvals of its covariance
X.covariance()
# calculate the Marcenko-Pastur distribution
""" x, rho = mp_distribution(X.q, X.lambda_p, X.lambda_m)
mp_plot(x, rho, X.lambda_m, X.lambda_p, X.nonzero_eig) """

# std of the random projector
sigma_M = 0.3
# create a random projector
P, _ = X.half_projector(sigma_M)
# take a random projection of X
Y = X.X @ P
# compute the covariance of the projection
# std of Y is 0.5 since half of the directions are killed
C_Y = 1/(X.T * 0.5**2) * Y.T @ Y
# compute the eigenvalues of the covariance
eig_Y = np.linalg.eigvalsh(C_Y)
# Filter out eigenvalues that are effectively zero
nonzero_eig = eig_Y[np.abs(eig_Y) > 1e-10]

x, rho, lambda_m, lambda_p = mp_distribution_half(X.q)

# plot the MP and eigenvalue distribution
mp_plot(x, rho, lambda_m, lambda_p, nonzero_eig)

'''
# RANDOM PROJECTION

# parameter for the variance of W and M
sigma_W = 0.5
sigma_M = 0.3

# define a Wigner matrix
W = X.wigner(sigma_W)
# define another Wigner matrix
P, M = X.projector(sigma_M)
# create a noisy version of M 
E = M + W
# TAKE PROJECTION OF EIGVECTORS OF E  
_, v_E = np.linalg.eigh(E)
print(P @ v_E)
'''

'''
# CHECKS

# check if cols are orthogonal
dot = np.dot(M[:, 0], M[:, 1])
t=0
for i in range(N):
    t += np.dot(V[:, 2], V[:, i])
print(t)

# check if M depends on sigma (it shouldn't)
np.random.seed(1) # put this to the first line
print(M) 
'''
