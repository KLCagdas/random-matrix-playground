from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution, mp_plot
import numpy as np

# create a random matrix
X = RandomMatrix(2000, 1000)
# compute the eigvals of its covariance
X.covariance()

# variance of the random projector
sigma_m = 0.3
# create a random projector
P = X.m_projector(sigma_m, 100)
# take a random projection of X
Y = X.X @ P
# define the Wishart ensemble 
C = Y.T @ Y
# check if C is symmetric
print(np.allclose(C, C.T))  # This will print True if C is symmetric
# compute the eigenvalues of the covariance
eig_C = np.linalg.eigvalsh(C)