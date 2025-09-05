from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution_half, mp_plot
import numpy as np

# create a random matrix
X = RandomMatrix(2000, 1000)
# compute the eigvals of its covariance
X.covariance()

# project onto eigenbasis with half of the directions killed and return nonzero eigenvalues
nonzero_eig = X.half_projector(0.3)
# calculate the Marcenko-Pastur distribution
x, rho, lambda_m, lambda_p = mp_distribution_half(X.q)

# plot the MP and eigenvalue distribution
mp_plot(x, rho, lambda_m, lambda_p, nonzero_eig)

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
