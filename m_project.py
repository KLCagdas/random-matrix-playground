
from random_matrix import RandomMatrix
from marchenko_pastur import mp_distribution, mp_plot_multiple
import numpy as np
import matplotlib.pyplot as plt

# create a random matrix
X = RandomMatrix(1000, 2000)
# compute the eigvals of its covariance
X.covariance()
# calculate the Marcenko-Pastur distribution of the Wishart matrix (before projection)
x, rho, lambda_low, lambda_upp = mp_distribution(X.q)

# List of m values to try
m_values = [100, 300, 500, 700, 900]
for m in m_values:
    gamma, nonzero_eig_m = X.m_projector(0.4, m)
    x_m, rho_m, lambda_low_m, lambda_upper_m = mp_distribution(X.q, X.q_p, gamma)
    mp_plot_multiple(
        x, rho, lambda_low, lambda_upp,
        x_m, rho_m, lambda_low_m, lambda_upper_m,
        X.nonzero_eig, nonzero_eig_m,
        m
    )